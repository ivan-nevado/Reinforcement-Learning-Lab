"""REINFORCE (gradiente de la política, Monte Carlo) para el entorno de trading de la práctica 1.

Va en un .py para que joblib pueda lanzar los entrenamientos en procesos separados en Windows.

Reutiliza de a3c_lib.py (sin modificarlo) la calibración de escalas, la red y la política, así que
REINFORCE y A3C tienen exactamente la misma arquitectura y se evalúan con el mismo código. La única
diferencia es cómo se calcula el gradiente:

- REINFORCE juega el episodio entero con su política estocástica y, al acabar, sube la probabilidad de
  cada acción en proporción a lo que se ganó desde ese paso hasta el final (el retorno G_t).
- Con baseline, a G_t se le resta V(s_t), que aprende una segunda cabeza de la red. No cambia el gradiente
  esperado, pero reduce su varianza. Es la idea que lleva a los métodos actor-crítico como A3C.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a3c_lib import (ENVS, VALIDAS, ACCIONES_VALIDAS, ActorCritic, calibrar, capas_numpy, logits_numpy,
                     get_flat, PoliticaA3C, politica_de)  # noqa: F401  (politica_de se usa desde la libreta)
from trader_rl import make_series, SEED_TRAIN


def jugar_episodio(env, capas, mu, sd, s_rew, rng):
    """Juega un episodio completo con la política estocástica. Devuelve estados normalizados, acciones y
    recompensas reescaladas."""
    obs, _ = env.reset()
    x = np.clip((obs - mu) / sd, -5, 5).astype(np.float32)
    S, A, R, fin = [], [], [], False
    while not fin:
        acts = ACCIONES_VALIDAS[int(x[1] > 0.5)]
        l = logits_numpy(x, capas)
        p2 = 1.0 / (1.0 + np.exp(np.clip(l[acts[0]] - l[acts[1]], -30, 30)))   # prob. de la segunda acción válida
        a = acts[1] if rng.random() < p2 else acts[0]
        obs, r, fin, _, _ = env.step(a)
        S.append(x); A.append(a); R.append(r / s_rew)
        x = np.clip((obs - mu) / sd, -5, 5).astype(np.float32)
    return np.array(S), np.array(A), np.array(R, dtype=np.float32)


def retornos(R, gamma):
    """G_t = R_{t+1} + γ G_{t+1}, calculado hacia atrás desde el final del episodio."""
    G, out = 0.0, np.empty(len(R), dtype=np.float32)
    for t in range(len(R) - 1, -1, -1):
        G = R[t] + gamma * G
        out[t] = G
    return out


def perdida(net, S, A, G, hp, beta, gamma_t=False):
    """Pérdida de un episodio. Su gradiente es el de REINFORCE (con o sin baseline) más la entropía."""
    S_t, A_t, G_t = torch.from_numpy(S), torch.from_numpy(A), torch.from_numpy(G)
    logits, v = net(S_t)
    valido = VALIDAS[(S_t[:, 1] > 0.5).long()]
    logp = F.log_softmax(logits.masked_fill(~valido, -1e9), dim=1)
    logp_a = logp.gather(1, A_t[:, None]).squeeze(1)
    peso = G_t - v.detach() if hp["baseline"] else G_t          # G_t - b(s_t), con b = V(s_t) o b = 0
    if gamma_t:                                                 # versión literal de las diapositivas: α·γ^t·G_t·∇ln π
        peso = peso * torch.from_numpy(hp["gamma"] ** np.arange(len(G), dtype=np.float32))
    ent = -torch.where(valido, logp.exp() * logp, torch.zeros_like(logp)).sum(1)
    loss = -(logp_a * peso).mean() - beta * ent.mean()
    if hp["baseline"]:
        loss = loss + hp.get("vcoef", 0.5) * (G_t - v).pow(2).mean()   # la cabeza V(s) aprende a predecir G_t
    return loss


def train_reinforce(gen, hp, n_ep, seed, eval_every=100, env_name="SimpleTradingEnv", escalas=None, gamma_t=False):
    """Entrena REINFORCE durante `n_ep` episodios, cada uno con una serie nueva del generador.

    Guarda instantáneas de los pesos cada `eval_every` episodios (se evalúan después, en paralelo) y devuelve
    lo mismo que train_a3c, para poder usar las mismas funciones de evaluación.
    """
    torch.set_num_threads(1)
    env_cls = ENVS[env_name]
    mu, sd, s_rew = escalas if escalas else calibrar(gen, env_name)
    torch.manual_seed(seed)
    net = ActorCritic(len(mu), hp["hidden"])
    capas = capas_numpy(net)                   # vistas de los pesos: siempre al día, para actuar sin PyTorch
    opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    rng = np.random.default_rng(seed)

    snaps, snap_ep = [get_flat(net)], [0]
    t0 = time.perf_counter()
    for ep in range(n_ep):
        frac = ep / n_ep
        for g in opt.param_groups:
            g["lr"] = hp["lr"] * hp["lr_final"] ** frac
        beta = hp["beta"] * hp["beta_final"] ** frac

        env = env_cls(make_series(gen, SEED_TRAIN + 10_000 * seed + ep))
        S, A, R = jugar_episodio(env, capas, mu, sd, s_rew, rng)
        G = retornos(R, hp["gamma"])
        loss = perdida(net, S, A, G, hp, beta, gamma_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), hp.get("clip", 1.0))
        opt.step()

        if (ep + 1) % eval_every == 0:
            snaps.append(get_flat(net))
            snap_ep.append(ep + 1)
    train_time = time.perf_counter() - t0
    if snap_ep[-1] != n_ep:
        snaps.append(get_flat(net))
        snap_ep.append(n_ep)
    return {"gen": gen, "env": env_name, "seed": seed, "workers": 1, "hp": hp, "n_ep": n_ep, "gamma_t": gamma_t,
            "mu": mu, "sd": sd, "s_rew": s_rew, "snap_ep": np.array(snap_ep), "snaps": np.array(snaps),
            "train_time": train_time}


def varianza_gradiente(run, gen, n_episodios=60, seed=12345, env_name="SimpleTradingEnv"):
    """Varianza del estimador del gradiente de la política (sin entropía) con la red final de `run`,
    calculado con y sin baseline sobre los MISMOS episodios. Devuelve la traza de la covarianza de cada uno
    y la norma del gradiente medio (la señal), para comparar ruido frente a señal."""
    torch.set_num_threads(1)
    hp = dict(run["hp"])
    net = ActorCritic(len(run["mu"]), hp["hidden"])
    from a3c_lib import load_flat
    load_flat(net, run["snaps"][-1])
    capas = capas_numpy(net)
    rng = np.random.default_rng(seed)
    params = [p for n, p in net.named_parameters() if not n.startswith("v.")]
    grads = {False: [], True: []}
    for ep in range(n_episodios):
        env = ENVS[env_name](make_series(gen, 700_000 + ep))
        S, A, R = jugar_episodio(env, capas, run["mu"], run["sd"], run["s_rew"], rng)
        G = retornos(R, hp["gamma"])
        for b in (False, True):
            loss = perdida(net, S, A, G, {**hp, "baseline": b, "vcoef": 0.0}, beta=0.0)
            g = torch.autograd.grad(loss, params, allow_unused=True)
            grads[b].append(torch.cat([x.ravel() for x in g if x is not None]).numpy().copy())
    out = {}
    for b, gs in grads.items():
        gs = np.array(gs)
        out[b] = {"varianza": float(gs.var(0).sum()), "señal": float(np.linalg.norm(gs.mean(0)))}
    return out
