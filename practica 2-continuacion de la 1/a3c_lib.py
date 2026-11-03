"""A3C (Asynchronous Advantage Actor-Critic) para el entorno de trading de la práctica 1.

Va en un .py y no dentro de a3c.ipynb porque los trabajadores de A3C son procesos de Python (en Windows se
crean con `spawn`) y cada uno tiene que poder importar la función que ejecuta; desde una libreta no pueden.

Qué hay aquí:
- `calibrar`: media y desviación de las observaciones y de la recompensa (las mismas escalas que en DQN).
- `ActorCritic`: una red con cuerpo compartido, una cabeza de política (3 logits) y una de valor V(s).
- `train_a3c`: lanza varios procesos que juegan episodios con su copia local de la red, calculan los
  gradientes de una ventana de n pasos y los aplican a los parámetros globales, que viven en memoria
  compartida y se actualizan sin bloqueos (estilo "Hogwild", como en el artículo de A3C).
- `PoliticaA3C`: la política de una red ya entrenada (greedy o estocástica), con la misma interfaz que
  usa `evaluate` de trader_rl.py.
"""

import sys
import time
import threading
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# trader_rl.py (práctica 1) está en la carpeta superior
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trader_rl import SimpleTradingEnv, ExtendedTradingEnv, make_series, SEED_TRAIN, SEED_CALIB  # noqa: E402

ENVS = {"SimpleTradingEnv": SimpleTradingEnv, "ExtendedTradingEnv": ExtendedTradingEnv}
_INIT_LOCK = threading.Lock()

# Acciones con efecto en cada posición: sin activo {mantener, comprar}, con activo {vender, mantener}
VALIDAS = torch.tensor([[False, True, True],
                        [True, True, False]])
ACCIONES_VALIDAS = ((1, 2), (0, 1))


def calibrar(gen, env_name="SimpleTradingEnv", n_series=40):
    """Media y desviación de cada componente de la observación y desviación de la recompensa por paso,
    con las series de calibración (semillas disjuntas de las de entrenamiento, validación y test)."""
    env_cls = ENVS[env_name]
    obs, rets = [], []
    for i in range(n_series):
        p = make_series(gen, SEED_CALIB + i)
        env = env_cls(p)
        o, _ = env.reset()
        done = False
        while not done:
            obs.append(o)
            o, _, done, _, _ = env.step(1)
        rets.append(p[1:] / p[:-1] - 1)
    obs = np.array(obs, dtype=np.float64)
    mu, sd = obs.mean(0), obs.std(0)
    mu[1], sd[1] = 0.0, 1.0                          # la posición (0/1) no se normaliza
    return mu.astype(np.float32), sd.astype(np.float32), float(np.std(np.concatenate(rets)))


class ActorCritic(nn.Module):
    """Cuerpo compartido (2 capas ocultas ReLU) y dos cabezas: logits de la política π(a|s) y valor V(s)."""

    def __init__(self, n_in=2, hidden=64):
        super().__init__()
        self.cuerpo = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, 3)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)    # política inicial casi uniforme: se empieza explorando
        nn.init.zeros_(self.pi.bias)

    def forward(self, x):
        h = self.cuerpo(x)
        return self.pi(h), self.v(h).squeeze(-1)


# --- parámetros planos: la red global vive en un vector de memoria compartida ---------------------

def n_params(net):
    return sum(p.numel() for p in net.parameters())


def get_flat(net):
    return np.concatenate([p.detach().numpy().ravel() for p in net.parameters()]).astype(np.float32)


def bind_flat(net, flat):
    """Hace que los parámetros de `net` sean vistas del vector `flat` (comparten memoria con él)."""
    o = 0
    for p in net.parameters():
        n = p.numel()
        p.data = torch.from_numpy(flat[o:o + n].reshape(tuple(p.shape)))
        o += n


def load_flat(net, flat):
    """Copia el vector `flat` dentro de los parámetros de `net`, sin cambiar su memoria."""
    o = 0
    for p in net.parameters():
        n = p.numel()
        p.data.copy_(torch.from_numpy(flat[o:o + n]).view(tuple(p.shape)))
        o += n


def capas_numpy(net):
    """Vistas numpy de los pesos (comparten memoria con la red): sirven para actuar sin llamar a PyTorch."""
    L = [(m.weight.detach().numpy(), m.bias.detach().numpy()) for m in net.cuerpo if isinstance(m, nn.Linear)]
    return L + [(net.pi.weight.detach().numpy(), net.pi.bias.detach().numpy())]


def logits_numpy(x, capas):
    h = np.maximum(capas[0][0] @ x + capas[0][1], 0)
    h = np.maximum(capas[1][0] @ h + capas[1][1], 0)
    return capas[2][0] @ h + capas[2][1]


# --- trabajador -----------------------------------------------------------------------------------

def _worker(wid, cfg, flat_buf, counter, snaps_buf, snap_eps, times):
    torch.set_num_threads(1)
    hp, n_ep, seed = cfg["hp"], cfg["n_ep"], cfg["seed"]
    mu, sd, s_rew = cfg["mu"], cfg["sd"], cfg["s_rew"]
    env_cls = ENVS[cfg["env"]]
    rng = np.random.default_rng(seed * 1000 + wid)

    flat = np.frombuffer(flat_buf, dtype=np.float32)
    snaps = np.frombuffer(snaps_buf, dtype=np.float32).reshape(-1, flat.size)
    eps_snap = np.frombuffer(snap_eps, dtype=np.float64)

    gnet = ActorCritic(len(mu), hp["hidden"])         # red global: sus parámetros son vistas de la memoria compartida
    bind_flat(gnet, flat)
    lnet = ActorCritic(len(mu), hp["hidden"])         # copia local de este trabajador
    capas = capas_numpy(lnet)
    opt = torch.optim.Adam(gnet.parameters(), lr=hp["lr"])   # cada trabajador lleva sus propios momentos de Adam

    gamma, n_steps, vcoef, clip = hp["gamma"], hp["n_steps"], hp.get("vcoef", 0.5), hp.get("clip", 1.0)
    primero = True
    while True:
        with counter.get_lock():
            ep = counter.value
            counter.value += 1
        if ep >= n_ep:
            break
        if primero:
            times[2 * wid] = time.time()
            primero = False
        if ep > 0 and ep % cfg["eval_every"] == 0:    # instantánea de los pesos globales, que se evaluará después
            i = ep // cfg["eval_every"]
            snaps[i] = flat
            eps_snap[i] = ep

        frac = ep / n_ep
        for g in opt.param_groups:
            g["lr"] = hp["lr"] * hp["lr_final"] ** frac
        beta = hp["beta"] * hp["beta_final"] ** frac

        load_flat(lnet, flat)
        env = env_cls(make_series(cfg["gen"], SEED_TRAIN + 10_000 * seed + ep))
        obs, _ = env.reset()
        x = np.clip((obs - mu) / sd, -5, 5).astype(np.float32)
        fin = False
        while not fin:
            S, A, R = [], [], []
            while not fin and len(S) < n_steps:
                acts = ACCIONES_VALIDAS[int(x[1] > 0.5)]
                l = logits_numpy(x, capas)
                p2 = 1.0 / (1.0 + np.exp(np.clip(l[acts[0]] - l[acts[1]], -30, 30)))   # prob. de la segunda acción válida
                a = acts[1] if rng.random() < p2 else acts[0]
                obs, r, fin, _, _ = env.step(a)
                S.append(x); A.append(a); R.append(r / s_rew)
                x = np.clip((obs - mu) / sd, -5, 5).astype(np.float32)

            # retorno de n pasos con bootstrapping: G_t = r_t + γ G_{t+1}, con G_{T} = V(s_T) (0 si el episodio acabó)
            with torch.no_grad():
                G = 0.0 if fin else float(lnet(torch.from_numpy(x)[None])[1])
            Gs = np.empty(len(R), dtype=np.float32)
            for t in range(len(R) - 1, -1, -1):
                G = R[t] + gamma * G
                Gs[t] = G

            S_t, A_t, G_t = torch.from_numpy(np.array(S)), torch.from_numpy(np.array(A)), torch.from_numpy(Gs)
            logits, v = lnet(S_t)
            valido = VALIDAS[(S_t[:, 1] > 0.5).long()]
            logp = F.log_softmax(logits.masked_fill(~valido, -1e9), dim=1)
            adv = G_t - v
            ent = -torch.where(valido, logp.exp() * logp, torch.zeros_like(logp)).sum(1)
            loss = (-(logp.gather(1, A_t[:, None]).squeeze(1) * adv.detach()).mean()     # actor (con ventaja)
                    + vcoef * adv.pow(2).mean()                                          # crítico
                    - beta * ent.mean())                                                 # entropía: exploración
            lnet.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(lnet.parameters(), clip)
            for gp, lp in zip(gnet.parameters(), lnet.parameters()):
                gp.grad = lp.grad.clone()
            opt.step()                                # actualiza la red global (memoria compartida) sin bloqueos
            load_flat(lnet, flat)                     # y se trae la versión más reciente
    times[2 * wid + 1] = time.time()


# --- entrenamiento --------------------------------------------------------------------------------

def train_a3c(gen, hp, n_ep, seed, workers=4, eval_every=50, env_name="SimpleTradingEnv", escalas=None):
    """Entrena un A3C con `workers` procesos y devuelve las instantáneas de la red global cada `eval_every`
    episodios (para evaluarlas después, en paralelo) y el tiempo de reloj del entrenamiento.

    `n_ep` es el nº total de episodios entre todos los trabajadores, así que cada uno juega unos n_ep / workers.
    """
    ctx = mp.get_context("spawn")
    mu, sd, s_rew = escalas if escalas else calibrar(gen, env_name)
    with _INIT_LOCK:                                  # la semilla de PyTorch es global: sin el cerrojo, dos entrenamientos en hilos distintos se pisarían
        torch.manual_seed(seed)
        net = ActorCritic(len(mu), hp["hidden"])
    P = n_params(net)
    n_snap = n_ep // eval_every + 2
    flat_buf = ctx.RawArray("f", P)
    snaps_buf = ctx.RawArray("f", n_snap * P)
    snap_eps = ctx.RawArray("d", n_snap)
    flat = np.frombuffer(flat_buf, dtype=np.float32)
    snaps = np.frombuffer(snaps_buf, dtype=np.float32).reshape(n_snap, P)
    eps_snap = np.frombuffer(snap_eps, dtype=np.float64)
    flat[:] = get_flat(net)
    eps_snap[:] = -1.0
    snaps[0], eps_snap[0] = flat, 0                   # red sin entrenar
    counter = ctx.Value("i", 0)
    times = ctx.RawArray("d", 2 * workers)

    cfg = {"gen": gen, "hp": hp, "n_ep": n_ep, "seed": seed, "mu": mu, "sd": sd, "s_rew": s_rew,
           "env": env_name, "eval_every": eval_every}
    procs = [ctx.Process(target=_worker, args=(w, cfg, flat_buf, counter, snaps_buf, snap_eps, times)) for w in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if any(p.exitcode != 0 for p in procs):
        raise RuntimeError(f"un trabajador de A3C ha fallado: {[p.exitcode for p in procs]}")
    snaps[n_snap - 1], eps_snap[n_snap - 1] = flat, n_ep      # red final
    t = np.frombuffer(times, dtype=np.float64)
    ok = eps_snap >= 0
    return {"gen": gen, "env": env_name, "seed": seed, "workers": workers, "hp": hp, "n_ep": n_ep,
            "mu": mu, "sd": sd, "s_rew": s_rew, "snap_ep": eps_snap[ok].astype(int), "snaps": snaps[ok].copy(),
            "train_time": float(t[1::2].max() - t[0::2].min())}


# --- política de una red ya entrenada -------------------------------------------------------------

class PoliticaA3C:
    """Política de una red entrenada. Por defecto es greedy (la acción válida con mayor logit; si empatan, mantener);
    con `estocastica=True` muestrea de π, que es la política con la que se ha entrenado."""

    def __init__(self, flat, n_in, hidden, mu, sd, estocastica=False, seed=0):
        net = ActorCritic(n_in, hidden)
        load_flat(net, flat)
        self.capas = [(W.copy(), b.copy()) for W, b in capas_numpy(net)]
        self.mu, self.sd, self.estocastica = mu, sd, estocastica
        self.rng = np.random.default_rng(seed)

    def logits(self, obs):
        x = np.clip((np.asarray(obs, dtype=np.float32) - self.mu) / self.sd, -5, 5)
        return logits_numpy(x, self.capas)

    def __call__(self, obs):
        l = self.logits(obs)
        acts = ACCIONES_VALIDAS[int(obs[1] > 0.5)]
        if self.estocastica:
            p2 = 1.0 / (1.0 + np.exp(np.clip(l[acts[0]] - l[acts[1]], -30, 30)))
            return acts[1] if self.rng.random() < p2 else acts[0]
        if obs[1] < 0.5:
            return 2 if l[2] > l[1] else 1
        return 0 if l[0] > l[1] else 1


def politica_de(run, i, estocastica=False):
    return PoliticaA3C(run["snaps"][i], len(run["mu"]), run["hp"]["hidden"], run["mu"], run["sd"], estocastica)
