"""Código compartido por monte_carlo.ipynb y comparativa.ipynb.

Al juntar los notebooks vimos que Q-Learning y SARSA se evaluaban con series de test distintas,
así que aquí está todo lo necesario para comparar los tres algoritmos con las mismas reglas:

- El entorno y los generadores del enunciado, sin tocar.
- El protocolo de evaluación de q_learning.ipynb (semillas, discretización, métricas, baselines y
  criterio de convergencia), que es el que hemos tomado como referencia.
- Los agentes. Q-Learning y SARSA están copiados de q_learning.ipynb y taller_trader_sarsa.ipynb;
  el de Q-Learning hace falta además para poder cargar los agentes guardados en cache_qlearning/.
  Monte Carlo es la parte nueva.

Va en un .py y no en el notebook porque joblib y las cachés necesitan importar las clases desde
cualquier sitio.
"""

import bisect
import random
import time

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

# ---------------------------------------------------------------------------------------------
# 1. Entorno y generadores (idénticos a taller_trader.ipynb)
# ---------------------------------------------------------------------------------------------


class SimpleTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, prices):
        super(SimpleTradingEnv, self).__init__()
        self.prices = prices
        self.max_steps = len(prices) - 1

        # Acciones: 0 = Vender, 1 = Mantener, 2 = Comprar
        self.action_space = spaces.Discrete(3)

        # Estado: [tendencia_precio (-1 a 1), posicion_actual (0 o 1)]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0], dtype=np.float32),
            high=np.array([1.0, 1], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 5  # Empezar con suficiente historia para la media
        self.position = 0      # 0: sin activo, 1: con activo

        # Gestión del capital
        self.initial_balance = 1000.0
        self.balance = self.initial_balance  # Valor total del portafolio (Efectivo + Activos)
        self.cash = self.initial_balance     # Dinero en efectivo disponible
        self.shares = 0.0                    # Cantidad de acciones poseídas

        return self._get_obs(), {}

    def _get_obs(self):
        # Calcular tendencia simple basada en los últimos 5 pasos
        window = self.prices[self.current_step - 5 : self.current_step + 1]
        trend = (window[-1] - np.mean(window)) / np.mean(window)
        return np.array([np.clip(trend, -1.0, 1.0), self.position], dtype=np.float32)

    def step(self, action):
        prev_price = self.prices[self.current_step]
        self.current_step += 1
        curr_price = self.prices[self.current_step]

        reward = 0.0
        cost_rate = 0.001  # Comisión del 0.1% por transacción

        # 1. Ejecutar acciones y actualizar efectivo/acciones
        if action == 2 and self.position == 0:  # Comprar
            # Usar todo el efectivo disponible para comprar acciones al precio anterior
            commission = self.cash * cost_rate
            effective_cash = self.cash - commission
            self.shares = effective_cash / prev_price
            self.cash = 0.0
            self.position = 1
            reward -= cost_rate

        elif action == 0 and self.position == 1:  # Vender
            # Vender todas las acciones al precio anterior
            sale_value = self.shares * prev_price
            commission = sale_value * cost_rate
            self.cash = sale_value - commission
            self.shares = 0.0
            self.position = 0
            reward -= cost_rate

        # 2. Calcular recompensa por variación del precio si mantiene la posición
        if self.position == 1:
            price_change = (curr_price - prev_price) / prev_price
            reward += price_change

        # 3. Actualizar el balance total (Net Worth) al finalizar el paso
        if self.position == 1:
            self.balance = self.cash + (self.shares * curr_price)
        else:
            self.balance = self.cash

        terminated = self.current_step >= self.max_steps
        truncated = False

        # Opcional: puedes pasar el balance actual en el diccionario de info
        info = {"balance": self.balance}

        return self._get_obs(), reward, terminated, truncated, info


class ExtendedTradingEnv(SimpleTradingEnv):
    """Estado ampliado de q_learning.ipynb (sección 10.1). Mismo entorno, pero la observación lleva
    también la volatilidad reciente y la distancia a la media de 20 pasos."""

    def __init__(self, prices):
        super().__init__(prices)
        self.observation_space = spaces.Box(low=np.array([-1, 0, 0, -1], dtype=np.float32),
                                            high=np.array([1, 1, np.inf, 1], dtype=np.float32), dtype=np.float32)

    def _get_obs(self):
        base = super()._get_obs()
        k = self.current_step
        log_rets = np.diff(np.log(self.prices[max(0, k - 10):k + 1]))
        hist = self.prices[max(0, k - 19):k + 1]
        dev = np.clip((hist[-1] - hist.mean()) / hist.mean(), -1, 1)
        return np.array([base[0], base[1], log_rets.std(), dev], dtype=np.float32)


EXT_BINS = {0: 5, 2: 4, 3: 5}


def generar_precios(n_steps=1000):
    # Combinar una tendencia suave (drift) + componente cíclico + ruido
    t = np.linspace(0, 50, n_steps)
    drift = 0.0005 * t
    ciclo = np.sin(t) * 0.05
    ruido = np.random.normal(0, 0.1, n_steps)

    # Precio base acumulado
    retornos = drift + ciclo + ruido
    precios = 100 * np.cumprod(1 + retornos)
    return precios


def generar_precios_browniano(n_steps=1000, start_price=100.0, mu=0.0001, sigma=0.02):
    dt = 1.0
    retornos = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_steps)
    precios = start_price * np.cumprod(np.exp(retornos))
    return precios


def generar_precios_reversion_media(n_steps=1000, start_price=100.0, mean_price=100.0, kappa=0.05, volatility=0.02):
    precios = np.zeros(n_steps)
    precios[0] = start_price
    for t in range(1, n_steps):
        delta = kappa * (mean_price - precios[t-1]) + np.random.normal(0, volatility * mean_price)
        precios[t] = precios[t-1] + delta
        precios[t] = max(precios[t], 0.1)
    return precios


def generar_precios_regimenes(n_steps=1000, start_price=100.0):
    precios = np.zeros(n_steps)
    precios[0] = start_price
    estado = 0
    retornos = []
    for _ in range(n_steps):
        if np.random.rand() < 0.02:
            estado = 1 - estado
        if estado == 0:
            ret = np.random.normal(0.001, 0.01)
        else:
            ret = np.random.normal(-0.0015, 0.025)
        retornos.append(ret)
    retornos = np.array(retornos)
    precios = start_price * np.cumprod(1 + retornos)
    precios = np.maximum(precios, 0.1)
    return precios


GENERATORS = {
    "Tendencia + ciclo": generar_precios,
    "Browniano": generar_precios_browniano,
    "Reversión a la media": generar_precios_reversion_media,
    "Regímenes": generar_precios_regimenes,
}

# ---------------------------------------------------------------------------------------------
# 2. Protocolo común (copiado de q_learning.ipynb)
# ---------------------------------------------------------------------------------------------

N_STEPS = 500          # longitud de cada serie, como en el enunciado
N_VAL = 30             # series de validación: para elegir hiperparámetros y para las curvas
N_TEST = 100           # series de test: solo para la evaluación final
N_SEEDS = 5            # semillas del entrenamiento final
FINAL_EPISODES = 3000  # episodios del entrenamiento final, iguales para los tres algoritmos
EVAL_EVERY = 100       # cada cuántos episodios miramos la política greedy en validación

# Rangos de semillas disjuntos: entrenamiento, calibración de la discretización, validación y test
SEED_TRAIN, SEED_CALIB, SEED_VAL, SEED_TEST = 0, 100_000, 500_000, 900_000


def make_series(gen_name, seed, n=N_STEPS):
    """Serie reproducible: los generadores usan el RNG global de numpy, así que fijamos la semilla."""
    np.random.seed(seed)
    return GENERATORS[gen_name](n)


_SETS = {}   # se generan la primera vez que se piden y se reutilizan


def val_sets():
    if "val" not in _SETS:
        _SETS["val"] = {g: [make_series(g, SEED_VAL + i) for i in range(N_VAL)] for g in GENERATORS}
    return _SETS["val"]


def test_sets():
    if "test" not in _SETS:
        _SETS["test"] = {g: [make_series(g, SEED_TEST + i) for i in range(N_TEST)] for g in GENERATORS}
    return _SETS["test"]


class Discretizer:
    """Convierte una observación continua en un índice de estado discreto.

    bins: {índice de la observación: nº de intervalos}. La posición (obs[1]) se añade siempre como
    última dimensión binaria.
    """

    def __init__(self, gen_name, bins={0: 10}, env_cls=SimpleTradingEnv, n_series=40):
        obs = []
        for i in range(n_series):
            env = env_cls(make_series(gen_name, SEED_CALIB + i))
            o, _ = env.reset()
            done = False
            while not done:
                obs.append(o)
                o, _, done, _, _ = env.step(1)
        obs = np.array(obs)
        self.idx = list(bins)
        self.edges = [np.quantile(obs[:, j], np.linspace(0, 1, bins[j] + 1)[1:-1]) for j in self.idx]
        self.sizes = [bins[j] for j in self.idx] + [2]
        self.n_states = int(np.prod(self.sizes))

    def __call__(self, obs):
        s = 0
        for e, j, size in zip(self.edges, self.idx, self.sizes):
            s = s * size + int(np.searchsorted(e, obs[j]))
        return s * 2 + int(obs[1])


def run_episode(prices, policy, env_cls=SimpleTradingEnv):
    env = env_cls(prices)
    obs, _ = env.reset()
    balances, positions, trades, done = [env.balance], [], 0, False
    while not done:
        a = policy(obs)
        positions.append(env.position)
        prev = env.position
        obs, _, done, _, info = env.step(a)
        trades += int(prev != env.position)
        balances.append(info["balance"])
    return np.array(balances), np.array(positions), trades


def episode_metrics(balances, positions, trades):
    rets = balances[1:] / balances[:-1] - 1
    peak = np.maximum.accumulate(balances)
    sd = rets.std()
    return {"final": balances[-1], "ret": balances[-1] / balances[0] - 1,
            "log_ret": np.log(balances[-1] / balances[0]),
            "sharpe": 0.0 if sd < 1e-12 else rets.mean() / sd * np.sqrt(252),
            "max_dd": ((peak - balances) / peak).max(), "trades": trades, "invested": positions.mean()}


def evaluate(series_list, make_policy, env_cls=SimpleTradingEnv):
    """make_policy(i) -> política (obs -> acción) para la serie i."""
    rows = [episode_metrics(*run_episode(p, make_policy(i), env_cls)) for i, p in enumerate(series_list)]
    return pd.DataFrame(rows)


def random_policy_factory(base_seed):
    def make(i):
        rng = np.random.default_rng(base_seed + i)
        return lambda obs: int(rng.integers(3))
    return make


BASELINES = {
    "Aleatorio": random_policy_factory(SEED_TEST),
    "Buy & Hold": lambda i: (lambda obs: 2),
    "Siempre fuera": lambda i: (lambda obs: 1),
    "Momentum (regla)": lambda i: (lambda obs: 2 if obs[0] > 0 else 0),
    "Contrarian (regla)": lambda i: (lambda obs: 2 if obs[0] < 0 else 0),
}


def convergence_episode(val_ep, val_curve, frac_final=0.25):
    """Criterio de q_learning.ipynb, el mismo para los tres algoritmos: primer episodio a partir
    del cual la curva de validación (suavizada con 3 puntos) ya no se sale de una banda alrededor
    de su valor final."""
    v = pd.Series(val_curve).rolling(3, min_periods=1).mean().values
    tail = v[-max(2, int(len(v) * frac_final)):]
    final = tail.mean()
    tol = max(2 * tail.std(), 0.05 * abs(final), 0.01)
    inside = np.abs(v - final) <= tol
    k = len(v) - 1
    while k > 0 and inside[k - 1]:
        k -= 1
    return val_ep[k]


def summarize(res, name, gen, extra=None):
    row = {"Generador": gen, "Agente": name,
           "Log-ret. medio": res["log_ret"].mean(),
           "Ret. geométrico (%)": 100 * (np.exp(res["log_ret"].mean()) - 1),
           "Ret. mediano (%)": 100 * res["ret"].median(),
           "Sharpe": res["sharpe"].mean(),
           "Max DD (%)": 100 * res["max_dd"].mean(),
           "Operaciones": res["trades"].mean(),
           "% tiempo invertido": 100 * res["invested"].mean()}
    row.update(extra or {})
    return row


# ---------------------------------------------------------------------------------------------
# 3a. Q-Learning (copia literal de q_learning.ipynb)
# ---------------------------------------------------------------------------------------------


class QLearningAgent:
    def __init__(self, n_states, alpha=0.1, gamma=0.9, eps_start=1.0, eps_min=0.01,
                 decay_frac=0.7, double=False, seed=0):
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_min, self.decay_frac = eps_start, eps_min, decay_frac
        self.double = double
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, 3))
        self.Q2 = np.zeros((n_states, 3)) if double else None

    def q_values(self, s):
        return self.Q[s] + self.Q2[s] if self.double else self.Q[s]

    def act(self, s, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(3))
        q = self.q_values(s)
        best = np.flatnonzero(q == q.max())
        return int(best[0] if len(best) == 1 else self.rng.choice(best))

    def greedy(self, s):
        q = self.q_values(s)
        best = np.flatnonzero(q == q.max())
        return 1 if 1 in best else int(best[0])

    def update(self, s, a, r, s2, done):
        if not self.double:
            target = r if done else r + self.gamma * self.Q[s2].max()
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])
        else:
            A, B = (self.Q, self.Q2) if self.rng.random() < 0.5 else (self.Q2, self.Q)
            target = r if done else r + self.gamma * B[s2, int(A[s2].argmax())]
            A[s, a] += self.alpha * (target - A[s, a])

    def greedy_policy(self, disc):
        return lambda obs: self.greedy(disc(obs))


def train_q_learning(gen_name, disc, n_episodes=2000, env_cls=SimpleTradingEnv, seed=0,
                     val_series=None, eval_every=100, **hp):
    """Entrena un agente con una serie nueva por episodio y, opcionalmente, mide en validación."""
    agent = QLearningAgent(disc.n_states, seed=seed, **hp)
    decay = (agent.eps_min / agent.eps_start) ** (1.0 / max(1, agent.decay_frac * n_episodes))
    eps = agent.eps_start
    hist = {"ep_reward": np.zeros(n_episodes), "val_ep": [], "val_logret": [], "eps": np.zeros(n_episodes)}
    t0 = time.perf_counter()
    for ep in range(n_episodes):
        env = env_cls(make_series(gen_name, SEED_TRAIN + seed * 1_000_000 + ep))
        obs, _ = env.reset()
        s = disc(obs)
        done, total = False, 0.0
        while not done:
            a = agent.act(s, eps)
            obs, r, done, _, _ = env.step(a)
            s2 = disc(obs)
            agent.update(s, a, r, s2, done)
            s, total = s2, total + r
        hist["ep_reward"][ep], hist["eps"][ep] = total, eps
        eps = max(agent.eps_min, eps * decay)
        if val_series is not None and ((ep + 1) % eval_every == 0 or ep == 0):
            res = evaluate(val_series, lambda i: agent.greedy_policy(disc), env_cls)
            hist["val_ep"].append(ep + 1)
            hist["val_logret"].append(res["log_ret"].mean())
    hist["train_time"] = time.perf_counter() - t0   # incluye las evaluaciones de validación
    return agent, hist


# ---------------------------------------------------------------------------------------------
# 3b. SARSA (copia de taller_trader_sarsa.ipynb)
# ---------------------------------------------------------------------------------------------

SEED_AJUSTE_SARSA = 30_000   # semilla con la que SARSA calibra sus intervalos


def crear_series(generador, n_series, n_steps=N_STEPS, seed=0):
    """Conjunto fijo y reproducible de series (no altera el generador aleatorio global)."""
    estado = np.random.get_state()
    np.random.seed(seed)
    series = [generador(n_steps) for _ in range(n_series)]
    np.random.set_state(estado)
    return series


class TrendDiscretizer:
    """Codifica la observación (tendencia, posición) como un entero: estado = 2 * intervalo + posición."""

    def __init__(self, limites, centros=None):
        self.limites = [float(x) for x in limites]
        self.n_bins = len(self.limites) + 1
        self.n_estados = 2 * self.n_bins
        self.centros = centros      # tendencia mediana de cada intervalo (solo para representar)

    @classmethod
    def ajustar(cls, generador, n_bins, n_series=40, n_steps=N_STEPS, seed=SEED_AJUSTE_SARSA):
        tendencias = []
        for precios in crear_series(generador, n_series, n_steps, seed):
            env = SimpleTradingEnv(precios)
            obs, _ = env.reset()
            terminated = False
            tendencias.append(obs[0])
            while not terminated:
                obs, _, terminated, _, _ = env.step(1)
                tendencias.append(obs[0])
        tendencias = np.array(tendencias, dtype=float)

        limites = np.unique(np.quantile(tendencias, np.linspace(0, 1, n_bins + 1)[1:-1]))
        intervalo = np.searchsorted(limites, tendencias, side="right")
        centros = [np.median(tendencias[intervalo == i]) if np.any(intervalo == i) else np.nan
                   for i in range(len(limites) + 1)]
        return cls(limites, centros)

    def __call__(self, obs):
        return 2 * bisect.bisect_right(self.limites, float(obs[0])) + int(obs[1])


class AgenteTabular:
    """Tabla Q, política ε-greedy y calendarios de α y ε. Las subclases definen `entrenar_episodio`."""

    # Acciones con efecto distinto en cada posición, con 'mantener' (1) primero para desempatar en evaluación
    ACCIONES = {0: (1, 2), 1: (1, 0)}

    def __init__(self, disc, alpha=0.1, alpha_final=None, gamma=0.9,
                 eps_inicial=1.0, eps_final=0.01, eps_frac=0.6, mascara=True, seed=0):
        self.disc = disc
        self.alpha0, self.alpha_final = alpha, alpha_final
        self.gamma = gamma
        self.eps_inicial, self.eps_final, self.eps_frac = eps_inicial, eps_final, eps_frac
        self.rng = random.Random(seed)
        todas = (1, 0, 2)
        self.validas = self.ACCIONES if mascara else {0: todas, 1: todas}
        self.Q = [[0.0] * 3 for _ in range(disc.n_estados)]
        self.visitas = [[0] * 3 for _ in range(disc.n_estados)]
        self.alpha, self.eps = alpha, eps_inicial

    def fijar_episodio(self, ep, n_episodios):
        """Calendario de ε y α para el episodio `ep` (0, 1, ..., n_episodios - 1)."""
        tramo = max(1.0, self.eps_frac * n_episodios)
        self.eps = max(self.eps_final, self.eps_inicial * (self.eps_final / self.eps_inicial) ** (ep / tramo))
        if self.alpha_final is not None:
            self.alpha = self.alpha0 * (self.alpha_final / self.alpha0) ** (ep / max(1, n_episodios - 1))

    def accion_greedy(self, s, aleatorio=True):
        acciones, q = self.validas[s & 1], self.Q[s]
        mejor = max(q[a] for a in acciones)
        empatadas = [a for a in acciones if q[a] == mejor]
        return self.rng.choice(empatadas) if aleatorio and len(empatadas) > 1 else empatadas[0]

    def elegir_accion(self, s):
        """Política ε-greedy respecto a la Q actual."""
        if self.rng.random() < self.eps:
            return self.rng.choice(self.validas[s & 1])
        return self.accion_greedy(s)

    def politica(self, obs):
        """Política greedy determinista (ε = 0), la que se usa para evaluar."""
        return self.accion_greedy(self.disc(obs), aleatorio=False)

    @property
    def tabla_q(self):
        """Q con forma (intervalo de tendencia, posición, acción)."""
        return np.array(self.Q).reshape(self.disc.n_bins, 2, 3)

    def entrenar_episodio(self, env):
        raise NotImplementedError


class AgenteSarsa(AgenteTabular):
    """Sarsa: control TD on-policy."""

    def actualizar(self, s, a, r, s2, a2, terminal):
        """Q(S,A) <- Q(S,A) + α [R + γ Q(S',A') - Q(S,A)], con Q(terminal, ·) = 0."""
        objetivo = r if terminal else r + self.gamma * self.Q[s2][a2]
        self.Q[s][a] += self.alpha * (objetivo - self.Q[s][a])

    def entrenar_episodio(self, env):
        obs, _ = env.reset()
        s = self.disc(obs)
        a = self.elegir_accion(s)                      # A elegida desde S con la política ε-greedy
        retorno, terminated, truncated = 0.0, False, False

        while not (terminated or truncated):
            obs, r, terminated, truncated, _ = env.step(a)
            s2 = self.disc(obs)
            a2 = None if terminated else self.elegir_accion(s2)   # A' desde S' (misma política)
            self.actualizar(s, a, r, s2, a2, terminated)
            self.visitas[s][a] += 1
            retorno += r
            s, a = s2, a2

        return retorno, env.balance


# Mejores hiperparámetros encontrados en taller_trader_sarsa.ipynb (MEJORES_PARAMS_GUARDADOS)
SARSA_BEST_PARAMS = {
    "Tendencia + ciclo": {"alpha": 0.014, "alpha_final": 0.0014, "gamma": 0.0, "n_bins": 5, "eps_final": 0.1, "eps_frac": 0.3},
    "Browniano": {"alpha": 0.0251, "alpha_final": 0.00251, "gamma": 0.5, "n_bins": 5, "eps_final": 0.1, "eps_frac": 0.6},
    "Reversión a la media": {"alpha": 0.007, "alpha_final": 7e-05, "gamma": 0.0, "n_bins": 25, "eps_final": 0.1, "eps_frac": 0.3},
    "Regímenes": {"alpha": 0.0152, "alpha_final": 0.00152, "gamma": 0.99, "n_bins": 15, "eps_final": 0.1, "eps_frac": 0.3},
}


def train_sarsa(gen_name, params, n_episodes=FINAL_EPISODES, seed=0, val_series=None, eval_every=EVAL_EVERY):
    """Igual que `entrenar_agente` del notebook de SARSA (mismas series de entrenamiento y mismos
    calendarios de α y ε). Lo único que cambia es la validación, que ahora usa las series comunes.
    Hace falta porque los agentes de SARSA no se guardaron y hay que volver a entrenarlos."""
    generador = GENERATORS[gen_name]
    params = dict(params)
    disc = TrendDiscretizer.ajustar(generador, params.pop("n_bins"))
    agente = AgenteSarsa(disc, seed=seed, **params)
    np.random.seed(seed)
    hist = {"ep_reward": np.zeros(n_episodes), "val_ep": [], "val_logret": []}
    t0 = time.perf_counter()
    for ep in range(n_episodes):
        agente.fijar_episodio(ep, n_episodes)
        env = SimpleTradingEnv(generador(N_STEPS))
        hist["ep_reward"][ep], _ = agente.entrenar_episodio(env)
        if val_series is not None and ((ep + 1) % eval_every == 0 or ep == 0):
            res = evaluate(val_series, lambda i: agente.politica)   # no toca el RNG de numpy, así que no cambia las series de entrenamiento
            hist["val_ep"].append(ep + 1)
            hist["val_logret"].append(res["log_ret"].mean())
    hist["train_time"] = time.perf_counter() - t0
    return agente, hist


# ---------------------------------------------------------------------------------------------
# 3c. Control Monte Carlo on-policy (nuevo)
# ---------------------------------------------------------------------------------------------


class MonteCarloAgent:
    """Control Monte Carlo on-policy con política ε-greedy (el "On-Policy MC Control" del tema 2).

    Se juega un episodio entero con la política actual y al final se recorre al revés acumulando
    el retorno G <- γG + R_{t+1}, que es lo que se usa como objetivo para Q(S_t, A_t).

    Dos opciones que no vienen fijadas en las diapositivas y que probamos en el ajuste:
    - first_visit: True usa solo la primera visita a cada (s, a) del episodio, como en clase.
      False usa todas (every-visit). Con tan pocos estados cada par se repite muchas veces por
      episodio, así que every-visit aprovecha bastantes más datos.
    - alpha: None hace la media de todos los retornos (α = 1/N, el MC "de libro"). Con un α fijo
      se van olvidando los retornos viejos, que vienen de políticas peores.

    Elegir acción, desempatar y bajar ε se hace igual que en QLearningAgent, para que la única
    diferencia entre los dos sea cómo se actualiza Q.
    """

    def __init__(self, n_states, alpha=0.01, gamma=0.9, eps_start=1.0, eps_min=0.01,
                 decay_frac=0.7, first_visit=False, seed=0):
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_min, self.decay_frac = eps_start, eps_min, decay_frac
        self.first_visit = first_visit
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, 3))
        self.N = np.zeros((n_states, 3))      # cuántos retornos lleva cada (s, a), para α = 1/N

    def act(self, s, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(3))
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        return int(best[0] if len(best) == 1 else self.rng.choice(best))

    def greedy(self, s):
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        return 1 if 1 in best else int(best[0])

    def update_episode(self, states, actions, rewards):
        """Actualiza Q con un episodio ya terminado, de atrás hacia delante."""
        if self.first_visit:
            first = {}   # (s, a) -> paso de su primera aparición
            for t, sa in enumerate(zip(states, actions)):
                first.setdefault(sa, t)
        G = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            G = self.gamma * G + rewards[t]
            s, a = states[t], actions[t]
            if self.first_visit and first[(s, a)] != t:
                continue
            self.N[s, a] += 1
            step = self.alpha if self.alpha is not None else 1.0 / self.N[s, a]
            self.Q[s, a] += step * (G - self.Q[s, a])

    def greedy_policy(self, disc):
        return lambda obs: self.greedy(disc(obs))


def train_monte_carlo(gen_name, disc, n_episodes=2000, env_cls=SimpleTradingEnv, seed=0,
                      val_series=None, eval_every=100, **hp):
    """Calcado de train_q_learning (mismas series, mismo ε y misma validación). La diferencia es que
    aquí se guarda la trayectoria y Q solo se actualiza cuando acaba el episodio."""
    agent = MonteCarloAgent(disc.n_states, seed=seed, **hp)
    decay = (agent.eps_min / agent.eps_start) ** (1.0 / max(1, agent.decay_frac * n_episodes))
    eps = agent.eps_start
    hist = {"ep_reward": np.zeros(n_episodes), "val_ep": [], "val_logret": [], "eps": np.zeros(n_episodes)}
    t0 = time.perf_counter()
    for ep in range(n_episodes):
        env = env_cls(make_series(gen_name, SEED_TRAIN + seed * 1_000_000 + ep))
        obs, _ = env.reset()
        s = disc(obs)
        states, actions, rewards, done = [], [], [], False
        while not done:
            a = agent.act(s, eps)
            obs, r, done, _, _ = env.step(a)
            states.append(s); actions.append(a); rewards.append(r)
            s = disc(obs)
        agent.update_episode(states, actions, rewards)
        hist["ep_reward"][ep], hist["eps"][ep] = sum(rewards), eps
        eps = max(agent.eps_min, eps * decay)
        if val_series is not None and ((ep + 1) % eval_every == 0 or ep == 0):
            res = evaluate(val_series, lambda i: agent.greedy_policy(disc), env_cls)
            hist["val_ep"].append(ep + 1)
            hist["val_logret"].append(res["log_ret"].mean())
    hist["train_time"] = time.perf_counter() - t0   # incluye las evaluaciones de validación
    return agent, hist
