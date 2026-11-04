"""Genera el informe de 2 páginas de la práctica 2 (informe/informe_practica2.pdf) con los resultados ya guardados.

Se lanza desde la carpeta de la práctica 2:  python informe/generar_informe.py

Saca los números de los CSV de resultados de DQN, A3C y REINFORCE (y de la práctica 1) y las curvas y políticas
de las cachés de agentes, así que no reentrena nada. Si cambian los resultados, basta con volver a ejecutarlo.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import load, Parallel, delayed
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Table, TableStyle

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # carpeta de la práctica 2
PADRE = os.path.dirname(AQUI)                                           # trader_rl.py (práctica 1)
for ruta in (AQUI, PADRE):
    sys.path.insert(0, ruta)
os.environ["PYTHONPATH"] = os.pathsep.join([AQUI, PADRE, os.environ.get("PYTHONPATH", "")])
os.chdir(AQUI)
import trader_rl as T  # noqa: E402
import a3c_lib as A  # noqa: E402

AUTORES = "Iván Nevado, Gonzalo De Ramón y Joseph Egusquiza"
OUT_DIR = os.path.join(AQUI, "informe")
GENS = list(T.GENERATORS)
ALGOS = ["DQN", "Doble DQN", "A3C", "REINFORCE", "REINFORCE con baseline"]
COLORS = {"Doble DQN": "#be185d", "A3C": "#0891b2", "REINFORCE": "#d97706", "Buy & Hold": "#eb6834"}

# ---------------------------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------------------------


def tablas():
    dqn = pd.read_csv("resultados_dqn.csv").set_index(["Generador", "Agente"])
    dqn_ext = pd.read_csv("resultados_dqn_estado_ampliado.csv", encoding="utf-8-sig").set_index(["Generador", "Agente"])
    a3c = pd.read_csv("resultados_a3c.csv", encoding="utf-8-sig").set_index(["Generador", "Agente"])
    rf = pd.read_csv("resultados_reinforce.csv", encoding="utf-8-sig").set_index(["Generador", "Agente"])
    p1 = pd.read_csv(os.path.join(PADRE, "tabla_comparativa.csv"), encoding="utf-8-sig").set_index(["Generador", "Agente"])
    return dqn, dqn_ext, a3c, rf, p1


def fila_de(fuentes, g, a):
    for f in fuentes:
        if (g, a) in f.index:
            return f.loc[(g, a)]
    raise KeyError((g, a))


def _eval_rf(run):
    pol = A.politica_de(run, len(run["snap_ep"]) - 1)
    return float(T.evaluate(T.test_sets()[run["gen"]], lambda i, p=pol: p)["log_ret"].mean())


def extras_reinforce():
    """Log-retorno en test de los experimentos extra de REINFORCE (γ = 0,99 y factor γ^t)."""
    r99 = load("cache_reinforce/runs_gamma099.joblib")
    rgt = load("cache_reinforce/runs_gamma_t.joblib")
    todos = r99 + rgt
    vals = Parallel(n_jobs=max(1, min(14, (os.cpu_count() or 2) - 1)))(delayed(_eval_rf)(r) for r in todos)
    out = {}
    for r, v in zip(r99, vals[:len(r99)]):
        out.setdefault((f"{r['variante']} con γ = 0,99", r["gen"]), []).append(v)
    for r, v in zip(rgt, vals[len(r99):]):
        if r["gamma_t"]:
            out.setdefault(("REINFORCE con baseline y factor γ^t", r["gen"]), []).append(v)
    return {k: float(np.mean(v)) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------------------------


def estilo():
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.3, "font.family": "Calibri"})


def figura_curvas(t_conv, path):
    """Curvas de validación de Doble DQN, A3C y REINFORCE (eje de episodios en escala logarítmica)."""
    dqn = [r for r in load("cache_dqn/runs_final.joblib") if r["variante"] == "Doble DQN"]
    a3c = load("cache_a3c/runs_final.joblib")
    rf = [r for r in load("cache_reinforce/runs_final.joblib") if r["variante"] == "REINFORCE"]
    curvas = {"Doble DQN": {g: [(np.array(r["hist"]["ep"]), np.array(r["hist"]["val"])) for r in dqn if r["gen"] == g] for g in GENS},
              "A3C": {g: [(np.array(r["snap_ep"]), np.array(r["val"])) for r in a3c if r["gen"] == g] for g in GENS},
              "REINFORCE": {g: [(np.array(r["snap_ep"]), np.array(r["val"])) for r in rf if r["gen"] == g] for g in GENS}}
    val = T.val_sets()
    estilo()
    fig, axes = plt.subplots(1, 4, figsize=(7.3, 1.45))
    for ax, g in zip(axes, GENS):
        for a, cs in curvas.items():
            ep = cs[g][0][0]
            v = np.array([c[1] for c in cs[g]])
            m = ep > 0
            ax.plot(ep[m], v.mean(0)[m], color=COLORS[a], lw=1.2, label=a)
            ax.fill_between(ep[m], v.min(0)[m], v.max(0)[m], color=COLORS[a], alpha=0.12, lw=0)
            ax.axvline(t_conv[(g, a)], color=COLORS[a], lw=0.8, ls=":")
        bh = T.evaluate(val[g], T.BASELINES["Buy & Hold"])["log_ret"].mean()
        ax.axhline(bh, color=COLORS["Buy & Hold"], ls="--", lw=1, label="Buy & Hold")
        ax.axhline(0, color="#8a8984", ls=":", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(g)
        ax.set_xlabel("episodio (escala log.)", labelpad=1)
        ax.tick_params(pad=1, length=2)
    axes[0].set_ylabel("log-retorno (validación)", labelpad=1)
    axes[0].legend(fontsize=5.5, loc="lower right", frameon=False, handlelength=1.5)
    plt.tight_layout(pad=0.3, w_pad=0.6)
    plt.savefig(path, dpi=300)
    plt.close(fig)


def _q_dqn(x, capas, dueling):
    h = np.maximum(capas[0][0] @ x + capas[0][1], 0)
    h = np.maximum(capas[1][0] @ h + capas[1][1], 0)
    if dueling:
        v, a = capas[2][0] @ h + capas[2][1], capas[3][0] @ h + capas[3][1]
        return v + a - a.mean()
    return capas[2][0] @ h + capas[2][1]


def _politica_dqn(exp):
    """Misma política greedy que PoliticaDQN de la libreta de DQN (acciones válidas; empates, a mantener)."""
    def pol(obs):
        x = np.clip((np.asarray(obs, dtype=np.float32) - exp["mu"]) / exp["sd"], -5, 5)
        q = _q_dqn(x, exp["capas"], exp["dueling"])
        if obs[1] < 0.5:
            return 2 if q[2] > q[1] else 1
        return 0 if q[0] > q[1] else 1
    return pol


def trend_values(p):
    w = np.lib.stride_tricks.sliding_window_view(p, 6)[:-1]
    return np.clip((w[:, -1] - w.mean(1)) / w.mean(1), -1, 1)


def figura_politicas(path):
    """Fracción de semillas que quedan invertidas tras actuar, según la tendencia y la posición."""
    dqn = [r for r in load("cache_dqn/runs_final.joblib") if r["variante"] == "Doble DQN"]
    a3c = load("cache_a3c/runs_final.joblib")
    rf = [r for r in load("cache_reinforce/runs_final.joblib") if r["variante"] == "REINFORCE"]
    pols = {"Doble DQN": {g: [_politica_dqn(r["pol"]) for r in dqn if r["gen"] == g] for g in GENS},
            "A3C": {g: [A.politica_de(r, len(r["snap_ep"]) - 1) for r in a3c if r["gen"] == g] for g in GENS},
            "REINFORCE": {g: [A.politica_de(r, len(r["snap_ep"]) - 1) for r in rf if r["gen"] == g] for g in GENS}}
    estilo()
    fig, axes = plt.subplots(1, 4, figsize=(7.3, 1.5))
    for ax, g in zip(axes, GENS):
        trs = np.concatenate([trend_values(T.make_series(g, T.SEED_CALIB + i)) for i in range(40)])
        edges = np.linspace(*np.quantile(trs, [0.02, 0.98]), 161)
        x = (edges[:-1] + edges[1:]) / 2
        M, labels = [], []
        for a in pols:
            for pos, name in [(0, "fuera"), (1, "dentro")]:
                row = []
                for xv in x:
                    o = np.array([xv, pos], dtype=np.float32)
                    acts = [f(o) for f in pols[a][g]]
                    row.append(np.mean([(ac == 2) if pos == 0 else (ac != 0) for ac in acts]))
                M.append(row)
                labels.append(f"{a} · {name}")
        ax.pcolormesh(edges, np.arange(len(M) + 1), np.array(M), cmap="Greens", vmin=0, vmax=1, shading="flat")
        for k in (2, 4):
            ax.axhline(k, color="white", lw=1.5)
        ax.axvline(0, color="#52514e", lw=0.6, ls=":")
        ax.set_ylim(len(M), 0)
        ax.set_title(g)
        ax.set_xlabel("tendencia", labelpad=1)
        ax.grid(False)
        ax.tick_params(pad=1, length=2)
        ax.set_yticks(np.arange(len(M)) + 0.5, labels if g == GENS[0] else [""] * len(M))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, symmetric=True))
    plt.tight_layout(pad=0.3, w_pad=0.6)
    plt.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Documento
# ---------------------------------------------------------------------------------------------

pdfmetrics.registerFont(TTFont("Calibri", "C:/Windows/Fonts/calibri.ttf"))
pdfmetrics.registerFont(TTFont("Calibri-Bold", "C:/Windows/Fonts/calibrib.ttf"))
pdfmetrics.registerFont(TTFont("Calibri-Italic", "C:/Windows/Fonts/calibrii.ttf"))
pdfmetrics.registerFont(TTFont("Calibri-BoldItalic", "C:/Windows/Fonts/calibriz.ttf"))
pdfmetrics.registerFontFamily("Calibri", normal="Calibri", bold="Calibri-Bold", italic="Calibri-Italic",
                              boldItalic="Calibri-BoldItalic")

INK, MUTED, RULE, HEAD_BG, BAND = "#1f2933", "#52606d", "#9aa5b1", "#e4e7eb", "#f5f7fa"
BODY = ParagraphStyle("body", fontName="Calibri", fontSize=9.1, leading=10.9, textColor=INK,
                      alignment=TA_JUSTIFY, spaceAfter=2.2)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=8, bulletIndent=1, spaceAfter=1.4)
H1 = ParagraphStyle("h1", fontName="Calibri-Bold", fontSize=10, leading=12, textColor=INK, spaceBefore=4, spaceAfter=2,
                    keepWithNext=1)
TITLE = ParagraphStyle("title", fontName="Calibri-Bold", fontSize=14, leading=16.5, textColor=INK, alignment=TA_CENTER)
SUB = ParagraphStyle("sub", fontName="Calibri", fontSize=8.6, leading=10.5, textColor=MUTED, alignment=TA_CENTER,
                     spaceAfter=4)
CAP = ParagraphStyle("cap", fontName="Calibri", fontSize=7.4, leading=8.8, textColor=MUTED, alignment=TA_JUSTIFY,
                     spaceBefore=1.5, spaceAfter=3)
CELL = ParagraphStyle("cell", fontName="Calibri", fontSize=7.3, leading=8.4, textColor=INK)


def num(x, d=2, sign=False):
    """Número con coma decimal y espacio como separador de miles."""
    x = 0.0 if round(x, d) == 0 else x                 # sin «−0,000»
    s = f"{x:+,.{d}f}" if sign else f"{x:,.{d}f}"
    return s.replace(",", " ").replace(".", ",").replace("-", "−")


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def main_table(fuentes):
    head = ["Mercado", "Agente", "Log-ret.", "± semillas", "Ret. geom. (%)", "% series > B&H", "Operac.",
            "Conv. (ep.)", "ms/ep.", "Conv. (s)"]
    rows, style = [head], []
    agents = ALGOS + ["Buy & Hold", "Aleatorio"]
    for gi, g in enumerate(GENS):
        best = max(ALGOS, key=lambda a: fila_de(fuentes, g, a)["Log-ret. medio"])
        for ai, a in enumerate(agents):
            r = fila_de(fuentes, g, a)
            is_algo = a in ALGOS
            lr = num(r["Log-ret. medio"], 3, sign=True)
            if a == best:
                lr = f"<b>{lr}</b>"
            ms = r["ms/episodio"] if is_algo else np.nan
            rows.append([g if ai == 0 else "", a, Paragraph(lr, CELL),
                         num(r["Std log-ret. entre semillas"], 3) if is_algo else "",
                         num(r["Ret. geométrico (%)"], 1, sign=True),
                         "—" if a == "Buy & Hold" else num(r["% series > B&H"], 0),
                         num(r["Operaciones"], 1),
                         num(r["Episodios hasta convergencia"], 0) if is_algo else "",
                         (num(ms, 0) + ("*" if a == "A3C" else "")) if is_algo else "",
                         num(r["Episodios hasta convergencia"] * ms / 1000, 1) if is_algo else ""])
        r0 = 1 + gi * len(agents)
        style += [("SPAN", (0, r0), (0, r0 + len(agents) - 1)),
                  ("LINEABOVE", (0, r0), (-1, r0), 0.5 if gi else 0.8, colors.HexColor(RULE)),
                  ("TEXTCOLOR", (1, r0 + len(ALGOS)), (-1, r0 + len(agents) - 1), colors.HexColor(MUTED))]
        if gi % 2:
            style.append(("BACKGROUND", (0, r0), (-1, r0 + len(agents) - 1), colors.HexColor(BAND)))
    widths = [2.6, 3.2, 1.35, 1.35, 1.85, 1.75, 1.25, 1.35, 1.2, 1.2]
    tb = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1)
    tb.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Calibri", 7.3),
        ("FONT", (0, 0), (-1, 0), "Calibri-Bold", 7.3),
        ("FONT", (0, 1), (0, -1), "Calibri-Bold", 7.3),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(HEAD_BG)),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 0.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor(RULE)),
    ] + style))
    return tb


def extra_table(fuentes, extras):
    grupos = [
        ("Variantes y extensiones de la práctica 2", [
            ("DQN en duelo", "DQN en duelo"),
            ("Doble DQN en duelo", "Doble DQN en duelo"),
            ("Doble DQN (estado ampliado)", "Doble DQN con estado ampliado"),
            ("A3C (estado ampliado)", "A3C con estado ampliado"),
            ("REINFORCE con γ = 0,99", "REINFORCE con γ = 0,99 (1500 ep.)"),
            ("REINFORCE con baseline con γ = 0,99", "REINFORCE con baseline y γ = 0,99 (1500 ep.)"),
            ("REINFORCE con baseline y factor γ^t", "REINFORCE con baseline y factor γ<super>t</super> (1500 ep.)")]),
        ("Algoritmos tabulares de la práctica 1", [
            ("Q-Learning", "Q-Learning"), ("SARSA", "SARSA"), ("Monte Carlo", "Monte Carlo"),
            ("Q-Learning (estado ampliado)", "Q-Learning con estado ampliado")])]
    rows, style = [["Variante"] + GENS], []
    for titulo, items in grupos:
        rows.append([Paragraph(f"<b>{titulo}</b>", CELL)] + [""] * 4)
        style.append(("SPAN", (0, len(rows) - 1), (-1, len(rows) - 1)))
        style.append(("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), colors.HexColor(BAND)))
        for key, label in items:
            vals = []
            for g in GENS:
                v = extras[(key, g)] if (key, g) in extras else fila_de(fuentes, g, key)["Log-ret. medio"]
                vals.append(num(v, 3, sign=True))
            rows.append([Paragraph(label, CELL)] + vals)
    tb = Table(rows, colWidths=[6.0 * cm] + [3.0 * cm] * 4)
    tb.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Calibri", 7.3),
        ("FONT", (0, 0), (-1, 0), "Calibri-Bold", 7.3),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(HEAD_BG)),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 0.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.7),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("LINEABOVE", (0, 1), (-1, 1), 0.8, colors.HexColor(RULE)),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor(RULE)),
    ] + style))
    return tb


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Calibri", 7)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawRightString(A4[0] - 1.4 * cm, 0.75 * cm, f"{doc.page}/2")
    canvas.restoreState()


def build():
    dqn, dqn_ext, a3c, rf, p1 = tablas()
    fuentes = [rf, a3c, dqn, dqn_ext, p1]
    t_conv = {(g, a): fila_de(fuentes, g, a)["Episodios hasta convergencia"] for g in GENS for a in ("Doble DQN", "A3C", "REINFORCE")}
    extras = extras_reinforce()
    fig_curvas = os.path.join(OUT_DIR, "figura_curvas.png")
    fig_pol = os.path.join(OUT_DIR, "figura_politicas.png")
    figura_curvas(t_conv, fig_curvas)
    figura_politicas(fig_pol)

    s = []
    s.append(p("Agente de <i>trading</i> con RL profundo: DQN, A3C y REINFORCE", TITLE))
    s.append(p(f"Taller de Aprendizaje por Refuerzo (MIAX-14), parte 2 · {AUTORES}", SUB))

    s.append(p("1. Qué hemos probado", H1))
    s.append(p(
        "Seguimos con el mismo entorno de la práctica 1 (estado = tendencia de los últimos 6 precios y posición; 0,1 % de "
        "comisión) y los cuatro generadores del enunciado, pero ahora el agente es una <b>red neuronal</b> en vez de una "
        "tabla. Hemos implementado los dos algoritmos que pide el enunciado, <b>DQN</b> y <b>A3C</b>, y como ampliación "
        "<b>REINFORCE</b>, el método de gradiente de la política que vimos en clase. Con los tres cubrimos las tres "
        "familias sin modelo: basados en valor, en la política y actor-crítico."))
    s.append(p(
        "Las tres redes son perceptrones con 2 capas ocultas. Una red admite la tendencia continua, así que ya no "
        "discretizamos, pero sí normalizamos: la tendencia se estandariza con la media y la desviación de cada mercado, y "
        "la recompensa se divide por la desviación típica de su retorno, para que todos los mercados trabajen en la misma "
        "escala. En cada posición solo se consideran las dos acciones que tienen efecto. Cada algoritmo funciona así:"))
    s += bullets([
        "<b>DQN:</b> aprende Q(s, a) con una memoria de 50 000 transiciones, lotes aleatorios y exploración ε-greedy "
        "decreciente, una actualización cada 4 pasos. Probamos también <b>Doble DQN</b> (una red objetivo que se copia "
        "cada k pasos y evalúa la acción que elige la red principal) y <b>DQN en duelo</b>, que separa V(s) de la ventaja "
        "de cada acción. 200 episodios por entrenamiento.",
        "<b>A3C:</b> 4 procesos juegan en paralelo y actualizan una red compartida sin esperarse. Actor y crítico comparten "
        "la red, el retorno es de n pasos (5 a 50) completado con V(s), y la entropía de la política hace el papel de ε. "
        "1500 episodios.",
        "<b>REINFORCE:</b> juega el episodio entero con su política estocástica y sube la probabilidad de cada acción en "
        "proporción al retorno completo obtenido, una actualización por episodio. Lo probamos sin y con <i>baseline</i> "
        "V(s), con la misma red que A3C. 3000 episodios.",
    ])
    s.append(p(
        "Ajustamos α (la tasa de aprendizaje de Adam y su decaimiento), la exploración (ε en DQN, la entropía en A3C y "
        "REINFORCE), γ y el tamaño de la red con una búsqueda aleatoria corta (10–16 configuraciones) y una reevaluación de "
        "las mejores con el entrenamiento completo y semillas nuevas, solo con las 30 series de validación. El número de "
        "episodios lo fijamos con las curvas de aprendizaje. Después entrenamos 5 semillas por algoritmo y mercado y lo "
        "evaluamos todo con el protocolo de la práctica 1: las mismas 100 series de test, las mismas métricas y el mismo "
        "criterio de convergencia, de modo que los resultados se comparan directamente con los algoritmos tabulares."))
    s.append(p(
        "Además, probamos la sobreestimación de Q y el periodo de copia de la red objetivo, el número de trabajadores de A3C "
        "(1 a 8), la entropía, la política estocástica frente a la <i>greedy</i>, un estado ampliado con la volatilidad "
        "reciente y la distancia a la media de 20 pasos, la varianza del gradiente de REINFORCE con y sin baseline, y el "
        "factor γ<super>t</super> de la fórmula de REINFORCE de las transparencias."))

    s.append(p("2. Resultados", H1))
    s.append(KeepTogether([main_table(fuentes), p(
        "<b>Tabla 1.</b> Resultados en test (100 series por mercado; en los algoritmos, media de 5 semillas). En negrita, el "
        "mejor algoritmo de cada mercado. ± semillas: desviación del log-retorno entre semillas. Conv.: episodios hasta "
        "converger (mediana de las semillas) y su equivalente en segundos. *A3C: ms de reloj con 4 procesos en paralelo "
        "(4 veces más de CPU).", CAP)]))
    s.append(KeepTogether([Image(fig_curvas, width=18.2 * cm, height=18.2 * cm * 1.45 / 7.3), p(
        "<b>Figura 1.</b> Curvas de aprendizaje: log-retorno medio de la política <i>greedy</i> en validación (media y rango "
        "de 5 semillas; episodios en escala logarítmica). Las líneas verticales punteadas marcan dónde converge cada "
        "algoritmo.", CAP)]))

    s.append(p("3. Análisis", H1))
    s.append(p(
        "<b>Estrategias aprendidas (Fig. 2).</b> Donde hay señal, las tres redes aprenden lo mismo que las tablas. En "
        "<b>tendencia + ciclo</b> todas aprenden <i>momentum</i> con histéresis (compran con tendencia claramente positiva "
        "y aguantan hasta que es claramente negativa) y ganan a <i>buy &amp; hold</i> en el 97–99 % de las series. En "
        "<b>reversión a la media</b> aprenden a comprar tras caídas fuertes, aunque con más dispersión entre semillas. En "
        "el <b>browniano</b> lo correcto es no operar: REINFORCE no opera nunca y DQN casi nunca (−0,5 %), mientras que A3C "
        "sigue invertido gran parte del tiempo y pierde un 7 %, más que el propio <i>buy &amp; hold</i> (−6 %). En "
        "<b>regímenes</b> DQN y Doble DQN sacan una pequeña ganancia (+3 % y +0,8 %); A3C y REINFORCE se quedan fuera."))
    s.append(KeepTogether([Image(fig_pol, width=18.2 * cm, height=18.2 * cm * 1.5 / 7.3), p(
        "<b>Figura 2.</b> Política aprendida: fracción de las 5 semillas que quedan invertidas después de actuar, según la "
        "tendencia. En la fila «fuera», verde significa que compra; en la fila «dentro», verde significa que mantiene y "
        "blanco que vende.", CAP)]))
    s.append(p(
        "<b>Hiperparámetros.</b> Se repite la lección de la práctica 1: casi todos los ajustes eligen γ bajo o medio (0–0,9) y "
        "una tasa de aprendizaje pequeña y decreciente, porque la recompensa de cada paso es casi todo ruido. En REINFORCE se ve "
        "muy claro: con γ = 0,99 el resultado en tendencia + ciclo cae de 6,49 a 2,80 (Tabla 2), por la varianza del "
        "retorno completo, igual que le pasaba a Monte Carlo. La exploración importa menos: en A3C la entropía apenas "
        "cambia el resultado, y en REINFORCE ayuda algo más mantenerla alta."))
    s.append(KeepTogether([extra_table(fuentes, extras), p(
        "<b>Tabla 2.</b> Variantes, extensiones y algoritmos tabulares de la práctica 1: log-retorno medio en test (mismo "
        "protocolo que la Tabla 1). Los experimentos de REINFORCE marcados se hicieron con 1500 episodios y 2 semillas.",
        CAP)]))
    s.append(p(
        "<b>Lo que enseñan las variantes.</b> Doble DQN no cambia la media, pero reduce a la mitad la variación entre "
        "semillas (por ejemplo, de 0,182 a 0,101 en tendencia). Medimos que DQN sobreestima Q, pero casi todo lo corrige "
        "la red objetivo y no cambia la política. El duelo no aporta con solo 3 acciones. En A3C, más trabajadores reducen "
        "el tiempo de reloj hasta unas 2 veces, pero no mejoran el aprendizaje. En REINFORCE, el <i>baseline</i> reduce "
        "2,4 veces la varianza del gradiente con γ = 0,99, pero nada con los γ cortos que elige el ajuste, así que las dos "
        "variantes rinden parecido. Por último, aplicar el factor γ<super>t</super> de las transparencias hunde el "
        "resultado (de 6,58 a 1,62 en tendencia): con γ = 0 el agente solo aprendería del primer paso de cada episodio."))
    s.append(p(
        "<b>Convergencia y coste.</b> DQN es el que menos episodios necesita (60–180), porque reutiliza cada experiencia "
        "muchas veces, pero cada episodio cuesta 134–204 ms. A3C y REINFORCE necesitan cientos o miles de episodios más "
        "baratos (unos 23–67 ms). En segundos, los algoritmos tabulares de la práctica 1 siguen siendo los más rápidos: "
        "2–5 s en tendencia + ciclo frente a 7–9 s de las redes, y 14–19 s frente a 23–55 s en reversión."))

    s.append(p("4. Conclusiones", H1))
    s.append(p(
        "Las tres redes aprenden la estrategia esperada en los mercados con señal y ganan a <i>buy &amp; hold</i> y al "
        "agente aleatorio, con la excepción de A3C en el browniano. Sus mejores resultados son 6,49 en tendencia + ciclo "
        "(REINFORCE, el más estable entre semillas) y +0,201 en reversión (Doble DQN)."))
    s.append(p(
        "Con un estado de solo dos variables, <b>ninguna red mejora a los algoritmos tabulares</b> de la práctica 1 "
        "(Monte Carlo: 6,52 y 0,238), y cuestan más cómputo. La ventaja de una red aparece cuando el estado crece: con la "
        "volatilidad y la distancia a la media, A3C sube a 6,69 en tendencia y Doble DQN a 0,27 en reversión y 0,13 en "
        "regímenes, sin discretizar nada. Igual que en la primera parte, importan más la información del estado y la "
        "elección de α y γ que el algoritmo concreto."))
    s.append(p(
        "Entre familias, los métodos de gradiente de la política (A3C y REINFORCE) son los que peor aprovechan la señal "
        "débil de reversión, y junto con SARSA son los que se quedan fuera en regímenes. Como <b>limitaciones</b>: los "
        "mercados son sintéticos, las búsquedas de hiperparámetros fueron cortas, cada algoritmo usa un número distinto de "
        "episodios, A3C no es reproducible exactamente por su asincronía, y con 5 semillas las diferencias pequeñas no son "
        "concluyentes."))

    doc = SimpleDocTemplate(os.path.join(OUT_DIR, "informe_practica2.pdf"), pagesize=A4,
                            leftMargin=1.4 * cm, rightMargin=1.4 * cm, topMargin=1.1 * cm, bottomMargin=1.1 * cm,
                            title="Agente de trading con RL profundo: DQN, A3C y REINFORCE",
                            author=AUTORES, subject="Taller de Aprendizaje por Refuerzo (MIAX-14), parte 2")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build()
    print("Generado:", os.path.join(OUT_DIR, "informe_practica2.pdf"))
