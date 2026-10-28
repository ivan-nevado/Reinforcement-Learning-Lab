"""Genera el informe de 2 páginas (informe/informe_taller_rl.pdf) con los resultados ya guardados.

Se lanza desde la carpeta del repositorio:  python informe/generar_informe.py

Saca los números de tabla_comparativa.csv y las curvas y políticas de las cachés de agentes, así que
no reentrena nada. Si cambian los resultados, basta con volver a ejecutarlo.
"""

import os
import sys
import __main__

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import load
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import trader_rl as T  # noqa: E402

# Los agentes de cache_qlearning/ se guardaron desde una libreta: el pickle busca las clases en __main__
__main__.QLearningAgent, __main__.Discretizer = T.QLearningAgent, T.Discretizer

AUTORES = "Iván Nevado, Gonzalo De Ramón y Joseph Egusquiza"
OUT_DIR = os.path.join(ROOT, "informe")
GENS = list(T.GENERATORS)
ALGOS = ["Q-Learning", "SARSA", "Monte Carlo"]
COLORS = {"Q-Learning": "#2a78d6", "SARSA": "#7c3aed", "Monte Carlo": "#c2410c", "Buy & Hold": "#eb6834"}

# ---------------------------------------------------------------------------------------------
# Figura: curvas de aprendizaje de los tres algoritmos
# ---------------------------------------------------------------------------------------------


def hists():
    ql = load("cache_qlearning/agentes_finales.joblib")
    mc = load("cache_montecarlo/agentes_finales.joblib")
    sa = load("cache_comparativa/agentes_sarsa.joblib")
    out = {"Q-Learning": {g: [r[4] for r in ql if r[0] == g] for g in GENS},
           "Monte Carlo": {g: [r[4] for r in mc if r[0] == g] for g in GENS},
           "SARSA": {g: [r[4] for r in sa if r[0] == "SARSA" and r[1] == g] for g in GENS}}
    return out


def trend_values(p):
    w = np.lib.stride_tricks.sliding_window_view(p, 6)[:-1]
    return np.clip((w[:, -1] - w.mean(1)) / w.mean(1), -1, 1)


def policy_figure(path):
    """Fracción de semillas que quedan invertidas tras actuar, según la tendencia y la posición."""
    ql = load("cache_qlearning/agentes_finales.joblib")
    mc = load("cache_montecarlo/agentes_finales.joblib")
    sa = load("cache_comparativa/agentes_sarsa.joblib")
    pols = {"Q-Learning": {g: [(lambda o, a=r[2], d=r[3]: a.greedy(d(o))) for r in ql if r[0] == g] for g in GENS},
            "SARSA": {g: [r[3].politica for r in sa if r[0] == "SARSA" and r[1] == g] for g in GENS},
            "Monte Carlo": {g: [(lambda o, a=r[2], d=r[3]: a.greedy(d(o))) for r in mc if r[0] == g] for g in GENS}}
    fig, axes = plt.subplots(1, 4, figsize=(7.3, 1.5))
    labels = []
    for ax, g in zip(axes, GENS):
        trs = np.concatenate([trend_values(T.make_series(g, T.SEED_CALIB + i)) for i in range(40)])
        edges = np.linspace(*np.quantile(trs, [0.02, 0.98]), 161)
        x = (edges[:-1] + edges[1:]) / 2
        M, labels = [], []
        for a in ALGOS:
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


def figure(table, path):
    H = hists()
    val = T.val_sets()
    plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.3, "font.family": "Calibri"})
    fig, axes = plt.subplots(1, 4, figsize=(7.3, 1.65))
    for ax, g in zip(axes, GENS):
        for a in ALGOS:
            hs = H[a][g]
            ep = hs[0]["val_ep"]
            v = np.array([h["val_logret"] for h in hs])
            ax.plot(ep, v.mean(0), color=COLORS[a], lw=1.2, label=a)
            ax.fill_between(ep, v.min(0), v.max(0), color=COLORS[a], alpha=0.12, lw=0)
            ax.axvline(table.loc[(g, a), "Episodios hasta convergencia"], color=COLORS[a], lw=0.8, ls=":")
        bh = T.evaluate(val[g], T.BASELINES["Buy & Hold"])["log_ret"].mean()
        ax.axhline(bh, color=COLORS["Buy & Hold"], ls="--", lw=1, label="Buy & Hold")
        ax.axhline(0, color="#8a8984", ls=":", lw=0.8)
        ax.set_title(g)
        ax.set_xlabel("episodio", labelpad=1)
        ax.tick_params(pad=1, length=2)
    axes[0].set_ylabel("log-retorno (validación)", labelpad=1)
    axes[0].legend(fontsize=5.5, loc="lower right", frameon=False, handlelength=1.5)
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
    """Número con coma decimal y espacio fino como separador de miles."""
    s = f"{x:+,.{d}f}" if sign else f"{x:,.{d}f}"
    return s.replace(",", " ").replace(".", ",").replace("-", "−")


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def main_table(t):
    head = ["Mercado", "Agente", "Log-ret.", "Ret. geom. (%)", "% series > B&H", "Sharpe", "DD máx. (%)",
            "Operac.", "Conv. (ep.)", "Conv. (s)"]
    rows, style = [head], []
    agents = ALGOS + ["Buy & Hold", "Aleatorio"]
    for gi, g in enumerate(GENS):
        best = max(ALGOS, key=lambda a: t.loc[(g, a), "Log-ret. medio"])
        for ai, a in enumerate(agents):
            r = t.loc[(g, a)]
            is_algo = a in ALGOS
            lr = num(r["Log-ret. medio"], 3, sign=True)
            if a == best:
                lr = f"<b>{lr}</b>"
            rows.append([g if ai == 0 else "", a, Paragraph(lr, CELL), num(r["Ret. geométrico (%)"], 1, sign=True),
                         "—" if a == "Buy & Hold" else num(r["% series > B&H"], 0),
                         num(r["Sharpe"], 2), num(r["Max DD (%)"], 1), num(r["Operaciones"], 1),
                         num(r["Episodios hasta convergencia"], 0) if is_algo else "",
                         num(r["Tiempo hasta convergencia (s)"], 1) if is_algo else ""])
        r0 = 1 + gi * len(agents)
        style += [("SPAN", (0, r0), (0, r0 + len(agents) - 1)),
                  ("LINEABOVE", (0, r0), (-1, r0), 0.5 if gi else 0.8, colors.HexColor(RULE)),
                  ("TEXTCOLOR", (1, r0 + 3), (-1, r0 + 4), colors.HexColor(MUTED))]
        if gi % 2:
            style.append(("BACKGROUND", (0, r0), (-1, r0 + len(agents) - 1), colors.HexColor(BAND)))
    widths = [2.7, 2.0, 1.45, 1.95, 1.75, 1.2, 1.55, 1.3, 1.45, 1.25]
    tb = Table(rows, colWidths=[w * cm for w in widths], rowHeights=None, repeatRows=1)
    tb.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Calibri", 7.3),
        ("FONT", (0, 0), (-1, 0), "Calibri-Bold", 7.3),
        ("FONT", (0, 1), (0, -1), "Calibri-Bold", 7.3),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(HEAD_BG)),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 0.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.9),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor(RULE)),
    ] + style))
    return tb


def control_table(t):
    rows_def = [("Q-Learning", "Q-Learning (base)"),
                ("Q-Learning (hiperparámetros de MC)", "Q-Learning con hiperparámetros de MC"),
                ("Monte Carlo", "Monte Carlo"),
                ("SARSA", "SARSA (base)"),
                ("SARSA (hiperparámetros de Q-Learning)", "SARSA con hiperparámetros de Q-Learning"),
                ("Double Q-Learning", "Double Q-Learning"),
                ("Q-Learning (estado ampliado)", "Q-Learning con estado ampliado")]
    rows = [["Variante"] + GENS]
    for key, label in rows_def:
        rows.append([label] + [num(t.loc[(g, key), "Log-ret. medio"], 3, sign=True) for g in GENS])
    tb = Table(rows, colWidths=[5.6 * cm] + [2.9 * cm] * 4)
    tb.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Calibri", 7.3),
        ("FONT", (0, 0), (-1, 0), "Calibri-Bold", 7.3),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(INK)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(HEAD_BG)),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 0.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.9),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("LINEABOVE", (0, 1), (-1, 1), 0.8, colors.HexColor(RULE)),
        ("LINEABOVE", (0, 4), (-1, 4), 0.5, colors.HexColor(RULE)),
        ("LINEABOVE", (0, 6), (-1, 6), 0.5, colors.HexColor(RULE)),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor(RULE)),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor(BAND)),
        ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor(BAND)),
    ]))
    return tb


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Calibri", 7)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawRightString(A4[0] - 1.4 * cm, 0.75 * cm, f"{doc.page}/2")
    canvas.restoreState()


def build():
    t = pd.read_csv("tabla_comparativa.csv", index_col=[0, 1], encoding="utf-8-sig")
    fig_path = os.path.join(OUT_DIR, "figura_curvas.png")
    figure(t, fig_path)
    pol_path = os.path.join(OUT_DIR, "figura_politicas.png")
    policy_figure(pol_path)

    s = []
    s.append(p("Agente de <i>trading</i> con RL tabular: Q-Learning, SARSA y Monte Carlo", TITLE))
    s.append(p(f"Taller de Aprendizaje por Refuerzo (MIAX-14) · {AUTORES}", SUB))

    s.append(p("1. Qué hemos probado", H1))
    s.append(p(
        "Hemos usado el entorno <i>SimpleTradingEnv</i> y las cuatro funciones generadoras tal y como venían en el "
        "enunciado: el estado es la tendencia de los últimos 6 precios más la posición, y cada operación paga un 0,1 % "
        "de comisión. Sobre ese entorno hemos implementado tres algoritmos tabulares vistos en clase: <b>Q-Learning</b>, "
        "<b>SARSA</b> y, como extra, <b>control Monte Carlo <i>on-policy</i></b> con política ε-greedy. Monte Carlo lo "
        "añadimos para comprobar en un caso real la comparación entre <i>bootstrapping</i> (TD) y retorno completo (MC) "
        "del tema 3."))
    s.append(p(
        "Como los algoritmos son tabulares, hay que discretizar la tendencia. Una rejilla uniforme en [−1, 1] no sirve: "
        "según el generador, la tendencia se mueve en ±0,03 o en ±0,3, y casi todos los datos caerían en uno o dos "
        "intervalos. Por eso usamos cuantiles calculados para cada generador y tratamos el número de intervalos (de 5 a "
        "25) como un hiperparámetro más. Cada episodio usa una serie nueva de 500 precios del mismo generador."))
    s.append(p(
        "Para ajustar α, γ, ε y el número de intervalos solo hemos usado 30 series de validación, con el log-retorno "
        "medio de la política <i>greedy</i> como criterio. Cada algoritmo tiene su propia búsqueda:"))
    s += bullets([
        "<b>Q-Learning:</b> una rejilla con α = {0,01; 0,05; 0,2}, γ = {0; 0,5; 0,9; 0,99} y 5, 10 o 20 intervalos, y "
        "después ε (396 entrenamientos).",
        "<b>SARSA:</b> una búsqueda aleatoria de 40 configuraciones con α decreciente y la reevaluación de las 3 mejores. "
        "Además, solo considera las acciones que tienen efecto en cada posición.",
        "<b>Monte Carlo:</b> otra rejilla, con α más pequeños (hasta 0,001 y 1/N), comparando además <i>every-visit</i> "
        "y <i>first-visit</i> (696 entrenamientos).",
    ])
    s.append(p(
        "Con las curvas de aprendizaje (Fig. 1) fijamos 3000 episodios, y entrenamos 5 semillas por algoritmo y mercado."))
    s.append(p(
        "Al juntar los resultados vimos que Q-Learning y SARSA no se habían evaluado con las mismas series de test ni "
        "con las mismas métricas, así que repetimos la evaluación con un protocolo común. Todos los agentes se prueban en "
        "las mismas 100 series de test por mercado, que no se han usado para ajustar nada. La métrica principal es el "
        "log-retorno medio del capital, porque la media del balance la dominan unas pocas series. Como convergencia "
        "tomamos el primer episodio a partir del cual la curva de validación ya no se sale de una banda alrededor de su "
        "valor final, y para pasarlo a segundos multiplicamos por el coste de un episodio, medido en serie en el mismo "
        "ordenador (8–11 ms en los tres). Las referencias son el agente aleatorio, <i>buy &amp; hold</i> (B&amp;H), no "
        "invertir nunca y dos reglas fijas, <i>momentum</i> y <i>contrarian</i>."))
    s.append(p(
        "Además probamos un estado ampliado (con la volatilidad reciente y la distancia a la media de 20 pasos), "
        "<i>Double</i> Q-Learning y varias ablaciones en SARSA. Por último hicimos dos experimentos de control "
        "intercambiando hiperparámetros entre algoritmos (sección 3)."))

    s.append(p("2. Resultados", H1))
    s.append(KeepTogether([main_table(t), p(
        "<b>Tabla 1.</b> Resultados en test (100 series por mercado; en los algoritmos, media de 5 semillas). En negrita, "
        "el mejor algoritmo de cada mercado. DD: <i>drawdown</i>. Conv.: episodios hasta converger (mediana de las "
        "semillas) y su equivalente en segundos. Como referencia, las reglas fijas sacan 6,11 (<i>momentum</i>) en "
        "tendencia + ciclo y 0,069 (<i>contrarian</i>) en reversión.", CAP)]))
    s.append(KeepTogether([Image(fig_path, width=18.2 * cm, height=18.2 * cm * 1.65 / 7.3), p(
        "<b>Figura 1.</b> Curvas de aprendizaje: log-retorno medio de la política <i>greedy</i> en validación (media y "
        "rango de las 5 semillas). Las líneas verticales punteadas marcan dónde converge cada algoritmo.", CAP)]))

    s.append(p("3. Análisis", H1))
    s.append(p(
        "En general, las estrategias aprendidas son las que esperábamos para cada mercado (Fig. 2). <b>Tendencia + "
        "ciclo</b> es el único con una señal clara (la correlación entre la tendencia y el retorno siguiente es 0,19), y "
        "los tres algoritmos aprenden <i>momentum</i>: compran cuando la tendencia es positiva y no venden hasta que es "
        "claramente negativa. Esa histéresis hace que ganen a la regla fija de <i>momentum</i> operando menos de la mitad "
        "de veces. En el <b>browniano</b> no hay nada que aprender y, encima, la deriva logarítmica es negativa "
        "(μ − σ²/2 &lt; 0), así que lo correcto es quedarse fuera. Q-Learning y SARSA lo hacen. Monte Carlo pierde un "
        "1,2 % porque sus hiperparámetros para este mercado se eligieron por una diferencia en validación que era ruido."))
    s.append(p(
        "En <b>reversión a la media</b>, los tres compran después de caídas fuertes y venden cuando el precio se "
        "recupera. Ganan entre un 17 % y un 27 %, cuando B&amp;H se queda en −0,5 % y la regla <i>contrarian</i> en "
        "+7 %. <b>Regímenes</b> es el caso más difícil, porque la tendencia apenas dice nada del régimen. Lo único que "
        "se consigue es salir ante movimientos muy bruscos, que suelen indicar el régimen bajista (más volátil). "
        "Q-Learning y Monte Carlo sacan un 3–4 %, SARSA directamente no opera, y B&amp;H pierde un 12 %. El agente "
        "aleatorio pierde más o menos lo que paga en comisiones (164 operaciones por episodio)."))
    s.append(KeepTogether([Image(pol_path, width=18.2 * cm, height=18.2 * cm * 1.5 / 7.3), p(
        "<b>Figura 2.</b> Política aprendida: fracción de las 5 semillas que quedan invertidas después de actuar, según la "
        "tendencia. En la fila «fuera», verde significa que compra; en la fila «dentro», verde significa que mantiene y "
        "blanco que vende. El hueco entre la tendencia a la que compra y la tendencia a la que vende es la histéresis.",
        CAP)]))
    s.append(p(
        "<b>Hiperparámetros.</b> El que más influye es α. La recompensa de cada paso es casi todo ruido (el precio se "
        "mueve un 1–10 % por paso y la señal es de décimas de punto), y con α ≥ 0,05 los agentes acaban sin operar. Con "
        "γ los algoritmos no coinciden. Q-Learning necesita 0,99 en reversión y regímenes, pero Monte Carlo prefiere "
        "γ = 0 en tres de los cuatro mercados. Tiene sentido: medimos que, con γ = 0,99, el retorno completo es entre 3 "
        "y 7,5 veces más ruidoso que una recompensa suelta, que es justo la desventaja de MC frente a TD que vimos en "
        "clase. Lo curioso es que con γ = 0 no se pierde la histéresis, porque ya la crea la comisión: el agente compra "
        "si espera ganar más que la comisión y vende si espera perder más que ella. El número de intervalos y el "
        "calendario de ε importan bastante menos, y en Monte Carlo <i>every-visit</i> funciona mejor que "
        "<i>first-visit</i>."))
    s.append(KeepTogether([control_table(t), p(
        "<b>Tabla 2.</b> Experimentos de control y extensiones: log-retorno medio en test (mismo protocolo que la "
        "Tabla 1).", CAP)]))
    s.append(p(
        "<b>¿Algoritmo o hiperparámetros?</b> Como cada algoritmo se ajustó con una búsqueda distinta, no estaba claro de dónde "
        "venían las diferencias, así que intercambiamos hiperparámetros (Tabla 2). Q-Learning con los de Monte Carlo "
        "saca prácticamente lo mismo que Monte Carlo. La ventaja venía de α = 0,001, un valor que no entraba en la "
        "rejilla de Q-Learning; además, con γ = 0 los dos algoritmos tienen el mismo objetivo. Con SARSA pasa lo "
        "contrario: con los hiperparámetros de Q-Learning sigue sin operar en regímenes y rinde menos en reversión, así "
        "que ahí la diferencia sí es del algoritmo. SARSA es <i>on-policy</i> y aprende el valor de la política ε-greedy "
        "que sigue. En este entorno explorar significa comprar y vender al azar pagando comisión, así que, cuando la "
        "señal es débil, le compensa no hacer nada. Es lo mismo que pasaba en el <i>Cliff Walking</i> de clase, solo "
        "que aquí no hay acantilado y esa prudencia simplemente cuesta dinero."))
    s.append(p(
        "<b>Convergencia.</b> La velocidad depende más del mercado que del algoritmo. Con señal fuerte bastan 300–500 "
        "episodios (2–5 s); con señal débil hacen falta 1300–2200 (11–19 s). La excepción es SARSA en regímenes, que "
        "converge rápido porque converge a no operar. El coste por episodio es casi igual en los tres, porque lo que más "
        "tarda es simular el entorno. De las extensiones, la que más ayuda con diferencia es el estado ampliado (+37 % "
        "en reversión y +17 % en regímenes); <i>Double</i> Q-Learning no se distingue del ruido entre semillas."))

    s.append(p("4. Conclusiones", H1))
    s.append(p(
        "Los tres algoritmos ganan a B&amp;H y al agente aleatorio en todos los mercados (en el browniano, quedándose "
        "fuera, que es lo mejor que se puede hacer) y aprenden la estrategia que cabía esperar en cada uno. Los mejores "
        "resultados en test son un log-retorno de 6,5 en tendencia + ciclo y un +27 % en reversión, los dos con Monte "
        "Carlo, y un +4 % en regímenes con Q-Learning."))
    s.append(p(
        "Lo que más nos ha sorprendido es que el algoritmo importa menos de lo que pensábamos. Pesan más la información "
        "que tiene el estado (el estado ampliado mejora más que cualquier cambio de algoritmo) y los valores de α y γ. "
        "La única diferencia de algoritmo que se mantiene en todas las pruebas es la de <i>on-policy</i> frente a "
        "<i>off-policy</i>: SARSA es más conservador y deja pasar señales débiles que Q-Learning sí aprovecha. Monte "
        "Carlo funciona bien, pero porque esquiva su punto débil, la varianza del retorno, eligiendo un horizonte de un "
        "solo paso."))
    s.append(p(
        "También hay que tener en cuenta las limitaciones. Los mercados son sintéticos y conocemos sus parámetros, y el "
        "entorno es de todo o nada con una comisión fija. La discretización se calibra con datos del mismo generador. "
        "Y con 5 semillas, las diferencias pequeñas entre algoritmos no son concluyentes."))

    doc = SimpleDocTemplate(os.path.join(OUT_DIR, "informe_taller_rl.pdf"), pagesize=A4,
                            leftMargin=1.4 * cm, rightMargin=1.4 * cm, topMargin=1.1 * cm, bottomMargin=1.1 * cm,
                            title="Agente de trading con RL tabular: Q-Learning, SARSA y Monte Carlo",
                            author=AUTORES, subject="Taller de Aprendizaje por Refuerzo (MIAX-14)")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build()
    print("Generado:", os.path.join(OUT_DIR, "informe_taller_rl.pdf"))
