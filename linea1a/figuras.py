"""Gráficos de dependencia SHAP por fold sobre la banda de sombras (6.6)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # paleta categórica validada (dataviz)
MARCAS = ["o", "s", "^", "D", "v"]
CENTROS = np.array([-2.0, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 2.0])


def dependencia(e2: dict, tabla: pd.DataFrame, ruta) -> None:
    variables = tabla.variable.tolist()
    k = len(variables)
    fig, ejes = plt.subplots(k, 1, figsize=(7, 3.2 * k), squeeze=False)
    banda = np.nanpercentile(e2["curvas_sombra"], [5, 95], axis=(0, 1))  # (2, k, bins)
    for j, v in enumerate(variables):
        ax = ejes[j, 0]
        ax.fill_between(CENTROS, banda[0, j], banda[1, j], color="0.85", label="sombras p5-p95")
        for f, b in enumerate(e2["base"]):
            fechas, x, s = b["shap_test"]
            ax.scatter(x[:, j], s[:, j], s=6, alpha=0.25, color=COLORES[f])
            ax.plot(CENTROS, b["curvas"][j], color=COLORES[f], lw=1.6, label=f"fold {f + 1}")
        t = tabla.set_index("variable").loc[v]
        ax.set_title(
            f"{v} (signo pre-registrado {C.SIGNOS[v]:+d})  G1={t.G1:.2g} [{t.umbral_G1:.2g}]  "
            f"G2={t.G2:.2g} [{t.umbral_G2:.2g}]  G3={t.G3:.2g} [{t.umbral_G3:.2g}]",
            fontsize=9,
        )
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlim(-3, 3)
        ax.set_xlabel("x estandarizada (media y desviación del entrenamiento)")
        ax.set_ylabel("SHAP (retorno log 5d)")
        if j == 0:
            ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def nula_procedimiento(nula: pd.DataFrame, real: dict, ruta) -> None:
    cols = [c for c in ("n_estables", "r2_oos", "sharpe_neto") if c in nula]
    fig, ejes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.2))
    for ax, c in zip(np.atleast_1d(ejes), cols):
        ax.hist(nula[c].dropna(), bins=20, color="0.7")
        ax.axvline(np.nanpercentile(nula[c], 95), color="0.3", ls="--", label="p95 nula")
        ax.axvline(real[c], color="#b84a62", lw=2, label="real")
        ax.set_title(c)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def etapa1_folds(tf: pd.DataFrame, tabla: pd.DataFrame, q1: float, ruta) -> None:
    """t con signo por fold (puntos) y U1 = peor fold (barra) frente al umbral familywise q1."""
    g = tabla[tabla.corte == "global"].set_index("variable")
    orden = g.U1.sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for f in sorted(tf.fold.unique()):
        d = tf[tf.fold == f].set_index("variable").reindex(orden)
        ax.scatter(d.t_con_signo, range(len(orden)), s=36, color=COLORES[f - 1], marker=MARCAS[f - 1],
                   edgecolor="#fcfcfb", linewidth=1, label=f"fold {f}", zorder=3)
    ax.scatter(g.loc[orden, "U1"], range(len(orden)), s=90, facecolor="none", edgecolor="#0b0b0b", lw=1.2,
               label="U1 (peor fold)", zorder=4)
    ax.axvline(q1, color="#0b0b0b", ls="--", lw=1)
    ax.text(q1, len(orden) - 0.4, f" q1 = {q1:.2f}", fontsize=8, va="bottom", color="#52514e")
    ax.axvline(0, color="#c3c2b7", lw=0.8)
    ax.set_yticks(range(len(orden)), [f"{v} ({C.SIGNOS[v]:+d})" if C.SIGNOS[v] else f"{v} (|t|)" for v in orden])
    ax.set_xlabel("t Newey-West × signo pre-registrado (bloque de test de cada fold)")
    ax.set_title("Etapa 1 · cribado univariante global, descubrimiento 2011-2018", fontsize=10)
    ax.grid(axis="x", color="#e8e7e2", lw=0.6)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
