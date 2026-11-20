"""Figuras (matplotlib, PNG). Paleta categórica de referencia en orden fijo; sombras en gris."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402

SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TINTA, TINTA2, GRIS, REJILLA = "#0b0b0b", "#52514e", "#b5b3ad", "#e4e2dc"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRIS, "axes.labelcolor": TINTA2,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA, "axes.grid": True, "grid.color": REJILLA,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
})


def _guardar(fig, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def dendrograma(info: dict, ruta):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 1]})
    dendrogram(info["linkage"], labels=info["cand"], ax=ax[0], leaf_rotation=90, color_threshold=None,
               above_threshold_color=TINTA2, link_color_func=lambda k: TINTA2)
    ax[0].set_title("Clustering jerárquico de las candidatas (distancia √(½(1−|ρ|)), enlace medio)")
    ax[0].set_ylabel("distancia")
    ax[0].grid(False)
    ks, s = zip(*sorted(info["siluetas"].items()))
    ax[1].plot(ks, s, color=SERIE[0], marker="o", ms=4)
    ks_ = info.get("k_silueta", info["k"])
    ax[1].axvline(ks_, color=GRIS, lw=1, ls="--")
    ax[1].set_title(f"Silueta media (máx. en k = {ks_})" + (" · no se usa (g3)" if info.get("modo") == "individuales" else ""))
    ax[1].set_xlabel("número de clusters")
    _guardar(fig, ruta)


def mda(tm: pd.DataFrame, sombras: pd.Series, ruta):
    t = tm.sort_values("mda_gbm")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(t) + 1.5))
    col = [SERIE[0] if s else GRIS for s in t.sobrevive]
    ax.barh(t.cluster, t.mda_gbm * 1e3, color=col, height=0.6)
    ax.scatter(sombras * 1e3, np.full(len(sombras), -0.9), color=TINTA2, s=10, marker="|", label="sombras")
    ax.axvline(t.p95_sombras_gbm.iloc[0] * 1e3, color=SERIE[1], lw=1.5, ls="--", label="percentil 95 de las sombras")
    ax.axvline(0, color=TINTA2, lw=0.8)
    ax.set_xlabel("aumento del MSE OOS al permutar el cluster (×10⁻³)")
    ax.set_title("Clustered MDA · GBM, CPCV 15 particiones")
    ax.legend(loc="lower right")
    _guardar(fig, ruta)


def dependencia_shap(XS: pd.DataFrame, bloques: list[pd.DataFrame], cols: list[str], ruta, exploratorio: bool):
    n = len(cols)
    nc = min(4, n)
    nf = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nf, nc, figsize=(max(3.2 * nc, 6.5), 2.6 * nf + (1.4 if n == 1 else 0.6)), squeeze=False)
    for ax, c in zip(axs.flat, cols):
        sh = [s for s in XS.columns if s.startswith(c + "__sh")]
        for b, S in enumerate(bloques):
            x = XS.loc[S.index]
            for s in sh:
                _curva(ax, x[s], S[s], GRIS, 1.0, None)
        for b, S in enumerate(bloques):
            x = XS.loc[S.index]
            _curva(ax, x[c], S[c], SERIE[b % len(SERIE)], 1.8, f"bloque {b + 1} ({S.index[0].year}-{S.index[-1].year})")
        ax.axhline(0, color=TINTA2, lw=0.6)
        ax.set_title(c, fontsize=8.5)
        ax.set_xlabel("compuesto z (media por decil)", fontsize=7.5)
        ax.set_ylabel("SHAP (×10⁻³ de y)", fontsize=7.5)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    h, l_ = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3 if n == 1 else 6, fontsize=7.5)
    tit = "Dependencia SHAP por bloque CPCV\n(GBM, 5 semillas; sombras en gris)"
    fig.suptitle(tit + (" · EXPLORATORIO: ningún cluster superó el MDA" if exploratorio else ""), fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.14 if n == 1 else 0.05, 1, 0.9))
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def _curva(ax, x, s, color, lw, label):
    if x.nunique() <= 2:
        g = s.groupby(x).mean()
        ax.plot(g.index, g.values * 1e3, color=color, lw=lw, marker="o", ms=4, label=label)
    else:
        q = pd.qcut(x.rank(method="first"), 10, labels=False)
        g = pd.DataFrame({"x": x, "s": s}).groupby(q).mean()
        ax.plot(g.x, g.s * 1e3, color=color, lw=lw, label=label)


def curvas_capital(bts: dict[str, pd.DataFrame], ruta, titulo):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ini = max(b.index[0] for b in bts.values())
    for i, (k, b) in enumerate(bts.items()):
        b = b.loc[ini:]
        c = (1 + b["neto"]).cumprod()
        ax.plot(c.index, c.values, color=SERIE[i], lw=2 if i == 0 else 1.6, label=k)
        ax.annotate(k.split(" (")[0], (c.index[-1], c.values[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=TINTA2, va="center")
    ax.set_yscale("log")
    ax.set_ylabel("valor de 1 $ (escala log)")
    ax.set_title(titulo)
    ax.legend(loc="upper left")
    _guardar(fig, ruta)


def sharpe_trayectorias(res_cpcv: dict, ruta):
    ks = [k for k in res_cpcv if k.startswith("sel|")]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for i, k in enumerate(ks):
        v = res_cpcv[k]["sharpe_trayectorias"]
        ax.scatter(np.full(len(v), i), v, color=SERIE[0], s=28, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=TINTA, lw=2)
    ax.set_xticks(range(len(ks)), [k.split("|")[1] for k in ks])
    ax.set_ylabel("Sharpe neto anual")
    ax.set_title("Sharpe de las 5 trayectorias CPCV (largo/neutral, 1992-2012)")
    _guardar(fig, ruta)
