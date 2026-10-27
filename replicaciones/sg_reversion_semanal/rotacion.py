"""Estadísticos de la reversión semanal simple y técnicas para reducir su rotación (exploratorio).

python -m replicaciones.sg_reversion_semanal.rotacion   →   resultados/rotacion/

Señal: −retorno a 5 sesiones, ordenado dentro del sector; cartera como en la réplica (quintiles neutrales a
sector, pesos iguales, señal el miércoles, ejecución al cierre siguiente). Periodo: el fuera de muestra del ML
(2018-2021). La rejilla de variantes se fijó antes de verla y se informa entera; no es una optimización.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import caracteristicas, datos
from . import config as C
from .cartera import backtest, fechas_semanales, metricas, pesos
from .caracteristicas import retorno_entre

SALIDA = C.RAIZ / "resultados" / "rotacion"
INICIO = pd.Timestamp("2018-01-03")
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def _excesos(d: pd.DataFrame, pxf: pd.DataFrame, lag: int) -> pd.Series:
    """Retorno de t+lag a t+lag+5 en exceso de la media de su sector."""
    dias = pxf.index
    p = dias.get_indexer(d["date"])
    ini = pd.Series(dias[np.minimum(p + lag, len(dias) - 1)])
    fin = pd.Series(dias[np.minimum(p + lag + C.HORIZONTE, len(dias) - 1)])
    r = pd.Series(retorno_entre(pxf, ini, fin, d["symbol"].reset_index(drop=True)), index=d.index)
    r[p + lag + C.HORIZONTE >= len(dias)] = np.nan
    return r - r.groupby([d["date"], d["sector"]]).transform("mean")


def estadisticos_senal(d: pd.DataFrame, pxf: pd.DataFrame) -> dict:
    out = {}
    # IC (Spearman transversal) y su decaimiento con el retraso de ejecución.
    decaimiento = {}
    for lag in range(0, 11):
        exc = _excesos(d, pxf, lag)
        ic = pd.concat([d["senal"], exc], axis=1).groupby(d["date"]).apply(
            lambda g: g.iloc[:, 0].corr(g.iloc[:, 1], method="spearman")).dropna()
        decaimiento[lag] = ic.mean()
        if lag in (0, 1):
            out[f"ic_lag{lag}"] = {
                "media": ic.mean(), "desv": ic.std(), "t": ic.mean() / ic.std() * np.sqrt(len(ic)),
                "ir_anual": ic.mean() / ic.std() * np.sqrt(C.SEMANAS_ANIO), "pct_positivo": (ic > 0).mean(), "n": len(ic),
            }
    out["ic_por_lag"] = decaimiento
    # Retorno medio en exceso del sector por decil de la señal (1 = mayores caídas → se compra).
    exc1 = _excesos(d, pxf, 1)
    dec = (d.groupby(["date", "sector"])["senal"].rank(pct=True) * 10).clip(upper=9.999).astype(int) + 1
    out["decil_ret_anual"] = (exc1.groupby(11 - dec).mean() * C.SEMANAS_ANIO).to_dict()
    # Persistencia de la señal: correlación de rangos semana a semana y permanencia en el quintil.
    r = d.pivot(index="date", columns="symbol", values="senal").rank(axis=1, pct=True)
    out["autocorr_rangos"] = float(np.nanmean([r.iloc[i].corr(r.iloc[i - 1]) for i in range(1, len(r))]))
    q = r > 0.8
    out["permanencia_quintil_sup"] = float((q & q.shift(1)).sum(axis=1).iloc[1:].sum() / q.shift(1).sum(axis=1).iloc[1:].sum())
    return out


def _banda(d: pd.DataFrame, entrada: float, salida: float) -> pd.Series:
    """Histéresis: se entra en el cuantil `entrada` (dentro del sector) y no se sale hasta caer fuera del
    `salida`. Pesos iguales en cada pata."""
    w = pd.Series(0.0, index=d.index)
    largos, cortos = set(), set()
    for f, g in d.groupby("date", sort=True):
        r = g["rk"]
        sym = g["symbol"]
        nl = set(sym[r > 1 - entrada]) | (largos & set(sym[r > 1 - salida]))
        nc = set(sym[r <= entrada]) | (cortos & set(sym[r <= salida]))
        nl, nc = nl - nc, nc - nl
        w[g.index[sym.isin(nl)]] = 1 / max(len(nl), 1)
        w[g.index[sym.isin(nc)]] = -1 / max(len(nc), 1)
        largos, cortos = nl, nc
    return w


def _parcial(d: pd.DataFrame, lam: float, obj: pd.Series | None = None) -> pd.Series:
    """Rebalanceo parcial: w_t = λ·objetivo + (1−λ)·w_{t−1}, renormalizado a 1 por pata."""
    obj = pd.Series(pesos(d, "senal"), index=d.index) if obj is None else obj
    w = pd.Series(0.0, index=d.index)
    previo = pd.Series(dtype=float)
    for f, g in d.groupby("date", sort=True):
        o = obj[g.index].set_axis(g["symbol"])
        n = lam * o + (1 - lam) * previo.reindex(o.index).fillna(0.0)
        pos, neg = n.clip(lower=0), n.clip(upper=0)
        n = pos / pos.sum() + neg / -neg.sum()
        w[g.index] = n.to_numpy()
        previo = n
    return w


def variantes(d: pd.DataFrame, pxf: pd.DataFrame, fs: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    d = d[d["date"].isin(fs)].copy()
    d["rk"] = d.groupby(["date", "sector"])["senal"].rank(pct=True)
    v = {}
    def bt(col, fechas=fs, df=d):
        return backtest(df, pxf, fechas, col, lag=C.LAG_INVERTIBLE, coste_pb=C.COSTE_PB)
    d["w"] = pesos(d, "senal")
    v["Base (quintiles, semanal)"] = bt("w")
    for q in (0.1, 0.3):
        d["wq"] = pesos(d, "senal", cuantil=q)
        v[f"Cuantil {q:.0%}"] = bt("wq")
    v["Quincenal"] = bt("w", fs[::2])
    for s in (0.35, 0.5):
        d[f"b{s}"] = _banda(d, 0.2, s)
        v[f"Banda 20 % → {s:.0%}"] = bt(f"b{s}")
    for lam in (0.5, 0.25):
        d[f"p{lam}"] = _parcial(d, lam)
        v[f"Parcial λ={lam}"] = bt(f"p{lam}")
    d["s2"] = d["senal_2s"]
    tmp = d.dropna(subset=["s2"]).copy()
    tmp["w2"] = pesos(tmp.assign(senal=tmp["s2"]), "senal")
    v["Formación 2 semanas"] = bt("w2", df=tmp)
    d["bp"] = _parcial(d, 0.5, obj=d["b0.5"])
    v["Banda 50 % + parcial λ=0.5"] = bt("bp")
    return v


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    dd = datos.cargar()
    panel, pxf = caracteristicas.construir(dd)
    d = panel[panel["date"] >= INICIO][["date", "symbol", "sector", "ret_1w", "log_adv_6m"]].copy()
    d["senal"] = -d["ret_1w"]
    p10 = pxf / pxf.shift(10) - 1
    d["senal_2s"] = -p10.stack(future_stack=True).reindex(pd.MultiIndex.from_frame(d[["date", "symbol"]])).to_numpy()
    fs = fechas_semanales(pd.DatetimeIndex(np.sort(d["date"].unique())))
    sem = d[d["date"].isin(fs)].reset_index(drop=True)

    est = estadisticos_senal(sem, pxf)
    v = variantes(sem, pxf, fs)
    filas = []
    for n, b in v.items():
        m = metricas(b["ret_bruto"])
        coste_anual_pb = b["rotacion"].mean() * C.SEMANAS_ANIO  # pb de coste anual por cada pb por lado
        filas.append({"variante": n, "ret_bruto": m["ret_anual"], "vol": m["vol_anual"], "sharpe_bruto": m["sharpe"],
                      "rotacion_semanal": b["rotacion"].mean(), "equilibrio_pb_lado": m["ret_anual"] * 1e4 / coste_anual_pb,
                      **{f"neto_{c:.0f}pb": m["ret_anual"] - coste_anual_pb * c / 1e4 for c in (2, 5, 10)},
                      "sharpe_neto_5pb": (b["ret_bruto"] - b["rotacion"] * 5e-4).mean() / b["ret_bruto"].std() * np.sqrt(C.SEMANAS_ANIO),
                      "t_bruto": m["ret_anual"] / m["vol_anual"] * np.sqrt(len(b) / C.SEMANAS_ANIO),
                      # sin las semanas del crash de 2020, que concentran casi toda la rentabilidad
                      "bruto_sin_mar_abr_2020": b["ret_bruto"][~b.index.to_series().between("2020-02-24", "2020-04-30")].mean() * C.SEMANAS_ANIO,
                      "neto_5pb_sin_mar_abr_2020": (b["ret_bruto"] - b["rotacion"] * 5e-4)[~b.index.to_series().between("2020-02-24", "2020-04-30")].mean() * C.SEMANAS_ANIO})
    t = pd.DataFrame(filas)
    t.to_csv(SALIDA / "variantes.csv", index=False, float_format="%.4f")

    base = v["Base (quintiles, semanal)"]
    r = base["ret_bruto"]
    est["distribucion_bruta_lag1"] = {"asimetria": r.skew(), "curtosis": r.kurt(), "peor_semana": r.min(), "mejor_semana": r.max(),
                                      "pct_semanas_pos": (r > 0).mean(), **metricas(r)}
    est["por_anio_bruto"] = (r.groupby(r.index.year).sum()).to_dict()
    est["sin_2020_bruto"] = r[r.index.year != 2020].mean() * C.SEMANAS_ANIO
    est["sin_mar_abr_2020_bruto"] = r[~((r.index >= "2020-02-24") & (r.index <= "2020-04-30"))].mean() * C.SEMANAS_ANIO
    est["patas_lag1"] = {"larga": base["larga"].mean() * C.SEMANAS_ANIO, "corta": base["corta"].mean() * C.SEMANAS_ANIO}
    with open(SALIDA / "estadisticos.json", "w", encoding="utf-8") as f:
        json.dump(est, f, indent=2, ensure_ascii=False, default=float)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    pd.Series(est["ic_por_lag"]).plot.bar(ax=ax[0], color=COLORES[0])
    ax[0].set_title("IC medio por retraso de ejecución (sesiones)")
    ax[0].tick_params(axis="x", rotation=0)
    pd.Series(est["decil_ret_anual"]).sort_index().plot.bar(ax=ax[1], color=COLORES[0])
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax[1].set_title("Exceso sobre el sector por decil (1 = mayores caídas), anual")
    ax[1].tick_params(axis="x", rotation=0)
    ax[2].scatter(t["rotacion_semanal"], t["ret_bruto"], color=COLORES[0])
    for _, f in t.iterrows():
        ax[2].annotate(f["variante"], (f["rotacion_semanal"], f["ret_bruto"]), fontsize=7)
    x = np.linspace(0, t["rotacion_semanal"].max() * 1.05, 50)
    for c, col in zip((5, 10), COLORES[1:]):
        ax[2].plot(x, x * C.SEMANAS_ANIO * c / 1e4, color=col, ls="--", label=f"coste anual a {c} pb/lado")
    ax[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax[2].set_xlabel("rotación semanal (|Δw|, ambas patas)")
    ax[2].set_title("Retorno bruto frente a rotación")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(SALIDA / "rotacion.png", dpi=130)
    print(t.round(4).to_string(index=False))
    print(json.dumps(est, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
