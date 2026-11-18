"""Fase 1 · Cribado univariante (A.3): regresión predictiva con t NW, observaciones no solapadas,
persistencia, quintiles con cortes del pasado, timing simple, estabilidad por bloques, IC(h),
nulo por desplazamiento circular y placebo contra retornos pasados."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .cv import bloques
from .estad import bh, ic_t, ols_hac, p_normal, pendiente_t
from .estrategia import backtest, metricas, registrar, tramos


def desplazamientos(n: int, r: int, rng) -> np.ndarray:
    lo = max(C.SOMBRA_MIN, int(np.ceil(C.SOMBRA_BORDE * n)))
    return rng.integers(lo, n - lo, size=r)


def y_horizonte(d: pd.DataFrame, h: int) -> pd.Series:
    p = d["close"]
    r = np.log(p).diff()
    sig = r.ewm(span=C.SPAN_SIGMA, min_periods=C.SPAN_SIGMA).std() * np.sqrt(h)
    rf = d["DFF"].ffill() / 100 / 252 * h
    return (p.shift(-h) / p - 1 - rf) / sig


def quintiles_pasado(x: pd.Series, y: pd.Series) -> list[float]:
    """Media de y por quintil de x con cortes expansivos (mínimo 252 sesiones, sin incluir t)."""
    q = [x.shift(1).expanding(252).quantile(k / 5) for k in (1, 2, 3, 4)]
    qn = sum((x > qk).astype(int) for qk in q).where(q[0].notna())
    return [float(y[qn == k].mean()) for k in range(5)]


def ejecutar(P: pd.DataFrame, d: pd.DataFrame, dia: pd.DataFrame, bt_bh: pd.DataFrame, bt_ts: pd.DataFrame) -> pd.DataFrame:
    cand = list(C.CANDIDATAS)
    X = P[cand].to_numpy().T
    y = P["y"].to_numpy()
    n = len(y)
    m = np.ones_like(X, bool)
    b, t = pendiente_t(X, y, m)
    rng = np.random.default_rng(C.SEMILLA)
    ks = desplazamientos(n, C.N_NULA_F1, rng)
    tnul = np.stack([pendiente_t(np.roll(X, k, axis=1), y, m)[1] for k in ks])  # (R, V)
    B = bloques(n)
    yh = {h: y_horizonte(d, h).reindex(P.index) for h in C.PREREG["inferencia"]["ic_horizontes"]}
    pasado = pd.DataFrame(
        {
            "r5": np.log(d["close"]).diff(5),
            "r21": np.log(d["close"]).diff(21),
            "r252": np.log(d["close"]).diff(252),
        }
    ).reindex(P.index)
    pasado = (pasado - pasado.mean()) / pasado.std()
    filas = []
    for j, c in enumerate(cand):
        s = C.SIGNOS[c]
        x = P[c]
        sg = s if s else int(np.sign(b[j]))
        # Nulo circular: unilateral con signo esperado, bilateral sin él.
        if s:
            p95 = np.percentile(s * tnul[:, j], 95)
            p_nula = float(np.mean(s * tnul[:, j] >= s * t[j]))
            supera = s * t[j] > p95
        else:
            p95 = np.percentile(np.abs(tnul[:, j]), 95)
            p_nula = float(np.mean(np.abs(tnul[:, j]) >= abs(t[j])))
            supera = abs(t[j]) > p95
        # No solapadas: una de cada h, cinco arranques.
        tno = []
        for o in range(C.H):
            sel = np.arange(o, n, C.H)
            tno.append(float(pendiente_t(X[j, sel][None], y[sel], np.ones((1, len(sel)), bool), lags=0)[1][0]))
        sb = [float(pendiente_t(X[j, i][None], y[i], np.ones((1, len(i)), bool))[0][0]) for i in B]
        cons = float(np.mean(np.sign(sb) == sg))
        xs = x.to_numpy()
        ar1 = float(np.corrcoef(xs[1:], xs[:-1])[0, 1])
        quint = quintiles_pasado(x, P["y"])
        mono = float(pd.Series(quint).corr(pd.Series(range(5)), method="spearman"))
        ics = {f"ic_h{h}": ic_t(x.to_numpy()[yh[h].notna()], yh[h].dropna().to_numpy())[0] for h in yh}
        _, tp = ols_hac(y, np.column_stack([xs, pasado.to_numpy()]))
        senal = tramos((x * sg > 0).astype(float))
        bt = backtest(senal, dia)
        met = metricas(bt, bt_bh, bt_ts)
        registrar(
            f"F1_timing_{c}", "fase1", f"timing largo/neutral con signo {sg:+d} de {c}", "regla_fija",
            met, bt["neto"] - bt["rf"], variables=[c], modelo="regla",
        )
        pasa_t = (abs(t[j]) >= 2) and (np.sign(t[j]) == s if s else True)
        filas.append(
            dict(
                variable=c, familia=C.FAMILIA[c], signo_esperado=s, b=b[j], t_nw=t[j],
                t_no_solapadas_min=min(tno), t_no_solapadas_max=max(tno),
                no_solapadas_mismo_signo=int(np.sum(np.sign(tno) == np.sign(t[j]))),
                ar1_x=ar1, stambaugh_relevante=ar1 > 0.9,
                **{f"q{k + 1}": v for k, v in enumerate(quint)}, monotonia_quintiles=mono,
                **ics, t_placebo=float(tp[1]),
                consistencia_bloques=cons, p95_nula=p95, p_nula=p_nula, supera_nula=bool(supera),
                sharpe_timing=met["sharpe_neto"], dif_sharpe_bh=met["dif_sharpe_bh"], t_dif_sharpe_bh=met["t_dif_sharpe_bh"],
                p_normal=p_normal(t[j], s), pasa_t=bool(pasa_t),
            )
        )
    df = pd.DataFrame(filas)
    df["bh_10"] = bh(np.maximum(df["p_nula"].to_numpy(), 1 / (C.N_NULA_F1 + 1)))
    df["pasa_fase1"] = df["pasa_t"] & df["supera_nula"] & (df["consistencia_bloques"] >= 0.7)
    return df
