"""Capa de volatilidad (8.1): HAR-RV con VIX en logaritmos, reestimado cada mes con ventana expansiva.

Se compara con un GJR-GARCH(1,1,1) t de Student (paquete arch) con la misma reestimación, por QLIKE semanal.
No cuenta como hallazgo de 1B: solo da sigma5 para la regla, las barreras y el tamaño.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl

from linea1a import config as C1A
from linea1a import datos as D1A

from . import config as C

EPS = 1e-12


def cargar_diario() -> pd.DataFrame:
    """Índice OHLC (repo de Jansen) y VIX (FRED), cortados en el final de validación: el holdout sigue sellado."""
    if C.VAL_FIN >= C.HOLDOUT_INI:
        raise PermissionError("El holdout está sellado.")
    C1A.usar_repo_ml4t()
    from data import load_sp500_index

    idx = load_sp500_index().filter(
        (pl.col("timestamp") >= C.INICIO_CARGA.date()) & (pl.col("timestamp") <= C.VAL_FIN.date())
    )
    d = idx.to_pandas().set_index("timestamp")
    d.index = pd.to_datetime(d.index).astype("datetime64[ns]")
    d["VIX"] = D1A._fred("VIXCLS").reindex(d.index).ffill(limit=5)
    d["r"] = np.log(d["close"]).diff()
    return d


def regresores_har(d: pd.DataFrame) -> pd.DataFrame:
    r2 = d["r"] ** 2
    X = pd.DataFrame(index=d.index)
    X["const"] = 1.0
    X["rv_d"] = np.log(5 * r2 + EPS)
    X["rv_w"] = np.log(5 * r2.rolling(5).mean() + EPS)
    X["rv_m"] = np.log(5 * r2.rolling(22).mean() + EPS)
    X["vix"] = np.log(5 * (d["VIX"] / 100) ** 2 / 252)
    # Objetivo: log de la varianza realizada en (t, t+5]; su última fecha conocida es t+5.
    fut = sum(r2.shift(-k) for k in range(1, C.H + 1))
    X["y"] = np.log(fut + EPS)
    X["fin"] = pd.Series(d.index, index=d.index).shift(-C.H)
    return X


def sigma5_har(d: pd.DataFrame) -> pd.Series:
    """sigma5 prevista en cada sesión con coeficientes estimados al inicio de su mes (solo objetivos realizados)."""
    X = regresores_har(d)
    cols = ["const", "rv_d", "rv_w", "rv_m", "vix"]
    out = pd.Series(np.nan, index=d.index)
    meses = pd.date_range(pd.Timestamp("1993-01-01"), d.index[-1], freq="MS")
    for m in meses:
        tr = X[(X.index >= C.INICIO_HAR) & (X.fin < m)].dropna(subset=cols + ["y"])
        b, *_ = np.linalg.lstsq(tr[cols].to_numpy(), tr["y"].to_numpy(), rcond=None)
        s2 = np.var(tr["y"].to_numpy() - tr[cols].to_numpy() @ b, ddof=len(cols))
        mes = X[(X.index >= m) & (X.index < m + pd.offsets.MonthBegin(1))].dropna(subset=cols)
        out.loc[mes.index] = np.sqrt(np.exp(mes[cols].to_numpy() @ b + s2 / 2))
    return out


def sigma5_gjr(d: pd.DataFrame) -> pd.Series:
    """Varianza a 5 días del GJR-GARCH(1,1,1) t, reestimado al inicio de cada mes con datos anteriores."""
    from arch import arch_model

    r = (d["r"].loc[C.INICIO_HAR :] * 100).dropna()
    out = pd.Series(np.nan, index=d.index)
    meses = pd.date_range(pd.Timestamp("1993-01-01"), r.index[-1], freq="MS")
    for m in meses:
        fin_mes = m + pd.offsets.MonthBegin(1)
        hasta = r[r.index < fin_mes]
        n_ajuste = int((r.index < m).sum())
        am = arch_model(hasta, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(last_obs=n_ajuste, disp="off")
            f = res.forecast(horizon=C.H, start=n_ajuste - 1, reindex=False)
        var5 = f.variance.sum(axis=1)  # en %^2, indexado por la fecha de origen
        var5 = var5[(var5.index >= m) & (var5.index < fin_mes)]
        out.loc[var5.index] = np.sqrt(var5.to_numpy()) / 100
    return out


def qlike(rv: np.ndarray, h: np.ndarray) -> float:
    ok = np.isfinite(rv) & np.isfinite(h) & (rv > 0) & (h > 0)
    q = rv[ok] / h[ok]
    return float(np.mean(q - np.log(q) - 1))
