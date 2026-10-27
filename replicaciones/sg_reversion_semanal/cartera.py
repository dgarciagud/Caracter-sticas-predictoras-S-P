"""Carteras largo/corto neutrales a sector y su backtest semanal o mensual.

La señal se calcula al cierre de la fecha de rebalanceo t y se ejecuta al cierre de t+lag; la posición
se mantiene hasta la ejecución del rebalanceo siguiente. Retorno L/S = retorno de la pata larga menos
el de la corta (1 de nominal por lado). Coste = rotación (suma de |Δw| en ambas patas) × coste por lado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .caracteristicas import retorno_entre


def fechas_semanales(dias: pd.DatetimeIndex, dia: int = C.DIA_REBALANCEO) -> pd.DatetimeIndex:
    return dias[dias.dayofweek == dia]


def fechas_mensuales(dias: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(dias, index=dias)
    return pd.DatetimeIndex(s.groupby([dias.year, dias.month]).last().to_numpy())


def pesos(df: pd.DataFrame, col: str, cuantil: float = C.CUANTIL, por_sector: bool = True) -> pd.Series:
    """+1/n_largos al cuantil superior de `col`, -1/n_cortos al inferior; ranking dentro del sector."""
    g = df.groupby(["date", "sector"] if por_sector else ["date"])[col]
    r = g.rank(pct=True)
    w = pd.Series(0.0, index=df.index)
    w[r > 1 - cuantil] = 1.0
    w[r <= cuantil] = -1.0
    largos = (w > 0).groupby(df["date"]).transform("sum")
    cortos = (w < 0).groupby(df["date"]).transform("sum")
    return np.where(w > 0, w / largos, np.where(w < 0, w / cortos, 0.0))


def backtest(
    df: pd.DataFrame,
    pxf: pd.DataFrame,
    fechas: pd.DatetimeIndex,
    col_peso: str,
    lag: int = C.LAG_INVERTIBLE,
    coste_pb: float = C.COSTE_PB,
) -> pd.DataFrame:
    """`df` (date, symbol, col_peso) en las fechas de señal. Devuelve una fila por periodo de tenencia."""
    dias = pxf.index
    fechas = fechas[np.isin(fechas, df["date"].unique())]
    ip = dias.get_indexer(fechas)
    ejec = dias[np.minimum(ip + lag, len(dias) - 1)]
    siguiente = ejec[1:].append(pd.DatetimeIndex([dias[min(ip[-1] + lag + C.HORIZONTE, len(dias) - 1)]]))
    mapa_ini = dict(zip(fechas, ejec))
    mapa_fin = dict(zip(fechas, siguiente))
    d = df[df["date"].isin(fechas) & (df[col_peso] != 0)][["date", "symbol", col_peso]].copy()
    d["r"] = retorno_entre(pxf, d["date"].map(mapa_ini), d["date"].map(mapa_fin), d["symbol"])
    d["r"] = d["r"].fillna(0.0)
    d["contrib"] = d[col_peso] * d["r"]
    d["larga"] = d["contrib"].where(d[col_peso] > 0, 0.0)  # retorno de la pata larga
    d["corta"] = -d["contrib"].where(d[col_peso] < 0, 0.0)  # retorno de la pata corta
    sal = d.groupby("date").agg(ret_bruto=("contrib", "sum"), larga=("larga", "sum"), corta=("corta", "sum"))
    w = d.pivot(index="date", columns="symbol", values=col_peso).reindex(fechas).fillna(0.0)
    sal["rotacion"] = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1).iloc[0]).reindex(sal.index)
    sal["ret_neto"] = sal["ret_bruto"] - sal["rotacion"] * coste_pb / 1e4
    sal["inicio"] = sal.index.map(mapa_ini)
    sal["fin"] = sal.index.map(mapa_fin)
    return sal


def metricas(r: pd.Series, periodos: int = C.SEMANAS_ANIO) -> dict:
    """Retorno anualizado = media × periodos (aritmético, como las barras del paper)."""
    r = r.dropna()
    curva = (1 + r).cumprod()
    return {
        "ret_anual": r.mean() * periodos,
        "vol_anual": r.std() * np.sqrt(periodos),
        "sharpe": r.mean() / r.std() * np.sqrt(periodos),
        "max_dd": (curva / curva.cummax() - 1).min(),
        "acierto": (r > 0).mean(),
        "n": len(r),
    }
