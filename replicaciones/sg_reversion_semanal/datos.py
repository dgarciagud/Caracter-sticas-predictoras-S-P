"""Datos de la réplica, reutilizando linea1a.datos (ETFs, FRED y el repo de Jansen).

Acciones: barras diarias AlgoSeek de los componentes del S&P 500 (`load_sp500_daily_bars`, 2017-2021,
pertenencia al índice día a día). Precio total-return = close · adj_factor (incluye splits y dividendos).
"""

from __future__ import annotations

import pandas as pd

from linea1a import config as C1
from linea1a.datos import SECTORES, _fred

from . import config as C

ETFS_REPLICA = ["SPY", *SECTORES]


def _ancho(df: pd.DataFrame, col: str) -> pd.DataFrame:
    w = df.pivot(index="date", columns="symbol", values=col).sort_index()
    w.index = pd.to_datetime(w.index).astype("datetime64[ns]")
    return w


def cargar() -> dict[str, pd.DataFrame]:
    """Frames anchos (fecha × símbolo) de acciones, ETFs y VIX, cacheados en parquet."""
    ruta = C.CACHE / "datos.parquet"
    if ruta.exists():
        largo = pd.read_parquet(ruta)
        return {k: g.drop(columns="tabla").dropna(axis=1, how="all") for k, g in largo.groupby("tabla")}
    C1.usar_repo_ml4t()
    from data import load_etfs, load_sp500_daily_bars

    b = load_sp500_daily_bars().to_pandas()
    if "date" not in b:
        b["date"] = b["timestamp"]
    b = b.drop_duplicates(["date", "symbol"])
    b["px"] = b["close"] * b["adj_factor"]
    b["hi"] = b["high"] * b["adj_factor"]
    b["lo"] = b["low"] * b["adj_factor"]
    b["dv"] = b["close"] * b["volume"]
    b["cierre"] = b["close"]  # sin ajustar, para la capitalización (v3)
    out = {k: _ancho(b, k) for k in ("px", "hi", "lo", "dv", "cierre")}

    e = load_etfs(symbols=ETFS_REPLICA, start_date=C.INICIO_ETF).to_pandas().rename(columns={"timestamp": "date"})
    out["etf"] = _ancho(e, "close").reindex(columns=ETFS_REPLICA)
    out["vix"] = _fred("VIXCLS").rename("VIXCLS").to_frame()

    largo = pd.concat([v.assign(tabla=k) for k, v in out.items()])
    largo.columns = largo.columns.astype(str)
    largo.to_parquet(ruta)
    return out
