"""Datos diarios point-in-time de la línea 1A (guía TFM).

Fuentes:
  * Índice S&P 500 (OHLC) del repositorio de Stefan Jansen (data/equities/market/sp500/sp500.csv),
    completado con Yahoo (^GSPC) solo después de su última fecha (2025-12-31). Volumen: Yahoo.
  * FRED (fredgraph.csv): VIXCLS, DFF, BAA10Y, T10Y3M.
  * Kenneth French: 12 carteras industriales diarias ponderadas por capitalización.
  * data_ref/fomc_decision_dates.csv: reuniones FOMC programadas.

Holdout sellado: `cargar()` recorta todas las series en `fin` ANTES de calcular nada y se niega a
cargar fechas del holdout (2013+) sin la llave de config.LLAVE_HOLDOUT en GUIA1A_HOLDOUT.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import requests

from . import config as C

FRED = ["VIXCLS", "DFF", "BAA10Y", "T10Y3M"]
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/12_Industry_Portfolios_daily_CSV.zip"
UA = {"User-Agent": "Mozilla/5.0 (research client)"}
INDUSTRIAS = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]


def _fred(serie: str) -> pd.Series:
    ruta = C.CACHE / f"fred_{serie}.csv"
    if not ruta.exists():
        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie}", timeout=120)
        r.raise_for_status()
        ruta.write_bytes(r.content)
    df = pd.read_csv(ruta, na_values=["."])
    s = pd.Series(df.iloc[:, 1].astype(float).to_numpy(), index=pd.to_datetime(df.iloc[:, 0]), name=serie)
    return s.dropna()


def _french() -> pd.DataFrame:
    ruta = C.CACHE / "french_12ind_daily.zip"
    if not ruta.exists():
        r = requests.get(FRENCH_URL, headers=UA, timeout=120)
        r.raise_for_status()
        ruta.write_bytes(r.content)
    with zipfile.ZipFile(ruta) as z:
        txt = z.read(z.namelist()[0]).decode("latin-1")
    lineas = txt.splitlines()
    ini = next(i for i, l in enumerate(lineas) if "Value Weighted Returns -- Daily" in l) + 1
    fin = next(i for i, l in enumerate(lineas) if "Equal Weighted Returns -- Daily" in l)
    df = pd.read_csv(io.StringIO("\n".join(lineas[ini:fin])), index_col=0)
    df = df[pd.to_numeric(df.index, errors="coerce").notna()]
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d")
    df.columns = [c.strip() for c in df.columns]
    df = df.astype(float).replace([-99.99, -999.0], np.nan) / 100
    return df[INDUSTRIAS]


def _indice() -> pd.DataFrame:
    jan = pd.read_csv(C.ML4T_REPO / "data" / "equities" / "market" / "sp500" / "sp500.csv", parse_dates=["date"])
    jan = jan.set_index("date")[["open", "high", "low", "close"]]
    ruta = C.CACHE / "yahoo_gspc.parquet"
    if not ruta.exists():
        import yfinance as yf

        d = yf.download("^GSPC", start="1980-01-01", progress=False, auto_adjust=False)
        d.columns = d.columns.get_level_values(0)
        d.to_parquet(ruta)
    yh = pd.read_parquet(ruta)
    yh.index = pd.to_datetime(yh.index).tz_localize(None)
    extra = yh.loc[yh.index > jan.index.max(), ["Open", "High", "Low", "Close"]]
    extra.columns = ["open", "high", "low", "close"]
    d = pd.concat([jan, extra]).sort_index()
    d["volume"] = yh["Volume"].reindex(d.index)
    return d


def fomc() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(pd.read_csv(C.DATA_REF / "fomc_decision_dates.csv")["date"]))


def cargar(fin: pd.Timestamp | str = C.DESC_FIN) -> pd.DataFrame:
    """Frame diario con el calendario de sesiones del índice, cortado en `fin` (incluido)."""
    fin = pd.Timestamp(fin)
    if fin >= C.HOLD_INI and not C.holdout_abierto():
        raise PermissionError(
            f"Holdout sellado: no se cargan datos desde {C.HOLD_INI.date()} sin GUIA1A_HOLDOUT={C.LLAVE_HOLDOUT}."
        )
    d = _indice()
    d = d.loc[(d.index >= C.INICIO_DATOS) & (d.index <= fin)].copy()
    for s in FRED:
        # Valor publicado del día; huecos (festivos FRED) se arrastran como mucho 5 sesiones.
        d[s] = _fred(s).loc[:fin].reindex(d.index).ffill(limit=5)
    ind = _french().loc[:fin]
    d[[f"ind_{c}" for c in INDUSTRIAS]] = ind.reindex(d.index).to_numpy()
    return d
