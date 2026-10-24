"""Carga de datos (loaders y datos del repo de Jansen) y panel semanal point-in-time.

El holdout queda sellado: la serie diaria se corta en el último día de validación antes
de calcular nada, de modo que ninguna etiqueta ni característica usa precios de 2023+.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import requests
from ml4t.engineer.features.trend.sma import sma
from ml4t.engineer.features.volatility.ewma_volatility import ewma_volatility
from ml4t.engineer.features.volatility.realized_volatility import realized_volatility

from . import config as C

SECTORES = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
ETFS = ["SPY", "RSP", "HYG", "IEF", *SECTORES]
FRED_SERIES = ["DFF", "T10Y2Y", "VIXCLS"]
CFTC_2006_2016 = "https://www.cftc.gov/files/dea/history/fin_fut_txt_2006_2016.zip"
ES_CODIGO_CFTC = "13874A"


def _fred(serie: str) -> pd.Series:
    """Serie FRED (fredgraph.csv, sin clave). Si hay ALFRED de Jansen, la usa."""
    alfred = C.ML4T_REPO / "data" / "macro" / "fred_macro_initial_release.parquet"
    if alfred.exists() and serie in pl.read_parquet_schema(alfred):
        df = pl.read_parquet(alfred, columns=["date", serie]).to_pandas()
        return df.set_index(pd.to_datetime(df["date"]))[serie].astype(float)
    ruta = C.CACHE / f"fred_{serie}.csv"
    if not ruta.exists():
        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie}", timeout=60)
        r.raise_for_status()
        ruta.write_bytes(r.content)
    df = pd.read_csv(ruta, na_values=["."])
    return df.set_index(pd.to_datetime(df.iloc[:, 0]).astype("datetime64[ns]"))[serie].astype(float)


def _cot_es() -> pd.DataFrame:
    """Posición neta de leveraged funds en ES: 2006-2016 (CFTC) + 2017+ (load_cot de Jansen)."""
    from data import load_cot

    recientes = load_cot(["ES"]).to_pandas()[["report_date", "lev_money_net", "open_interest"]]
    ruta = C.CACHE / "fin_fut_txt_2006_2016.zip"
    if not ruta.exists():
        r = requests.get(CFTC_2006_2016, timeout=120)
        r.raise_for_status()
        ruta.write_bytes(r.content)
    with zipfile.ZipFile(ruta) as z:
        txt = z.read(z.namelist()[0])
    hist = pd.read_csv(io.BytesIO(txt), low_memory=False)
    hist = hist[hist["CFTC_Contract_Market_Code"].astype(str).str.strip() == ES_CODIGO_CFTC]
    hist = pd.DataFrame(
        {
            "report_date": pd.to_datetime(hist["Report_Date_as_YYYY-MM-DD"], format="%m/%d/%Y %I:%M:%S %p").astype("datetime64[ns]"),
            "lev_money_net": hist["Lev_Money_Positions_Long_All"] - hist["Lev_Money_Positions_Short_All"],
            "open_interest": hist["Open_Interest_All"],
        }
    )
    recientes["report_date"] = pd.to_datetime(recientes["report_date"]).astype("datetime64[ns]")
    cot = pd.concat([hist[hist.report_date < recientes.report_date.min()], recientes])
    return cot.sort_values("report_date").drop_duplicates("report_date").reset_index(drop=True)


def _fomc() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_csv(C.DATA_REF / "fomc_decision_dates.csv")["date"].pipe(pd.to_datetime))


def cargar_diario(fin: pd.Timestamp = C.VAL_FIN) -> pd.DataFrame:
    """Frame diario (calendario del índice) con todo lo necesario, cortado en `fin`."""
    if fin >= C.HOLDOUT_INI:
        raise PermissionError("El holdout está sellado: no se cargan precios de 2023 en adelante.")
    C.usar_repo_ml4t()
    from data import load_etfs, load_sp500_index

    ini = "2003-01-01"
    idx = load_sp500_index().filter(pl.col("timestamp") >= pl.lit(ini).str.to_date())
    idx = idx.filter(pl.col("timestamp") <= fin.date())
    d = idx.select(pl.col("timestamp").alias("date"), "close").sort("date")
    d = d.with_columns(
        r=pl.col("close").log().diff(),
        sigma_d=ewma_volatility("close", span=17),
        sma20=sma("close", 20, implementation="polars"),
        sma200=sma("close", 200, implementation="polars"),
        sd20=pl.col("close").rolling_std(20),
    ).with_columns(
        rv5=realized_volatility("r", period=5),
        rv21=realized_volatility("r", period=21),
    )
    diario = d.to_pandas().set_index("date")
    diario.index = pd.to_datetime(diario.index).astype("datetime64[ns]")

    etf = load_etfs(symbols=ETFS, start_date=ini).to_pandas()
    etf["timestamp"] = pd.to_datetime(etf["timestamp"]).astype("datetime64[ns]")
    precios = etf.pivot(index="timestamp", columns="symbol", values="close").reindex(diario.index)
    for s in ETFS:
        diario[f"px_{s}"] = precios[s]

    for s in FRED_SERIES:
        diario[s] = _fred(s).reindex(diario.index).ffill(limit=5)

    cot = _cot_es()
    # Publicación: viernes posterior al martes de la foto (report_date + 3 días).
    cot["disponible"] = cot["report_date"] + pd.Timedelta(days=3)
    z = (cot["lev_money_net"] - cot["lev_money_net"].shift(1).rolling(156, min_periods=52).mean()) / cot[
        "lev_money_net"
    ].shift(1).rolling(156, min_periods=52).std()
    cot["F12"] = z
    cot_d = pd.merge_asof(
        diario.reset_index()[["date"]], cot[["disponible", "F12"]].dropna(), left_on="date", right_on="disponible"
    )
    diario["F12"] = cot_d["F12"].to_numpy()
    return diario


def construir_panel(diario: pd.DataFrame) -> pd.DataFrame:
    """Panel semanal: candidatas F01-F14, regímenes, y, sigma_5 y retorno simple del índice."""
    d = diario.copy()
    p = d["close"]
    lp = np.log(p)
    sigma5 = np.sqrt(5) * d["sigma_d"]
    f = pd.DataFrame(index=d.index)
    f["F01"] = (lp - lp.shift(5)) / sigma5
    f["F02"] = (p - d["sma20"]) / d["sd20"]
    f["F03"] = lp.shift(21) - lp.shift(252)
    f["F04"] = p / d["sma200"] - 1
    f["F05"] = np.log(d["VIXCLS"] / d["VIXCLS"].shift(5))
    f["F06"] = (d["VIXCLS"] / 100) ** 2 - d["rv21"] ** 2
    f["F07"] = d["rv5"] / d["rv21"]
    lr = lambda s, n: np.log(d[f"px_{s}"] / d[f"px_{s}"].shift(n))  # noqa: E731
    f["F08"] = lr("RSP", 20) - lr("SPY", 20)
    sobre, disponibles = 0, 0
    for s in SECTORES:
        px = d[f"px_{s}"]
        ma = px.rolling(50, min_periods=50).mean()
        ok = ma.notna()
        sobre = sobre + ((px > ma) & ok).astype(float)
        disponibles = disponibles + ok.astype(float)
    f["F09"] = sobre / disponibles.replace(0, np.nan)
    f["F10"] = lr("HYG", 5) - lr("IEF", 5)
    f["F11"] = d["T10Y2Y"] - d["T10Y2Y"].shift(5)
    f["F12"] = d["F12"]

    sesiones = d.index
    fin_etiqueta = pd.Series(pd.NaT, index=sesiones)
    fin_etiqueta.iloc[: len(sesiones) - C.H] = sesiones[C.H :]
    # Calendario conocido ex ante: FOMC programadas y primera sesión de cada mes.
    fomc = _fomc()
    es_fomc = pd.Series(sesiones.isin(fomc).astype(int), index=sesiones)
    primera_mes = pd.Series(np.r_[True, sesiones.month[1:] != sesiones.month[:-1]].astype(int), index=sesiones)
    fwd = lambda s: s[::-1].rolling(C.H, min_periods=1).sum()[::-1].shift(-1)  # noqa: E731  suma en (t, t+H]
    f["F13"] = (fwd(es_fomc) > 0).astype(float)
    f["F14"] = (fwd(primera_mes) > 0).astype(float)
    # Las semanas con ventana incompleta al final de la muestra no tienen calendario completo.
    f.loc[fin_etiqueta.isna(), ["F13", "F14"]] = np.nan

    vix = d["VIXCLS"]
    p70 = vix.shift(1).rolling(756, min_periods=756).quantile(0.70)
    f["R_vol"] = np.where(p70.notna(), (vix > p70).astype(float), np.nan)
    f["R_trend"] = np.where(d["sma200"].notna(), (p > d["sma200"]).astype(float), np.nan)

    f["y"] = (lp.shift(-C.H) - lp) - d["DFF"] / 100 / 52
    f["ret_simple"] = p.shift(-C.H) / p - 1
    f["rf"] = d["DFF"] / 100 / 52
    f["sigma5"] = sigma5
    f["fin_etiqueta"] = fin_etiqueta

    # Decisión: último día hábil de cada semana ISO.
    iso = d.index.isocalendar()
    clave = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    ultimo = d.index.to_series().groupby(clave.to_numpy()).transform("max")
    panel = f[d.index == ultimo.to_numpy()].copy()
    panel.index.name = "fecha"
    return panel
