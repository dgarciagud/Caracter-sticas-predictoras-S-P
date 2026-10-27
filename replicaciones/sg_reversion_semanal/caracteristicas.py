"""Características point-in-time por acción y día, y retornos futuros.

El paper usa ~decenas de factores de SG (revisiones de BPA a 7/75/180 días, crecimiento FY2, rentabilidad
por dividendo, FCF/capital invertido, ...). Con los datos públicos del repositorio sólo hay precios y
volumen, así que se replican los factores de precio/volumen de su lista (págs. 14-15) y se documenta la
ausencia de los fundamentales. Todas las ventanas terminan en el cierre de t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .datos import SECTORES

# Nombre de la característica → etiqueta del paper (o la más cercana).
ETIQUETAS = {
    "ret_1w": "1-week return",
    "ret_1w_sec": "1-week return vs sector",
    "ret_1m": "1m Price Reversals (1-month return)",
    "res_ret_1m": "1m Res Reversals",
    "mom_3m1m": "3m1m Price Mom",
    "mom_12m1m": "12m1m Price Mom",
    "res_mom_12m1m": "12m1m Res Mom",
    "beta_12m": "12-month Beta",
    "vol_6m": "Low Vol (6-month volatility)",
    "res_vol_6m": "Low Residual Vol",
    "vol_1m": "1-month volatility",
    "rango_1m": "1-month high-low range",
    "log_adv_6m": "Size proxy (log 6m ADV)",
    "shock_vol": "Volume shock (5d / 6m ADV)",
    "dist_max_52s": "Distance to 52-week high",
    "ret_sector_1w": "Sector 1-week return",
}
# Sólo características de la acción, como en el paper. El VIX y el retorno del mercado quedan en el panel
# para el análisis por regímenes pero fuera del modelo: con un objetivo transversal sólo sirven para que los
# árboles identifiquen fechas (en la primera corrida eran las dos primeras por SHAP y el L/S era plano).
FEATURES = list(ETIQUETAS)


def _beta(r: pd.DataFrame, m: pd.Series, v: int, minp: int) -> pd.DataFrame:
    mx = m.rolling(v, min_periods=minp).mean()
    vm = m.rolling(v, min_periods=minp).var(ddof=0)
    cov = r.mul(m, axis=0).rolling(v, min_periods=minp).mean().sub(r.rolling(v, min_periods=minp).mean().mul(mx, axis=0))
    return cov.div(vm, axis=0).where(r.notna())


def asignar_sectores(r: pd.DataFrame, r_sec: pd.DataFrame, r_m: pd.Series, fechas: pd.DatetimeIndex) -> pd.DataFrame:
    """Sector de cada acción en cada fecha: el ETF SPDR cuyo exceso sobre SPY más correla con el residuo
    de la acción en las últimas V_SECTOR sesiones. No hay GICS histórico en los datos; así es point-in-time."""
    res = r.sub(r_m, axis=0)
    exc = r_sec.sub(r_m, axis=0)
    out = pd.DataFrame(index=fechas, columns=r.columns, dtype=object)
    pos = r.index.get_indexer(fechas)
    for f, p in zip(fechas, pos):
        a = res.iloc[max(0, p - C.V_SECTOR + 1) : p + 1]
        s = exc.iloc[max(0, p - C.V_SECTOR + 1) : p + 1].dropna(axis=1, thresh=C.MIN_SECTOR)
        ok = a.notna().sum() >= C.MIN_SECTOR
        if s.shape[1] == 0 or not ok.any():
            continue
        a = a.loc[:, ok]
        za = (a - a.mean()) / a.std()
        zs = (s - s.mean()) / s.std()
        corr = za.fillna(0).T.to_numpy() @ zs.fillna(0).to_numpy()
        out.loc[f, a.columns] = np.asarray(s.columns)[corr.argmax(axis=1)]
    return out


def construir(d: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (panel largo fecha×símbolo con características y sector, precios ffill para retornos)."""
    ruta = C.CACHE / "panel.parquet"
    px = d["px"]
    dias = px.index
    pxf = px.ffill(limit=C.HORIZONTE)  # una acción que sale del índice conserva su último precio
    if ruta.exists():
        return pd.read_parquet(ruta), pxf
    etf = d["etf"].ffill().reindex(dias)
    r = np.log(px).diff().where(px.notna())
    r_et = np.log(d["etf"].ffill()).diff().reindex(dias)
    r_m = r_et["SPY"]
    vix = d["vix"]["VIXCLS"].ffill().reindex(dias, method="ffill")

    beta = _beta(r, r_m, C.V_BETA, C.MIN_BETA)
    res = r - beta.shift(1).mul(r_m, axis=0)
    lp = np.log(px)
    f = {
        "ret_1w": px / px.shift(5) - 1,
        "ret_1m": px / px.shift(21) - 1,
        "res_ret_1m": res.rolling(21, min_periods=15).sum(),
        "mom_3m1m": px.shift(21) / px.shift(63) - 1,
        "mom_12m1m": px.shift(21) / px.shift(252) - 1,
        "res_mom_12m1m": res.shift(21).rolling(231, min_periods=C.MIN_BETA).sum(),
        "beta_12m": beta,
        "vol_6m": r.rolling(C.V_VOL, min_periods=C.MIN_VOL).std() * np.sqrt(252),
        "res_vol_6m": res.rolling(C.V_VOL, min_periods=C.MIN_VOL).std() * np.sqrt(252),
        "vol_1m": r.rolling(21, min_periods=15).std() * np.sqrt(252),
        "rango_1m": np.log(d["hi"] / d["lo"]).rolling(21, min_periods=15).mean(),
        "log_adv_6m": np.log(d["dv"].rolling(C.V_VOL, min_periods=21).mean()),
        "shock_vol": d["dv"].rolling(5).mean() / d["dv"].rolling(C.V_VOL, min_periods=21).mean(),
        "dist_max_52s": lp - lp.rolling(252, min_periods=C.MIN_VOL).max(),
    }
    sec = asignar_sectores(r, r_et[SECTORES], r_m, dias)
    largo = pd.concat({k: v.where(px.notna()).stack(future_stack=True) for k, v in f.items()}, axis=1)
    largo.index.names = ["date", "symbol"]
    largo["sector"] = sec.stack(future_stack=True).reindex(largo.index)
    largo = largo[largo["ret_1w"].notna() & largo["sector"].notna()].copy()
    ret_sec_1w = (etf / etf.shift(5) - 1)[SECTORES].stack(future_stack=True)
    ret_sec_1w.index.names = ["date", "sector"]
    largo["ret_sector_1w"] = ret_sec_1w.reindex(pd.MultiIndex.from_arrays([largo.index.get_level_values(0), largo["sector"]])).to_numpy()
    largo["ret_1w_sec"] = largo["ret_1w"] - largo.groupby(["date", "sector"])["ret_1w"].transform("median")
    fechas = largo.index.get_level_values(0)
    largo["vix"] = vix.reindex(fechas).to_numpy()
    largo["ret_mercado_1w"] = (etf["SPY"] / etf["SPY"].shift(5) - 1).reindex(fechas).to_numpy()
    largo = largo.reset_index()
    largo.to_parquet(ruta)
    return largo, pxf


def retorno_entre(pxf: pd.DataFrame, ini: pd.Series, fin: pd.Series, simbolos: pd.Series) -> np.ndarray:
    """Retorno simple de cada símbolo entre el cierre de `ini` y el de `fin` (vectorizado)."""
    fi = pxf.index.get_indexer(ini)
    ff = pxf.index.get_indexer(fin)
    cj = pxf.columns.get_indexer(simbolos)
    a = pxf.to_numpy()
    ok = (fi >= 0) & (ff >= 0) & (cj >= 0)
    out = np.full(len(ini), np.nan)
    out[ok] = a[ff[ok], cj[ok]] / a[fi[ok], cj[ok]] - 1
    return out


def anadir_objetivo(panel: pd.DataFrame, pxf: pd.DataFrame) -> pd.DataFrame:
    """y = rango transversal (centrado) del retorno de t a t+5 sesiones en exceso de la media del sector,
    porque la cartera es neutral a sector."""
    dias = pxf.index
    pos = dias.get_indexer(panel["date"])
    fin = pd.Series(dias[np.minimum(pos + C.HORIZONTE, len(dias) - 1)])
    valido = pos + C.HORIZONTE < len(dias)
    p = panel.copy()
    p["fwd_5d"] = np.where(valido, retorno_entre(pxf, p["date"], fin, p["symbol"]), np.nan)
    exceso = p["fwd_5d"] - p.groupby(["date", "sector"])["fwd_5d"].transform("mean")
    p["y"] = exceso.groupby(p["date"]).rank(pct=True) - 0.5
    return p


def rangos(panel: pd.DataFrame) -> pd.DataFrame:
    """Características en rango transversal centrado."""
    return panel.groupby("date")[FEATURES].rank(pct=True) - 0.5
