"""Características point-in-time por acción y día, y retornos futuros.

El paper usa ~decenas de factores de SG (revisiones de BPA a 7/75/180 días, crecimiento FY2, rentabilidad
por dividendo, FCF/capital invertido, ...). La v2 usa sólo los factores de precio/volumen de su lista
(págs. 14-15). La v3 añade dividendos (AlgoSeek), fundamentales XBRL de la SEC y dos sustitutos de las
revisiones de analistas (SUE y retorno en torno a la publicación de resultados), ver `anadir_v3`.
Todas las ventanas terminan en el cierre de t; un dato de la SEC se usa desde la sesión siguiente a su
presentación.
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
ETIQUETAS_V3 = {
    "div_yield": "Dividend Yield",
    "fcf_ic": "Free Cash Flow to Capital Invested",
    "bp_tangible": "Equity-Intangibles to Market Value",
    "ey": "Earnings yield (TTM E/P)",
    "roe": "Return on Equity",
    "gp_activos": "Gross Profits / Assets",
    "crec_activos": "Asset Growth",
    "crec_bpa": "EPS growth (TTM, / price) [≈ FY2 growth]",
    "sue": "SUE [≈ 75/180-day EPS revisions]",
    "ear": "Earnings-announcement return [≈ EPS revisions]",
    "dias_anuncio": "Days since earnings announcement",
    "log_mcap": "Size (log market cap)",
}
if C.VERSION != "v2":
    ETIQUETAS.update(ETIQUETAS_V3)
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


def _retornos(d: dict[str, pd.DataFrame]):
    """Retornos log diarios, de los ETFs, beta a 12 meses y residuo frente a SPY (beta del día anterior)."""
    px = d["px"]
    r = np.log(px).diff().where(px.notna())
    r_et = np.log(d["etf"].ffill()).diff().reindex(px.index)
    beta = _beta(r, r_et["SPY"], C.V_BETA, C.MIN_BETA)
    return r, r_et, beta, r - beta.shift(1).mul(r_et["SPY"], axis=0)


def construir(d: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (panel largo fecha×símbolo con características y sector, precios ffill para retornos)."""
    ruta = C.CACHE / "panel.parquet"
    px = d["px"]
    dias = px.index
    pxf = px.ffill(limit=C.HORIZONTE)  # una acción que sale del índice conserva su último precio
    if ruta.exists():
        return pd.read_parquet(ruta), pxf
    etf = d["etf"].ffill().reindex(dias)
    r, r_et, beta, res = _retornos(d)
    r_m = r_et["SPY"]
    vix = d["vix"]["VIXCLS"].ffill().reindex(dias, method="ffill")
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


def _rango(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return (df.groupby("date")[cols].rank(pct=True) - 0.5).mean(axis=1, skipna=True)


def anadir_v3(panel: pd.DataFrame, d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rentabilidad por dividendo, fundamentales XBRL y sustitutos de las revisiones de BPA.

    - SUE: (BPA trimestral − BPA del mismo trimestre del año anterior) / desviación de ese cambio en los
      8 trimestres previos. Las revisiones de los analistas se concentran tras la publicación de resultados.
    - EAR: retorno residual (frente a SPY con beta) en [a−1, a+1], donde a es la sesión de mayor volumen
      relativo de las 40 anteriores a la presentación del 10-Q/10-K (el anuncio de resultados suele ir
      unos días o semanas antes del formulario). Se conoce al presentarse el formulario.
    """
    from . import fundamentales as F

    ruta = C.CACHE / f"panel{C.SUFIJO}.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta)
    px, cierre, dv = d["px"], d["cierre"], d["dv"]
    dias = px.index
    _, _, _, res = _retornos(d)

    div = F.dividendos().reindex(index=dias, columns=px.columns).fillna(0.0)
    n = px.notna().rolling(252, min_periods=1).sum()
    div_yield = (div.rolling(252, min_periods=1).sum() * 252 / n / cierre).where(n >= 126)

    ev = F.panel_fundamental(px)
    ev = ev[ev["symbol"].isin(px.columns)].sort_values("filed").reset_index(drop=True)
    pos = dias.searchsorted(ev["filed"].to_numpy(), side="right")  # primera sesión tras la presentación
    ev = ev[pos < len(dias)].copy()
    ev["pos"] = pos[pos < len(dias)]
    ev["date"] = dias[ev["pos"]]
    j = px.columns.get_indexer(ev["symbol"])
    pxa, cierrea = px.to_numpy(), cierre.to_numpy()
    ev["base_mcap"] = pd.to_numeric(ev["acciones"], errors="coerce").to_numpy() * cierrea[ev["pos"], j] / pxa[ev["pos"], j]

    sv = (dv / dv.rolling(60, min_periods=20).median()).to_numpy()
    resa = res.to_numpy()
    ear, pos_a = np.full(len(ev), np.nan), np.full(len(ev), np.nan)
    for i, (p, c) in enumerate(zip(ev["pos"].to_numpy(), j)):
        lo, hi = max(1, p - 41), p - 1  # sesiones hasta la presentación (incluida)
        if hi - lo < 10 or np.all(np.isnan(sv[lo : hi + 1, c])):
            continue
        a = lo + int(np.nanargmax(sv[lo : hi + 1, c]))
        ear[i] = np.nansum(resa[a - 1 : a + 2, c])
        pos_a[i] = a
    ev["ear"], ev["pos_anuncio"] = ear, pos_a

    def ancho(col: str) -> pd.DataFrame:
        w = pd.to_numeric(ev[col], errors="coerce").groupby([ev["date"], ev["symbol"]]).last().unstack()
        return w.reindex(index=dias, columns=px.columns).astype(float).ffill(limit=300)

    mcap = ancho("base_mcap") * px
    mcap = mcap.where((mcap >= 5e8) & (mcap <= 5e12))  # acciones en circulación mal etiquetadas en `dei`
    patrimonio = ancho("patrimonio")
    capital = patrimonio + ancho("deuda_lp").fillna(0)
    tangible = patrimonio - ancho("fondo_comercio").fillna(0) - ancho("intangibles").fillna(0)
    posd = pd.DataFrame(np.arange(len(dias))[:, None].repeat(len(px.columns), 1), index=dias, columns=px.columns)
    f = {
        "div_yield": div_yield,
        "fcf_ic": ((ancho("ocf_ttm") - ancho("capex_ttm").fillna(0)) / capital).where(capital > 0),
        "bp_tangible": tangible / mcap,
        "ey": ancho("ni_ttm") / mcap,
        "roe": (ancho("ni_ttm") / patrimonio).where(patrimonio > 0),
        "gp_activos": ancho("gp_ttm") / ancho("activos"),
        "crec_activos": ancho("crec_activos"),
        "crec_bpa": ancho("dif_eps_ttm") / cierre,
        "sue": ancho("sue").clip(-10, 10),
        "ear": ancho("ear"),
        "dias_anuncio": posd - ancho("pos_anuncio"),
        "log_mcap": np.log(mcap.where(mcap > 0)),
    }
    largo = pd.concat({k: v.stack(future_stack=True) for k, v in f.items()}, axis=1).replace([np.inf, -np.inf], np.nan)
    largo.index.names = ["date", "symbol"]
    p = panel.merge(largo.reset_index(), on=["date", "symbol"], how="left")
    # Compuestos para exposiciones, atribución y carteras factoriales (como los combos de SG).
    p["valor"] = _rango(p, ["ey", "bp_tangible"])
    p["calidad"] = _rango(p, ["roe", "fcf_ic", "gp_activos"])
    p["crecimiento"] = _rango(p, ["crec_bpa"])
    p.to_parquet(ruta)
    return p
