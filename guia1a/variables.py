"""Variables candidatas, regímenes, objetivo y pesos (guía, apartados A.3 fase 0 y A.4).

Cada columna en la fecha t usa solo datos hasta el cierre de t. La estandarización es temporal:
winsorización 1/99 con percentiles expansivos del pasado y z-score expansivo (mínimo 252 sesiones).
El test anti look-ahead (tests/test_lookahead.py) recorta los datos en una fecha y comprueba que
todo lo anterior sale idéntico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .datos import INDUSTRIAS, fomc

MINP = int(C.PREREG["estandarizacion"]["minimo_sesiones"])
P_WINS = float(C.PREREG["estandarizacion"]["winsor"])


# ---------------------------------------------------------------- estandarización (A.4)
def wins_past(s: pd.Series, p: float = P_WINS, minp: int = MINP) -> pd.Series:
    lo = s.expanding(minp).quantile(p)
    hi = s.expanding(minp).quantile(1 - p)
    return s.clip(lo, hi)


def z_expanding(s: pd.Series, minp: int = MINP) -> pd.Series:
    return (s - s.expanding(minp).mean()) / s.expanding(minp).std()


def estandarizar(s: pd.Series) -> pd.Series:
    return z_expanding(wins_past(s))


# ---------------------------------------------------------------- piezas
def _rv(r: pd.Series, n: int) -> pd.Series:
    """Volatilidad realizada anualizada (raíz de la media de r^2, sin restar la media)."""
    return np.sqrt((r**2).rolling(n, min_periods=n).mean() * 252)


def _ventana_futura(evento: pd.Series, h: int) -> pd.Series:
    """1 si algún día de (t, t+h] (sesiones) tiene el evento. El calendario es conocido ex ante."""
    v = evento.astype(float)
    fwd = v[::-1].rolling(h, min_periods=1).sum()[::-1].shift(-1)
    return (fwd > 0).astype(float)


def _tercer_viernes(idx: pd.DatetimeIndex) -> pd.Series:
    """Última sesión el o antes del tercer viernes de cada mes (vencimiento mensual de opciones)."""
    s = pd.Series(0, index=idx)
    for (y, m), g in s.groupby([idx.year, idx.month]):
        primero = pd.Timestamp(y, m, 1)
        tv = primero + pd.Timedelta(days=(4 - primero.weekday()) % 7 + 14)
        cand = g.index[g.index <= tv]
        # Solo si el calendario de sesiones ya llega al tercer viernes (si no, la última sesión
        # disponible se confundiría con el vencimiento: fallo detectado por el test anti look-ahead).
        if len(cand) and idx.max() >= tv:
            s.loc[cand[-1]] = 1
    return s


# ---------------------------------------------------------------- candidatas en bruto
def brutas(d: pd.DataFrame) -> pd.DataFrame:
    p, lp = d["close"], np.log(d["close"])
    r = lp.diff()
    sig_d = r.ewm(span=C.SPAN_SIGMA, min_periods=C.SPAN_SIGMA).std()
    sig5 = sig_d * np.sqrt(C.H)
    sma = lambda n: p.rolling(n, min_periods=n).mean()  # noqa: E731
    rv5, rv21, rv252 = _rv(r, 5), _rv(r, 21), _rv(r, 252)
    vix = d["VIXCLS"]
    f = pd.DataFrame(index=d.index)
    # Tendencia
    f["TEN_mom12_1"] = lp.shift(21) - lp.shift(252)
    f["TEN_sma50_200"] = sma(50) / sma(200) - 1
    f["TEN_mom1m"] = lp - lp.shift(21)
    # Reversión
    f["REV_r5"] = (lp - lp.shift(5)) / sig5
    f["REV_z20"] = (p - sma(20)) / p.rolling(20, min_periods=20).std()
    f["REV_r1"] = r / sig_d
    # Forma de la volatilidad
    f["VOL_rvratio"] = rv5 / rv21
    f["VOL_vrp"] = (vix / 100) ** 2 - rv21**2
    f["VOL_dvix5"] = np.log(vix / vix.shift(5))
    f["VOL_rvterm"] = np.log(rv21 / rv252)
    neg = (r.clip(upper=0) ** 2).rolling(21, min_periods=21).sum()
    f["VOL_downshare"] = neg / (r**2).rolling(21, min_periods=21).sum()
    park = np.sqrt((np.log(d["high"] / d["low"]) ** 2).rolling(21, min_periods=21).mean() / (4 * np.log(2)) * 252)
    f["VOL_rangerv"] = np.log(park / rv21)
    # Calendario (conocido ex ante)
    s = d.index
    es_fomc = pd.Series(s.isin(fomc()), index=s)
    primera_mes = pd.Series(np.r_[True, s.month[1:] != s.month[:-1]], index=s)
    f["CAL_fomc"] = _ventana_futura(es_fomc, C.H)
    f["CAL_tom"] = _ventana_futura(primera_mes, C.H)
    f["CAL_opex"] = _ventana_futura(_tercer_viernes(s), C.H)
    # Volumen
    lv = np.log(d["volume"])
    f["VOLU_abn"] = lv - lv.shift(1).rolling(50, min_periods=50).mean()
    sv = (np.sign(r) * d["volume"]).rolling(20, min_periods=20).sum()
    f["VOLU_signed"] = sv / d["volume"].rolling(20, min_periods=20).sum()
    # Sectores y amplitud (12 industrias de French)
    ind = d[[f"ind_{c}" for c in INDUSTRIAS]]
    ind.columns = INDUSTRIAS
    lind = np.log1p(ind).rolling(21, min_periods=21).sum()
    f["SEC_defcyc"] = lind[["NoDur", "Utils", "Hlth"]].mean(1) - lind[["Durbl", "Manuf", "BusEq"]].mean(1)
    f["SEC_disp"] = lind.std(1)
    nivel = np.log1p(ind).cumsum()
    f["AMP_ma50"] = (nivel > nivel.rolling(50, min_periods=50).mean()).mean(axis=1).where(nivel.notna().all(axis=1))
    f["AMP_corr"] = _correlacion_media(ind, 63)
    # Crédito y tipos
    f["CRE_dbaa21"] = d["BAA10Y"] - d["BAA10Y"].shift(21)
    f["TIP_slope"] = d["T10Y3M"]
    return f[list(C.CANDIDATAS)]


def _correlacion_media(ind: pd.DataFrame, n: int) -> pd.Series:
    """Correlación media entre pares de industrias en las n sesiones previas (Pollet y Wilson, 2010)."""
    x = ind.to_numpy()
    out = np.full(len(x), np.nan)
    iu = np.triu_indices(x.shape[1], 1)
    for t in range(n - 1, len(x)):
        w = x[t - n + 1 : t + 1]
        if np.isfinite(w).all():
            out[t] = np.corrcoef(w.T)[iu].mean()
    return pd.Series(out, index=ind.index)


def regimenes(d: pd.DataFrame) -> pd.DataFrame:
    p, lp = d["close"], np.log(d["close"])
    r = lp.diff()
    g = pd.DataFrame(index=d.index)
    g["REG_rv20"] = np.log(_rv(r, 20))
    g["REG_vix"] = np.log(d["VIXCLS"])
    g["REG_trend"] = (p > p.rolling(200, min_periods=200).mean()).astype(float).where(p.rolling(200).count() == 200)
    g["REG_dd"] = p / p.rolling(252, min_periods=252).max() - 1
    return g[list(C.REGIMENES)]


def objetivo(d: pd.DataFrame) -> pd.DataFrame:
    """y = (P[t+h]/P[t] - 1 - rf) / sigma_h(t); sigma con EWMA de 60 sesiones conocida en t."""
    p = d["close"]
    r = np.log(p).diff()
    sigma_h = r.ewm(span=C.SPAN_SIGMA, min_periods=C.SPAN_SIGMA).std() * np.sqrt(C.H)
    rf_d = d["DFF"].ffill() / 100 / 252
    fwd = p.shift(-C.H) / p - 1
    rf_h = rf_d * C.H
    o = pd.DataFrame(index=d.index)
    o["ret_h"] = fwd
    o["rf_h"] = rf_h
    o["sigma_h"] = sigma_h
    o["y"] = (fwd - rf_h) / sigma_h
    o["ret_d"] = p.pct_change()
    o["rf_d"] = rf_d
    return o


def unicidad(n: int, h: int = C.H) -> np.ndarray:
    """Unicidad media de etiquetas solapadas de h sesiones (López de Prado, 2018, cap. 4)."""
    conc = np.zeros(n + h)
    for i in range(n):
        conc[i + 1 : i + 1 + h] += 1
    return np.array([np.mean(1 / conc[i + 1 : i + 1 + h]) for i in range(n)])


def panel(d: pd.DataFrame) -> pd.DataFrame:
    """Panel diario: candidatas estandarizadas, regímenes estandarizados, interacciones y objetivo."""
    B = brutas(d)
    X = B.apply(estandarizar)
    # Binarias de calendario: se dejan en 0/1 (la estandarización no cambia su orden).
    for c in ("CAL_fomc", "CAL_tom", "CAL_opex"):
        X[c] = B[c]
    Rg = regimenes(d)
    R = Rg.apply(estandarizar)
    R["REG_trend"] = Rg["REG_trend"]
    for nombre, spec in C.INTERACCIONES.items():
        R[nombre] = z_expanding(X[spec["x"]] * R[spec["r"]])
    o = objetivo(d)
    out = pd.concat([X, R, o], axis=1)
    # Calendario incompleto al final de la muestra: sin ventana (t, t+h] completa no hay dato.
    out.loc[o["ret_h"].isna(), ["CAL_fomc", "CAL_tom", "CAL_opex"]] = np.nan
    return out
