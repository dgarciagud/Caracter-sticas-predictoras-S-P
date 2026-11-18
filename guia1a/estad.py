"""Estadística: pendiente con t de Newey-West (vectorizada), medias HAC, Clark-West, R2 OOS,
diferencia de Sharpe con HAC (Ledoit y Wolf, 2008), alfa de spanning y Benjamini-Hochberg."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C


def pendiente_t(x: np.ndarray, y: np.ndarray, m: np.ndarray, lags: int = C.NW_LAGS):
    """OLS y = a + b x con t de b Newey-West, vectorizado sobre los ejes iniciales de x y m."""
    w = m.astype(float)
    x = np.where(m, x, 0.0)
    y = np.where(m, np.broadcast_to(y, x.shape), 0.0)
    nv = w.sum(-1)
    xm = (w * x).sum(-1) / nv
    ym = (w * y).sum(-1) / nv
    xc = (x - xm[..., None]) * w
    yc = (y - ym[..., None]) * w
    sxx = (xc * xc).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (xc * yc).sum(-1) / sxx
        u = xc * (yc - b[..., None] * xc)
        s = (u * u).sum(-1)
        for lag in range(1, lags + 1):
            s = s + 2 * (1 - lag / (lags + 1)) * (u[..., lag:] * u[..., :-lag]).sum(-1)
        t = b / np.sqrt(s / sxx**2)
    return b, t


def media_nw(z, lags: int = C.NW_LAGS) -> tuple[float, float]:
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    n = len(z)
    u = z - z.mean()
    s = (u * u).sum()
    for lag in range(1, lags + 1):
        s += 2 * (1 - lag / (lags + 1)) * (u[lag:] * u[:-lag]).sum()
    se = np.sqrt(max(s, 1e-300)) / n
    return float(z.mean()), float(z.mean() / se)


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int = C.NW_LAGS) -> tuple[np.ndarray, np.ndarray]:
    """OLS con constante (columna 0 de la salida) y t HAC."""
    import statsmodels.api as sm

    r = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return np.asarray(r.params), np.asarray(r.tvalues)


def r2_oos(y, f, bench) -> float:
    return float(1 - np.sum((y - f) ** 2) / np.sum((y - bench) ** 2))


def clark_west(y, bench, f) -> float:
    adj = (y - bench) ** 2 - ((y - f) ** 2 - (bench - f) ** 2)
    return media_nw(adj)[1]


def dm_t(y, f_ant, f_nuevo) -> tuple[float, float]:
    """Diferencia de pérdidas cuadráticas e_ant^2 - e_nuevo^2: (media, t NW). > 0 = mejora el nuevo."""
    return media_nw((y - f_ant) ** 2 - (y - f_nuevo) ** 2)


def ic_t(pred, y) -> tuple[float, float]:
    """IC de Spearman y su t NW (pendiente de rangos estandarizados)."""
    a = pd.Series(pred).rank().to_numpy()
    b = pd.Series(y).rank().to_numpy()
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    _, t = pendiente_t(a[None, :], b, np.ones((1, len(a)), bool))
    return float(np.mean(a * b)), float(t[0])


def sharpe(r, ppy: int = 252) -> float:
    r = np.asarray(r, float)
    return float(np.nanmean(r) / np.nanstd(r, ddof=1) * np.sqrt(ppy))


def dif_sharpe_hac(r1, r2, lags: int = C.NW_LAGS, ppy: int = 252) -> tuple[float, float]:
    """Diferencia de Sharpe anualizado (r1 - r2) y su t con el método delta HAC de Ledoit y Wolf (2008)."""
    r1, r2 = np.asarray(r1, float), np.asarray(r2, float)
    ok = np.isfinite(r1) & np.isfinite(r2)
    r1, r2 = r1[ok], r2[ok]
    m1, m2 = r1.mean(), r2.mean()
    g1, g2 = (r1**2).mean(), (r2**2).mean()
    V = np.column_stack([r1 - m1, r2 - m2, r1**2 - g1, r2**2 - g2])
    n = len(V)
    S = V.T @ V / n
    for lag in range(1, lags + 1):
        G = V[lag:].T @ V[:-lag] / n
        S += (1 - lag / (lags + 1)) * (G + G.T)
    s1, s2 = np.sqrt(g1 - m1**2), np.sqrt(g2 - m2**2)
    d = m1 / s1 - m2 / s2
    grad = np.array([g1 / s1**3, -g2 / s2**3, -m1 / (2 * s1**3), m2 / (2 * s2**3)])
    se = np.sqrt(grad @ S @ grad / n)
    return float(d * np.sqrt(ppy)), float(d / se)


def bh(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg: True donde se rechaza."""
    p = np.asarray(p, float)
    n = len(p)
    orden = np.argsort(p)
    lim = q * np.arange(1, n + 1) / n
    pasa = p[orden] <= lim
    k = np.max(np.where(pasa)[0]) + 1 if pasa.any() else 0
    out = np.zeros(n, bool)
    out[orden[:k]] = True
    return out


def p_normal(t: float, signo: int) -> float:
    return float(1 - stats.norm.cdf(t * signo)) if signo else float(2 * (1 - stats.norm.cdf(abs(t))))
