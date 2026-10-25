"""Estadísticos vectorizados: pendiente con t Newey-West, monotonía por quintiles, Clark-West."""

from __future__ import annotations

import numpy as np
from scipy import stats


def nw_lags(n: int) -> int:
    """Regla de Newey-West (1994): floor(4 (n/100)^(2/9))."""
    return int(np.floor(4 * (max(n, 1) / 100) ** (2 / 9)))


def pendiente_t(x: np.ndarray, y: np.ndarray, m: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS y = a + b x con t de b Newey-West, vectorizado sobre los ejes iniciales.

    x, m: (..., n); y: (n,) o (..., n). Las filas con m=False no cuentan (residuo 0).
    """
    w = m.astype(float)
    x = np.where(m, x, 0.0)
    y = np.where(m, y, 0.0)
    nv = w.sum(-1)
    xm = (w * x).sum(-1) / nv
    ym = (w * y).sum(-1) / nv
    xc = (x - xm[..., None]) * w
    yc = (y - ym[..., None]) * w
    sxx = (xc * xc).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (xc * yc).sum(-1) / sxx
        e = yc - b[..., None] * xc
        u = xc * e
        s = (u * u).sum(-1)
        for lag in range(1, lags + 1):
            s = s + 2 * (1 - lag / (lags + 1)) * (u[..., lag:] * u[..., :-lag]).sum(-1)
        t = b / np.sqrt(s / sxx**2)
    return b, t, nv


def monotonia_quintiles(x: np.ndarray, y: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Spearman entre la media de y por quintil de x y el número de quintil.

    Empates de x se rompen por orden temporal (rango ordinal estable).
    """
    xr = np.where(m, x, np.inf)
    rango = np.argsort(np.argsort(xr, axis=-1, kind="stable"), axis=-1, kind="stable")
    nv = m.sum(-1, keepdims=True)
    q = np.floor(rango * 5 / np.maximum(nv, 1)).astype(int)
    medias = []
    yy = np.broadcast_to(y, x.shape)
    for k in range(5):
        sel = (q == k) & m
        with np.errstate(invalid="ignore", divide="ignore"):
            medias.append((sel * np.where(m, yy, 0.0)).sum(-1) / sel.sum(-1))
    medias = np.stack(medias, -1)
    rm = np.argsort(np.argsort(medias, axis=-1), axis=-1)
    d = rm - np.arange(5)
    rho = 1 - 6 * (d**2).sum(-1) / (5 * 24)
    return np.where(np.isfinite(medias).all(-1), rho, np.nan)


def spearman_enmascarado(x: np.ndarray, y: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Correlación de rangos x-y vectorizada sobre los ejes iniciales; solo filas con m=True."""
    yy = np.broadcast_to(y, x.shape)
    rx = np.argsort(np.argsort(np.where(m, x, np.inf), axis=-1, kind="stable"), axis=-1).astype(float)
    ry = np.argsort(np.argsort(np.where(m, yy, np.inf), axis=-1, kind="stable"), axis=-1).astype(float)
    w = m.astype(float)
    nv = w.sum(-1, keepdims=True)
    rx = np.where(m, rx - (w * rx).sum(-1, keepdims=True) / nv, 0.0)
    ry = np.where(m, ry - (w * ry).sum(-1, keepdims=True) / nv, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx * ry).sum(-1) / np.sqrt((rx * rx).sum(-1) * (ry * ry).sum(-1))


def media_nw(z: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """Media de z con error estándar Newey-West; devuelve (media, t)."""
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    n = len(z)
    lags = nw_lags(n) if lags is None else lags
    u = z - z.mean()
    s = (u * u).sum()
    for lag in range(1, lags + 1):
        s += 2 * (1 - lag / (lags + 1)) * (u[lag:] * u[:-lag]).sum()
    se = np.sqrt(s) / n
    return float(z.mean()), float(z.mean() / se)


def clark_west(y: np.ndarray, f_bench: np.ndarray, f_model: np.ndarray) -> tuple[float, float]:
    """Test de Clark-West (2007), unilateral; devuelve (t, p)."""
    adj = (y - f_bench) ** 2 - ((y - f_model) ** 2 - (f_bench - f_model) ** 2)
    _, t = media_nw(adj)
    return t, float(1 - stats.norm.cdf(t))


def r2_oos(y: np.ndarray, f_model: np.ndarray, f_bench: np.ndarray) -> float:
    return float(1 - np.sum((y - f_model) ** 2) / np.sum((y - f_bench) ** 2))


def pendiente_stambaugh(x: np.ndarray, y: np.ndarray, lags: int) -> dict:
    """U4: pendiente sobre todo el periodo con NW; corrección de Stambaugh si AR(1) de x > 0.9."""
    ok = np.isfinite(x) & np.isfinite(y)
    b, t, n = pendiente_t(x[None, :], y, ok[None, :], lags)
    b, t, n = float(b[0]), float(t[0]), int(n[0])
    xs = x[ok]
    rho = float(np.corrcoef(xs[1:], xs[:-1])[0, 1])
    out = {"b": b, "t": t, "n": n, "ar1_x": rho, "stambaugh": False}
    if rho > 0.9:
        # Stambaugh (1999): sesgo de b = (cov(u,v)/var(v)) * E[rho_hat - rho], E ~ -(1+3 rho)/T.
        ys = y[ok]
        xa, xb, ya = xs[:-1], xs[1:], ys[:-1]
        u = ya - np.polyval(np.polyfit(xa, ya, 1), xa)
        v = xb - np.polyval(np.polyfit(xa, xb, 1), xa)
        sesgo = (np.cov(u, v)[0, 1] / np.var(v, ddof=1)) * (-(1 + 3 * rho) / len(xa))
        out.update(b=b - sesgo, t=t * (b - sesgo) / b if b != 0 else t, stambaugh=True)
    out["p_bilateral"] = float(2 * (1 - stats.norm.cdf(abs(out["t"]))))
    return out
