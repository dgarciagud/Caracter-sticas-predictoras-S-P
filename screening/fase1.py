"""Fase 1 · screening OR: C1 (ICIR incondicional), C2 (interacción con estados) y C3 (IC latente predecible).

Todo se calcula en los bloques de test de los folds de `linea1a.etapa1.folds` y se resume con la media de folds.
La nula de cada criterio es el máximo sobre candidatas (y estados en C2) de las sombras circulares de X.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from linea1a import config as C
from linea1a.estadistica import nw_lags, pendiente_t
from linea1a.etapa1 import Periodo, folds, indices_sombra, offsets

ESTADOS = ["R_vol", "R_trend"]
Q_REJILLA = np.logspace(-4, 0, 13)


# ------------------------------------------------------------------ C2: OLS con interacción y t Newey-West por lotes
def t_interaccion(x: np.ndarray, s: np.ndarray, y: np.ndarray, m: np.ndarray, lags: int) -> np.ndarray:
    """t NW del coeficiente de x·s en y = a + b x + c s + d x·s. x, m: (B, n); s, y: (n,)."""
    B, n = x.shape
    w = m.astype(float)
    Xd = np.stack([np.ones_like(x), x, np.broadcast_to(s, x.shape), x * s], -1) * w[..., None]  # (B, n, 4)
    Xd = np.nan_to_num(Xd)
    yy = np.nan_to_num(np.broadcast_to(y, x.shape) * w)
    XtX = np.einsum("bni,bnj->bij", Xd, Xd)
    ok = np.linalg.matrix_rank(XtX) == 4
    XtX[~ok] = np.eye(4)
    inv = np.linalg.inv(XtX)
    beta = np.einsum("bij,bj->bi", inv, np.einsum("bni,bn->bi", Xd, yy))
    e = (yy - np.einsum("bni,bi->bn", Xd, beta)) * w
    u = Xd * e[..., None]
    S = np.einsum("bni,bnj->bij", u, u)
    for lag in range(1, lags + 1):
        g = np.einsum("bni,bnj->bij", u[:, lag:], u[:, :-lag])
        S = S + (1 - lag / (lags + 1)) * (g + np.transpose(g, (0, 2, 1)))
    V = inv @ S @ inv
    t = beta[:, 3] / np.sqrt(np.maximum(V[:, 3, 3], 1e-300))
    t[~ok] = np.nan
    return t


# ------------------------------------------------------------------ C3: nivel local (Kalman) sobre el IC instantáneo
@njit(cache=True)
def _loglik_nivel_local(z, valido, q):
    """Log-verosimilitud concentrada (varianza de medida = 1, factor de escala concentrado)."""
    a, P = 0.0, 1e6
    suma_logF, suma_v2F, n = 0.0, 0.0, 0
    for t in range(z.shape[0]):
        P = P + q
        if valido[t]:
            F = P + 1.0
            v = z[t] - a
            if n > 0:  # el primer paso es difuso
                suma_logF += np.log(F)
                suma_v2F += v * v / F
            n += 1
            K = P / F
            a = a + K * v
            P = P * (1 - K)
    if n < 3:
        return -1e300
    return -0.5 * (suma_logF + (n - 1) * np.log(suma_v2F / (n - 1)))


@njit(cache=True)
def _predicciones(z, valido, q):
    """a_{t|t-1} para todas las t (predicción a un paso del nivel)."""
    out = np.empty(z.shape[0])
    a, P = 0.0, 1e6
    for t in range(z.shape[0]):
        P = P + q
        out[t] = a
        if valido[t]:
            F = P + 1.0
            K = P / F
            a = a + K * (z[t] - a)
            P = P * (1 - K)
    return out


@njit(cache=True)
def _c3_serie(z, valido, n_tr, qs):
    """Elige q por máxima verosimilitud en el entrenamiento [0, n_tr) y devuelve predicciones y q."""
    mejor, q_opt = -1e301, qs[0]
    for q in qs:
        ll = _loglik_nivel_local(z[:n_tr], valido[:n_tr], q)
        if ll > mejor:
            mejor, q_opt = ll, q
    return _predicciones(z, valido, q_opt), q_opt


def c3_fold(x: np.ndarray, y: np.ndarray, m: np.ndarray, tr_end: int, te: np.ndarray) -> tuple[float, float]:
    """IC instantáneo z = x~·y~ (estandarizados con el entrenamiento); estadístico = t NW de sign(a_{t|t-1})·z en test."""
    tr = np.arange(tr_end)
    ok_tr = m[tr]
    if ok_tr.sum() < 52 or m[te].sum() < 30:
        return np.nan, np.nan
    mx, sx = x[tr][ok_tr].mean(), x[tr][ok_tr].std()
    my, sy = y[tr][ok_tr].mean(), y[tr][ok_tr].std()
    z = np.where(m, (x - mx) / sx * (y - my) / sy, 0.0)
    pred, q = _c3_serie(z, m, tr_end, Q_REJILLA)
    g = np.sign(pred[te]) * z[te]
    g = g[m[te]]
    se = np.sqrt(_nw_var_media(g))
    return float(g.mean() / se), float(q)


def _nw_var_media(g: np.ndarray) -> float:
    n = len(g)
    L = nw_lags(n)
    u = g - g.mean()
    s = (u * u).sum()
    for lag in range(1, L + 1):
        s += 2 * (1 - lag / (L + 1)) * (u[lag:] * u[:-lag]).sum()
    return s / n**2


# ------------------------------------------------------------------ estadísticos de una matriz X (real o sombra)
def estadisticos(per: Periodo, X: np.ndarray, okm: np.ndarray, fl, con_c3: bool = True) -> dict:
    """C1 (p,), C2 (p,) = máximo sobre estados, C3 (p,), q de C3 (p,). X: (T, p); okm: (T,) máscara de la sombra."""
    T, p = X.shape
    signos = np.array([C.SIGNOS[v] for v in per.nombres])
    reg = per.reg
    c1_f, c2_f, c3_f, q_f = [], [], [], []
    for tr, te in fl:
        filas = te[per.ym[te] & okm[te]]
        n = len(filas)
        x = X[filas].T  # (p, n)
        m = np.isfinite(x)
        y = per.y[filas]
        _, t, _ = pendiente_t(x, y, m, nw_lags(n))
        c1_f.append(np.where(signos != 0, signos * t, np.abs(t)))
        t_s = []
        for s in ESTADOS:
            sv = reg[s].to_numpy(float)[filas]
            t_s.append(t_interaccion(np.where(m, x, 0.0), sv, y, m & np.isfinite(sv), nw_lags(n)))
        c2_f.append(np.stack(t_s, -1))  # (p, estados)
        if con_c3:
            tr_end = int(te[0])
            c3v, qv = [], []
            for j in range(p):
                mj = per.ym & okm & np.isfinite(X[:, j])
                a, q = c3_fold(np.nan_to_num(X[:, j]), np.nan_to_num(per.y), mj, tr_end, te)
                c3v.append(a)
                qv.append(q)
            c3_f.append(c3v)
            q_f.append(qv)
    with np.errstate(all="ignore"):
        c1 = np.nanmean(c1_f, 0)
        c2 = np.nanmax(np.abs(np.nanmean(np.stack(c2_f), 0)), -1)  # |media de folds| y máximo sobre estados
        c3 = np.nanmean(c3_f, 0) if con_c3 else np.full(p, np.nan)
        q = np.nanmedian(q_f, 0) if con_c3 else np.full(p, np.nan)
    return dict(C1=c1, C2=c2, C3=c3, q=q)


def fase1(per: Periodo, R: int = 500, semilla: int = 0, p_s: float = 80.0, con_c3: bool = True) -> dict:
    T = len(per.fechas)
    fl = folds(per)
    real = estadisticos(per, per.X, np.ones(T, bool), fl, con_c3)
    maximos = {c: [] for c in ("C1", "C2", "C3")}
    for k in offsets(T, R, semilla):
        idx, ok = indices_sombra(T, int(k))
        s = estadisticos(per, per.X[idx], ok, fl, con_c3)
        for c in maximos:
            maximos[c].append(np.nanmax(s[c]) if np.isfinite(s[c]).any() else np.nan)
    maximos = {c: np.array(v) for c, v in maximos.items()}
    return dict(real=real, maximos=maximos, tabla=tabla_fase1(per.nombres, real, maximos, p_s), folds=fl)


def tabla_fase1(nombres, real, maximos, p_s: float) -> pd.DataFrame:
    t = pd.DataFrame({"variable": nombres, "q_C3": real["q"]})
    pasa = np.zeros(len(nombres), bool)
    for c in ("C1", "C2", "C3"):
        u = float(np.nanpercentile(maximos[c], p_s)) if np.isfinite(maximos[c]).any() else np.nan
        t[c], t[f"umbral_{c}"] = real[c], u
        t[f"pasa_{c}"] = np.isfinite(real[c]) & (real[c] > u)
        pasa |= t[f"pasa_{c}"].to_numpy()
    t["pasa_OR"] = pasa
    return t
