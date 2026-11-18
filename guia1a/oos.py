"""Predicciones fuera de muestra en CPCV y walk-forward, y la puerta de skill (A.6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as C
from .cv import bloques, cpcv, trayectorias, walk_forward
from .estad import ic_t, r2_oos


def _ajusta(fabrica, X, y, w, tr, te):
    m = fabrica().fit(X[tr], y[tr], w[tr])
    return m.predict(X[te]), float(np.average(y[tr], weights=w[tr])), m


def cpcv_pred(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, fabrica, n_jobs: int = 4, devolver_modelos=False):
    """Devuelve dict con: M (n x particiones, NaN fuera de prueba), bench (idem), agregada (media por fecha),
    bench_agregado, tray (n x trayectorias) y, opcionalmente, los modelos."""
    Xv = X.to_numpy()
    parts = cpcv(len(X))
    res = Parallel(n_jobs=n_jobs)(delayed(_ajusta)(fabrica, Xv, y, w, tr, te) for _, tr, te in parts)
    n, k = len(X), len(parts)
    M = np.full((n, k), np.nan)
    Bm = np.full((n, k), np.nan)
    for i, ((_, _, te), (p, mu, _)) in enumerate(zip(parts, res)):
        M[te, i] = p
        Bm[te, i] = mu
    B = bloques(n)
    tray = np.full((n, len(trayectorias())), np.nan)
    for j, mapa in enumerate(trayectorias()):
        for b, i in mapa.items():
            tray[B[b], j] = M[B[b], i]
    out = dict(M=M, bench=Bm, agregada=np.nanmean(M, 1), bench_agregado=np.nanmean(Bm, 1), tray=tray, parts=parts)
    if devolver_modelos:
        out["modelos"] = [r[2] for r in res]
    return out


def media_historica(y: pd.Series) -> pd.Series:
    """Media expansiva de las etiquetas ya realizadas en t (la de t-h es la última conocida)."""
    return y.expanding(252).mean().shift(C.H)


def wf_pred(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, fabrica, n_jobs: int = 4) -> pd.Series:
    Xv = X.to_numpy()
    parts = walk_forward(X.index)
    res = Parallel(n_jobs=n_jobs)(delayed(_ajusta)(fabrica, Xv, y, w, tr, te) for _, tr, te in parts)
    pred = pd.Series(np.nan, index=X.index)
    for (_, _, te), (p, _, _) in zip(parts, res):
        pred.iloc[te] = p
    return pred.dropna()


def puerta(pred: np.ndarray, y: np.ndarray, bench: np.ndarray) -> dict:
    ic, t = ic_t(pred, y)
    return dict(ic=ic, t_ic=t, r2_oos=r2_oos(y, pred, bench))
