"""Fase 2 · contribución fuera de muestra: SFI con variables de estado y MDA clusterizado, contra sombras.

Modelo: LightGBM con los parámetros fijos de la 1A; pérdida MSE del objetivo en los bloques de test de cada fold.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from linea1a import config as C
from linea1a.etapa1 import Periodo, clusters, folds, indices_sombra, offsets

from .fase1 import ESTADOS


def _mse_fold(Xtr, ytr, Xte, yte, perm_grupos=None, rng=None, n_perm=3):
    """MSE de test y, si se piden, aumentos del MSE al permutar juntos los grupos de columnas indicados."""
    m = lgb.LGBMRegressor(**C.LGBM_PARAMS, random_state=C.SEMILLA).fit(Xtr, ytr)
    base = float(np.mean((yte - m.predict(Xte)) ** 2))
    if perm_grupos is None:
        return base, None
    aumentos = []
    for cols in perm_grupos:
        d = []
        for _ in range(n_perm):
            Xp = Xte.copy()
            Xp[:, cols] = Xte[rng.permutation(len(Xte))][:, cols]
            d.append(float(np.mean((yte - m.predict(Xp)) ** 2)) - base)
        aumentos.append(np.mean(d))
    return base, np.array(aumentos)


def _filas(per, tr, te, X):
    ok_tr = tr[per.ym[tr] & np.isfinite(X[tr]).all(1)]
    ok_te = te[per.ym[te] & np.isfinite(X[te]).all(1)]
    return ok_tr, ok_te


def sfi(per: Periodo, X: np.ndarray, fl) -> np.ndarray:
    """SFI_j = MSE(solo estados) - MSE({j} + estados), media de folds; X: (T, k)."""
    S = per.reg[ESTADOS].to_numpy(float)
    out = np.zeros((len(fl), X.shape[1]))
    for f, (tr, te) in enumerate(fl):
        for j in range(X.shape[1]):
            Z = np.column_stack([X[:, j], S])
            a, b = _filas(per, tr, te, Z)
            base, _ = _mse_fold(S[a], per.y[a], S[b], per.y[b])
            con, _ = _mse_fold(Z[a], per.y[a], Z[b], per.y[b])
            out[f, j] = base - con
    return out.mean(0)


def _replica(per, Xr, grupos, fl, k, semilla):
    """SFI de las sombras y MDA clusterizado del modelo conjunto [reales, sombras, estados] para un desplazamiento k."""
    T = len(per.fechas)
    idx, ok = indices_sombra(T, int(k))
    Xs = per.X[idx][:, [per.nombres.index(v) for v in grupos["variables"]]].copy()
    Xs[~ok] = np.nan
    sfi_s = sfi(per, Xs, fl)
    kr = Xr.shape[1]
    S = per.reg[ESTADOS].to_numpy(float)
    Z = np.column_stack([Xr, Xs, S])
    perm = [list(g) for g in grupos["indices"]] + [[kr + i for i in g] for g in grupos["indices"]]
    rng = np.random.default_rng(semilla)
    mda = []
    for tr, te in fl:
        a, b = _filas(per, tr, te, Z)
        _, aum = _mse_fold(Z[a], per.y[a], Z[b], per.y[b], perm, rng)
        mda.append(aum)
    mda = np.mean(mda, 0)
    ng = len(grupos["indices"])
    return dict(sfi_sombra=sfi_s, mda_real=mda[:ng], mda_sombra=mda[ng:])


def fase2(per: Periodo, variables: list[str], R: int = 20, semilla: int = 0, n_jobs: int = 4) -> dict:
    """Estadísticos reales y máximos de las sombras (los umbrales se fijan después con el percentil p)."""
    fl = folds(per)
    j = [per.nombres.index(v) for v in variables]
    Xr = per.X[:, j]
    asign, _ = clusters(per.X[np.isfinite(per.X).all(1)], per.nombres)
    etiquetas = sorted({asign[v] for v in variables})
    indices = [[i for i, v in enumerate(variables) if asign[v] == c] for c in etiquetas]
    grupos = dict(variables=variables, indices=indices, clusters=etiquetas)
    sfi_real = sfi(per, Xr, fl)
    reps = Parallel(n_jobs=n_jobs)(
        delayed(_replica)(per, Xr, grupos, fl, kk, semilla + i) for i, kk in enumerate(offsets(len(per.fechas), R, semilla))
    )
    return dict(
        variables=variables,
        grupos=grupos,
        sfi_real=sfi_real,
        mda_real=np.median([r["mda_real"] for r in reps], 0),
        max_sfi_sombra=np.array([np.nanmax(r["sfi_sombra"]) for r in reps]),
        max_mda_sombra=np.array([np.nanmax(r["mda_sombra"]) for r in reps]),
    )


def seleccion(f2: dict | None, p: float) -> tuple[list[str], pd.DataFrame]:
    """Selecciona si el MDA de su cluster supera el umbral o su SFI supera el suyo; un representante por cluster."""
    if f2 is None or not f2["variables"]:
        return [], pd.DataFrame()
    u_sfi = float(np.nanpercentile(f2["max_sfi_sombra"], p))
    u_mda = float(np.nanpercentile(f2["max_mda_sombra"], p))
    filas = []
    for ci, (c, idx) in enumerate(zip(f2["grupos"]["clusters"], f2["grupos"]["indices"])):
        for i in idx:
            filas.append(dict(variable=f2["variables"][i], cluster=c, SFI=f2["sfi_real"][i], umbral_SFI=u_sfi,
                              MDA_cluster=f2["mda_real"][ci], umbral_MDA=u_mda))
    t = pd.DataFrame(filas)
    t["pasa_SFI"] = t.SFI > t.umbral_SFI
    t["pasa_MDA"] = t.MDA_cluster > t.umbral_MDA
    t["candidata"] = t.pasa_SFI | t.pasa_MDA
    t["seleccionada"] = False
    for c, g in t[t.candidata].groupby("cluster"):
        t.loc[g.SFI.idxmax(), "seleccionada"] = True
    return t[t.seleccionada].variable.tolist(), t
