"""Estabilidad SHAP bloque a bloque frente a variables sombra (A.6)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from . import config as C
from .cv import bloques, cpcv
from .modelos import GBM

FONDO = int(C.PREREG["estabilidad_shap"]["shap"].split("fondo de ")[1].split(" ")[0])


def _shap_particion(Xv, y, w, tr, te, semillas, i):
    m = GBM().fit(Xv[tr], y[tr], w[tr])
    rng = np.random.default_rng(C.SEMILLA + i)
    fondo = Xv[tr][rng.choice(len(tr), size=min(FONDO, len(tr)), replace=False)]
    sv = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for mod in m.ms:
            masker = shap.maskers.Independent(fondo, max_samples=len(fondo))
            ex = shap.TreeExplainer(mod, data=masker, feature_perturbation="interventional")
            sv.append(ex.shap_values(Xv[te], check_additivity=False))
    return np.mean(sv, axis=0)  # media de las semillas


def shap_por_bloque(X: pd.DataFrame, y, w, n_jobs: int = 4) -> list[pd.DataFrame]:
    """Una matriz SHAP por bloque: media de las particiones en que el bloque es de prueba."""
    Xv = X.to_numpy()
    parts = cpcv(len(X))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_shap_particion)(Xv, y, w, tr, te, C.SEMILLAS_GBM, i) for i, (_, tr, te) in enumerate(parts)
    )
    n, p = Xv.shape
    acum = np.zeros((n, p))
    cuenta = np.zeros(n)
    for (_, _, te), sv in zip(parts, res):
        acum[te] += sv
        cuenta[te] += 1
    S = acum / cuenta[:, None]
    return [pd.DataFrame(S[b], index=X.index[b], columns=X.columns) for b in bloques(n)]


def metricas_bloque(X: pd.DataFrame, S: pd.DataFrame, bins: int = 10) -> dict:
    res = {}
    for c in X.columns:
        x, s = X.loc[S.index, c], S[c]
        binaria = x.nunique() <= 2
        rho = spearmanr(x, s).correlation if s.std() > 0 else 0.0
        if binaria:
            curva = s.groupby(x).mean()
            curva = curva.reindex(sorted(x.unique()))
        else:
            q = pd.qcut(x.rank(method="first"), bins, labels=False)
            curva = s.groupby(q).mean()
        res[c] = dict(
            rho=float(0.0 if not np.isfinite(rho) else rho),
            curva=((curva - curva.mean()) / (curva.std() + 1e-12)).to_numpy(),
            curva_bruta=curva.to_numpy(),
            dif_binaria=float(curva.iloc[-1] - curva.iloc[0]) if binaria else np.nan,
            mean_abs=float(s.abs().mean()),
            binaria=binaria,
        )
    return res


def estabilidad(por_bloque: list[dict], c: str) -> dict:
    rhos = np.array([b[c]["rho"] for b in por_bloque])
    if por_bloque[0][c]["binaria"]:
        difs = np.array([b[c]["dif_binaria"] for b in por_bloque])
        forma = float(np.mean(np.sign(difs) == np.sign(np.median(difs))))
    else:
        Cm = np.corrcoef(np.vstack([b[c]["curva"] for b in por_bloque]))
        forma = float(np.nanmean(Cm[np.triu_indices_from(Cm, 1)]))
    sg = np.sign(rhos)
    return dict(
        signo_bloques=sg.astype(int).tolist(),
        signo_constante=bool(np.all(sg == sg[0]) and sg[0] != 0),
        signo_dominante=int(np.sign(np.sum(sg))),
        monotonia_min=float(np.abs(rhos).min()),
        forma_media=forma,
        relevancia=float(np.mean([b[c]["mean_abs"] for b in por_bloque])),
    )
