"""Fase 2 · Redundancia y clustering (A.3) y clustered MDA (marco común; López de Prado, 2020)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from . import config as C
from .cv import cpcv
from .estad import pendiente_t
from .variables import z_expanding


def distancia(Pd: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rho = Pd.corr(method="spearman")
    Dv = np.sqrt(np.clip(0.5 * (1 - rho.abs().to_numpy()), 0, None))
    np.fill_diagonal(Dv, 0)
    return rho, pd.DataFrame(Dv, index=rho.index, columns=rho.columns)


def clusters(Pd: pd.DataFrame) -> tuple[dict[str, list[str]], dict]:
    cand = list(C.CANDIDATAS)
    rho, D = distancia(Pd[cand])
    Z = linkage(squareform(D.to_numpy(), checks=False), method="average")
    sil = {}
    for k in range(2, len(cand)):
        lab = fcluster(Z, k, criterion="maxclust")
        if len(set(lab)) < 2:
            continue
        sil[k] = float(silhouette_score(D.to_numpy(), lab, metric="precomputed"))
    k = max(sil, key=sil.get)
    lab = fcluster(Z, k, criterion="maxclust")
    grupos: dict[str, list[str]] = {}
    for c in cand:  # orden del pre-registro: el primer miembro da nombre y signo de alineación
        grupos.setdefault(int(lab[cand.index(c)]), []).append(c)
    out = {}
    for miembros in grupos.values():
        nombre = miembros[0] if len(miembros) == 1 else "CL_" + miembros[0]
        out[nombre] = miembros
    info = dict(k=k, silueta=sil[k], siluetas=sil, rho=rho, D=D, linkage=Z, cand=cand)
    if C.PREREG["clusters"].get("modo") == "individuales":
        # g3: sin compuestos; la partición por silueta solo se reporta (k_silueta).
        info.update(k_silueta=k, k=len(cand), modo="individuales")
        out = {c: [c] for c in cand}
    return out, info


def compuestos(P: pd.DataFrame, grupos: dict[str, list[str]], rho: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Compuesto = media de z alineados por signo con el primer miembro, re-estandarizada (z expansivo)."""
    X = pd.DataFrame(index=P.index)
    signo, alineacion = {}, {}
    for nombre, miembros in grupos.items():
        if len(miembros) == 1:
            X[nombre] = P[miembros[0]]
            signo[nombre] = C.SIGNOS[miembros[0]]
            alineacion[nombre] = {miembros[0]: 1}
            continue
        ref = miembros[0]
        al = {m: int(np.sign(rho.loc[ref, m])) if m != ref else 1 for m in miembros}
        X[nombre] = z_expanding(sum(al[m] * P[m] for m in miembros) / len(miembros))
        # Signo del compuesto: el del primer miembro con hipótesis, una vez alineado.
        sgs = [al[m] * C.SIGNOS[m] for m in miembros if C.SIGNOS[m]]
        signo[nombre] = sgs[0] if sgs and all(s == sgs[0] for s in sgs) else 0
        alineacion[nombre] = al
    return X, signo, alineacion


def vif(Xd: pd.DataFrame) -> pd.Series:
    R = np.corrcoef(Xd.to_numpy().T)
    return pd.Series(np.diag(np.linalg.pinv(R)), index=Xd.columns)


def t_simple(x: np.ndarray, y: np.ndarray) -> float:
    return float(pendiente_t(x[None], y, np.ones((1, len(x)), bool))[1][0])


def sombras(X: pd.DataFrame, k: int, semilla: int) -> pd.DataFrame:
    """k copias desplazadas circularmente de cada columna (>= 252 sesiones, fuera del 10 % de los bordes)."""
    rng = np.random.default_rng(semilla)
    n = len(X)
    lo = max(C.SOMBRA_MIN, int(np.ceil(C.SOMBRA_BORDE * n)))
    out = {}
    for c in X.columns:
        for j in range(k):
            out[f"{c}__sh{j}"] = np.roll(X[c].to_numpy(), rng.integers(lo, n - lo))
    return pd.DataFrame(out, index=X.index)


def _mda_particion(fabrica, Xv, y, w, tr, te, grupos_idx, n_perm, semilla):
    m = fabrica().fit(Xv[tr], y[tr], w[tr])
    base = np.average((y[te] - m.predict(Xv[te])) ** 2, weights=w[te])
    rng = np.random.default_rng(semilla)
    out = {}
    for g, cols in grupos_idx.items():
        dl = []
        for _ in range(n_perm):
            Xp = Xv[te].copy()
            perm = rng.permutation(len(te))
            Xp[:, cols] = Xp[perm][:, cols]  # se permuta el cluster entero, conjuntamente
            dl.append(np.average((y[te] - m.predict(Xp)) ** 2, weights=w[te]) - base)
        out[g] = float(np.mean(dl))
    return out


def clustered_mda(X: pd.DataFrame, y, w, fabrica, n_perm: int, semilla: int, n_jobs: int = 4) -> pd.DataFrame:
    """Aumento de MSE OOS al permutar cada columna-cluster en el bloque de prueba, por partición CPCV."""
    Xv = X.to_numpy()
    grupos_idx = {c: [i] for i, c in enumerate(X.columns)}
    parts = cpcv(len(X))
    res = Parallel(n_jobs=n_jobs)(
        delayed(_mda_particion)(fabrica, Xv, y, w, tr, te, grupos_idx, n_perm, semilla + i)
        for i, (_, tr, te) in enumerate(parts)
    )
    return pd.DataFrame(res, index=[str(c) for c, _, _ in parts])
