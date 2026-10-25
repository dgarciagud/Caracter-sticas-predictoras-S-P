"""Etapa 2: ridge, elastic net y LightGBM con sombras conjuntas; estadísticos L1, G1-G3, I1."""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from . import config as C
from .etapa1 import Periodo, folds, indices_sombra, offsets

CLAVES = ["L1_ridge", "L1_enet", "G1", "G2", "G3", "I1"]
REJILLA = np.array([-np.inf, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, np.inf])
ALPHAS_RIDGE = np.logspace(-1, 5, 31)


def _std(a: np.ndarray, tr: np.ndarray) -> np.ndarray:
    mu = np.nanmean(a[tr], 0)
    sd = np.nanstd(a[tr], 0)
    sd[sd == 0] = 1
    return (a - mu) / sd


def _pendiente(x, s):
    ok = np.isfinite(x) & np.isfinite(s)
    if ok.sum() < 10 or np.std(x[ok]) == 0:
        return np.nan
    return float(np.polyfit(x[ok], s[ok], 1)[0])


def _spear(x, s):
    ok = np.isfinite(x) & np.isfinite(s)
    if ok.sum() < 10 or np.std(x[ok]) == 0 or np.std(s[ok]) == 0:
        return np.nan
    return float(stats.spearmanr(x[ok], s[ok]).statistic)


def _curva(x, s):
    b = np.digitize(x, REJILLA[1:-1])
    return np.array([np.nanmean(s[b == k]) if np.any(b == k) else np.nan for k in range(len(REJILLA) - 1)])


class Diseno:
    """Variables que entran en la etapa 2 y cómo entran en cada modelo."""

    def __init__(self, variables: list[str], condicionales: dict[str, str]):
        self.variables = variables
        self.condicionales = condicionales  # variable -> estado "R_vol=1" etc.

    def lineal(self, Xs: np.ndarray, per: Periodo, filas: np.ndarray) -> np.ndarray:
        cols = []
        for j, v in enumerate(self.variables):
            x = Xs[filas, j]
            if v in self.condicionales:
                var, val = self.condicionales[v].split("=")
                x = x * (per.reg[var].to_numpy()[filas] == float(val))
            cols.append(x)
        return np.column_stack(cols)


def _ajustar_fold(per, dis, Xr, Xsh, tr, te, hp_lin, con_curvas):
    """Ajusta los tres modelos en un fold. Xsh puede ser None (corrida sin sombras)."""
    k = len(dis.variables)
    Xall = Xr if Xsh is None else np.column_stack([Xr, Xsh])
    Xz = _std(Xall, tr)
    y = per.y
    tr = tr[np.isfinite(Xz[tr]).all(1) & per.ym[tr]]
    te_ok = te[np.isfinite(Xz[te]).all(1)]
    out = {}

    def lin_design(filas):
        a = dis.lineal(Xz[:, :k], per, filas)
        if Xsh is not None:
            a = np.column_stack([a, dis.lineal(Xz[:, k:], per, filas)])
        return a

    Ltr = lin_design(tr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if hp_lin is None:  # solo con variables reales: se fija y se congela
            rcv = RidgeCV(alphas=ALPHAS_RIDGE, cv=TimeSeriesSplit(5)).fit(Ltr, y[tr])
            ecv = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], alphas=50, cv=TimeSeriesSplit(5), max_iter=20000).fit(
                Ltr, y[tr]
            )
            hp_lin = dict(ridge_alpha=float(rcv.alpha_), enet_alpha=float(ecv.alpha_), enet_l1=float(ecv.l1_ratio_))
        ridge = Ridge(alpha=hp_lin["ridge_alpha"]).fit(Ltr, y[tr])
        enet = ElasticNet(alpha=hp_lin["enet_alpha"], l1_ratio=hp_lin["enet_l1"], max_iter=20000).fit(Ltr, y[tr])
    out["coef_ridge"] = ridge.coef_
    out["coef_enet"] = enet.coef_
    out["hp_lin"] = hp_lin

    reg = per.reg.to_numpy(float)
    G = np.column_stack([Xz, reg])
    gbm = lgb.LGBMRegressor(**C.LGBM_PARAMS, random_state=C.SEMILLA).fit(G[tr], y[tr])
    sv = shap.TreeExplainer(gbm).shap_values(G[te_ok])
    xte = Xz[te_ok]
    ncol = Xz.shape[1]
    out["G1"] = np.array([_pendiente(xte[:, c], sv[:, c]) for c in range(ncol)])
    out["G2"] = np.array([_spear(xte[:, c], sv[:, c]) for c in range(ncol)])
    out["G3"] = np.abs(sv[:, :ncol]).mean(0)
    i1 = np.full(ncol, np.nan)
    for c in range(ncol):
        v = dis.variables[c % k]
        if v in dis.condicionales:
            var, val = dis.condicionales[v].split("=")
            st = per.reg[var].to_numpy()[te_ok] == float(val)
            if st.sum() >= 10 and (~st).sum() >= 10:
                i1[c] = _pendiente(xte[st, c], sv[st, c]) - _pendiente(xte[~st, c], sv[~st, c])
    out["I1"] = i1
    if con_curvas:
        out["curvas"] = np.array([_curva(xte[:, c], sv[:, c]) for c in range(ncol)])
        out["shap_test"] = (per.fechas[te_ok], xte[:, :ncol], sv[:, :ncol])
    if Xsh is None:
        out["pred_test"] = (te_ok, gbm.predict(G[te_ok]), ridge.predict(lin_design(te_ok)))
    return out


def _agregar(res_folds, k, signos):
    """Estadísticos 6.4 sobre los folds de una réplica; columnas 0..k-1 reales, k..2k-1 sombras."""
    s = np.r_[signos, signos].astype(float)
    con = s != 0

    def apilar(nombre, con_signo=True):
        a = np.stack([r[nombre] for r in res_folds])  # (F, 2k)
        return np.where(con, s * a, np.abs(a)) if con_signo else a

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return dict(
            L1_ridge=C.agregar_folds(apilar("coef_ridge"), 0),
            L1_enet=C.agregar_folds(apilar("coef_enet"), 0),
            G1=C.agregar_folds(apilar("G1"), 0),
            G2=C.agregar_folds(apilar("G2"), 0),
            G3=np.nanmedian(apilar("G3", con_signo=False), 0),
            I1=C.agregar_folds(apilar("I1"), 0),
            signo_ridge=np.sign(np.nanmedian(apilar("coef_ridge", con_signo=False), 0)),
            signo_gbm=np.sign(np.nanmedian(apilar("G1", con_signo=False), 0)),
        )


def _replica(per, dis, Xr, fl, hp, k_off, j_sel):
    T = len(per.fechas)
    idx, ok = indices_sombra(T, int(k_off))
    Xsh = per.X[idx][:, j_sel].copy()
    Xsh[~ok] = np.nan
    res = [_ajustar_fold(per, dis, Xr, Xsh, tr, te, hp[f], con_curvas=True) for f, (tr, te) in enumerate(fl)]
    agg = _agregar(res, len(dis.variables), np.array([C.SIGNOS[v] for v in dis.variables]))
    agg["curvas_sombra"] = np.stack([r["curvas"][len(dis.variables) :] for r in res])  # (F, k, bins)
    return agg


def etapa2(per: Periodo, dis: Diseno, R: int = C.R2, semilla: int = C.SEMILLA + 2, n_jobs: int = 4) -> dict:
    k = len(dis.variables)
    j_sel = [per.nombres.index(v) for v in dis.variables]
    Xr = per.X[:, j_sel]
    fl = folds(per)
    signos = np.array([C.SIGNOS[v] for v in dis.variables])

    # Corrida sin sombras: fija hiperparámetros lineales por fold y da curvas/predicciones.
    base = [_ajustar_fold(per, dis, Xr, None, tr, te, None, con_curvas=True) for tr, te in fl]
    hp = [b["hp_lin"] for b in base]

    ks = offsets(len(per.fechas), R, semilla)
    reps = Parallel(n_jobs=n_jobs)(delayed(_replica)(per, dis, Xr, fl, hp, kk, j_sel) for kk in ks)

    reales = {c: np.nanmedian(np.stack([r[c][:k] for r in reps]), 0) for c in CLAVES}
    maximos = {}
    for c in CLAVES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            maximos[c] = np.array([np.nanmax(r[c][k:]) if np.isfinite(r[c][k:]).any() else np.nan for r in reps])
    signo_ridge = np.sign(np.nanmedian(np.stack([r["signo_ridge"][:k] for r in reps]), 0))
    signo_gbm = np.sign(np.nanmedian(np.stack([r["signo_gbm"][:k] for r in reps]), 0))
    base_tabla = dict(variables=dis.variables, signos=signos, reales=reales, maximos=maximos,
                      signo_ridge=signo_ridge, signo_gbm=signo_gbm)
    tabla = tabla_etapa2(base_tabla, C.PERCENTIL)
    curvas_sombra = np.stack([r["curvas_sombra"] for r in reps])  # (R, F, k, bins)
    return dict(tabla=tabla, base=base, hp=hp, curvas_sombra=curvas_sombra, folds=fl, base_tabla=base_tabla)


def tabla_etapa2(b: dict, percentil: float) -> pd.DataFrame:
    """Tabla 6.4 con umbrales = percentil dado del máximo de las sombras (permite recalibrar sin reentrenar)."""
    reales, signo_ridge, signo_gbm = b["reales"], b["signo_ridge"], b["signo_gbm"]
    umbrales = {
        c: float(np.nanpercentile(m, percentil)) if np.isfinite(m).any() else np.nan for c, m in b["maximos"].items()
    }
    tabla = pd.DataFrame({"variable": b["variables"], "signo_prereg": b["signos"]})
    for c in CLAVES:
        tabla[c] = reales[c]
        tabla[f"umbral_{c}"] = umbrales[c]
    tabla["cumple_lineal"] = (tabla.L1_ridge > umbrales["L1_ridge"]) & (tabla.L1_enet > umbrales["L1_enet"])
    tabla["cumple_gbm"] = np.logical_and.reduce([tabla[g] > umbrales[g] for g in C.CRITERIO_GBM])
    tabla["cumple_I1"] = tabla.I1 > umbrales["I1"] if np.isfinite(umbrales["I1"]) else False
    # Forma dentro del régimen para las condicionales: G2 si forma parte del criterio GBM, si no G1.
    g_forma = "G2" if "G2" in C.CRITERIO_GBM else "G1"
    tabla["cumple_G2"] = tabla[g_forma] > umbrales[g_forma]
    tabla["signo_ridge"] = signo_ridge
    tabla["signo_gbm"] = signo_gbm
    return tabla


def clasificar(salida_e1: pd.DataFrame, tabla_e2: pd.DataFrame | None, rep: dict[str, str], X: pd.DataFrame) -> pd.DataFrame:
    """Clasificación final 6.5."""
    filas = []
    for _, r in salida_e1.iterrows():
        v, c1 = r.variable, r.clase_e1
        fila = dict(variable=v, clase_e1=c1, estados=r.estados, representada_por=rep.get(v, ""))
        if c1 == "descartada_e1":
            fila.update(clase="descartada", motivo="no pasa la etapa 1 (U1/U2 frente a q1/q2)")
        elif rep.get(v, v) != v:
            fila.update(clase="representada", motivo=f"en el cluster de {rep[v]}")
        elif tabla_e2 is None or v not in set(tabla_e2.variable):
            fila.update(clase="descartada", motivo="sin etapa 2")
        else:
            t = tabla_e2.set_index("variable").loc[v]
            opuestos = t.signo_ridge != 0 and t.signo_gbm != 0 and t.signo_ridge != t.signo_gbm
            sin_signo = C.SIGNOS[v] == 0
            if opuestos and t.cumple_lineal and t.cumple_gbm:
                clase, motivo = "descartada", "lineal y GBM con signos opuestos"
            elif c1 == "pasa_global" and t.cumple_lineal and t.cumple_gbm:
                clase, motivo = ("control" if sin_signo else "estable_incondicional"), "cumple L1 y G1-G3"
            elif c1 in ("pasa_global", "condicional") and t.cumple_I1 and t.cumple_G2:
                clase, motivo = ("control" if sin_signo else "estable_condicional"), f"cumple I1 y G2 en {r.estados}"
            elif c1 == "excepcion" and t.cumple_lineal and t.cumple_gbm:
                clase, motivo = "supresion_sospechosa", "excepción que cumple en etapa 2"
            elif c1 in ("pasa_global", "condicional"):
                otras = [o for o in tabla_e2.variable if o != v]
                pc = X[otras].corrwith(X[v]).abs().idxmax() if otras else ""
                clase, motivo = "redundante", f"no cumple en etapa 2; comparte efecto con {pc}"
            else:
                fallos = [
                    f"{s}={t[s]:.3g}<= {t['umbral_' + s]:.3g}"
                    for s in ["L1_ridge", "L1_enet", *C.CRITERIO_GBM]
                    if not (t[s] > t["umbral_" + s])
                ]
                clase, motivo = "descartada", "etapa 2: " + "; ".join(fallos)
            fila.update(clase=clase, motivo=motivo)
        filas.append(fila)
    return pd.DataFrame(filas)
