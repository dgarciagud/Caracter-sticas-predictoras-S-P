"""Meta-label de la 1B (8.4-8.7): datos de eventos, sombras en el calendario semanal, etapas 1 y 2 y clasificación."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from . import config as C

ANEXO = C.PREREG.get("anexo_meta_label", {})
FEATURES: list[str] = list(ANEXO.get("meta_features_efectivas", ["F05", "R_vol"]))
SIGNOS: dict[str, int] = dict(ANEXO.get("signos", {"F05": 1, "R_vol": 0}))
K_MIN = 52
C_REJILLA = np.logspace(-3, 2, 11)


@dataclass
class Eventos:
    """Eventos de descubrimiento alineados con sus features y con el calendario semanal (para las sombras)."""

    ev: pd.DataFrame  # índice = fecha de decisión; columnas fin, ret_neto, meta_label, sigma5
    X: np.ndarray  # (n, p) features reales en las fechas de los eventos
    y: np.ndarray  # meta-label
    w: np.ndarray  # pesos 8.4
    cal: pd.DataFrame  # calendario semanal de descubrimiento con las features (para desplazar)
    pos: np.ndarray  # posición de cada evento en el calendario

    @classmethod
    def construir(cls, ev: pd.DataFrame, panel: pd.DataFrame) -> "Eventos":
        cal = panel.loc[C.DESC_INI : C.DESC_FIN, FEATURES]
        pos = cal.index.get_indexer(ev.index)
        assert (pos >= 0).all(), "hay eventos fuera del calendario semanal"
        return cls(ev=ev, X=cal.to_numpy(float)[pos], y=ev.meta_label.to_numpy(int), w=pesos(ev),
                   cal=cal, pos=pos)

    def con_y(self, y: np.ndarray) -> "Eventos":
        return Eventos(self.ev, self.X, y, self.w, self.cal, self.pos)


def pesos(ev: pd.DataFrame) -> np.ndarray:
    """Unicidad (media de 1/concurrencia en la vida del evento) × |retorno|, normalizados a sumar n."""
    ini, fin = ev.index.to_numpy(), pd.to_datetime(ev["fin"]).to_numpy()
    uni = np.empty(len(ev))
    for i in range(len(ev)):
        solapan = (ini < fin[i]) & (fin > ini[i])
        uni[i] = 1 / max(solapan.sum(), 1)
    w = uni * np.abs(ev["ret_neto"].to_numpy())
    return w * len(w) / w.sum()


def folds(ev: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """5 bloques de test de igual duración en la ventana de test; entrenamiento expansivo con purga (y embargo)."""
    bordes = pd.date_range(C.TEST_INI, C.TEST_FIN + pd.Timedelta(days=1), periods=6)
    fechas, fin = ev.index, pd.to_datetime(ev["fin"])
    out = []
    for a, b in zip(bordes[:-1], bordes[1:]):
        te = np.where((fechas >= a) & (fechas < b))[0]
        t0 = fechas[te[0]]
        tr = np.where((fechas < t0) & (fin.to_numpy() < t0.to_datetime64()))[0]
        out.append((tr, te))
    return out


def sombra(E: Eventos, k: int) -> np.ndarray:
    """Features desplazadas k semanas en el calendario completo y leídas en las fechas de los eventos."""
    T = len(E.cal)
    idx = (np.arange(T) + k) % T
    Xs = E.cal.to_numpy(float)[idx]
    Xs[[T - k - 1, (T - k) % T]] = np.nan  # unión del desplazamiento
    return Xs[E.pos]


def offsets(T: int, R: int, semilla: int) -> np.ndarray:
    return np.random.default_rng(semilla).integers(K_MIN, T - K_MIN + 1, size=R)


# ------------------------------------------------------------------ etapa 1 (8.6)
def _auc(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0 or np.ptp(x) == 0:
        return np.nan
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def estadistico_e1(X: np.ndarray, y: np.ndarray, fl) -> np.ndarray:
    """U por feature: media de folds de signo*(AUC - 0,5), o |AUC - 0,5| sin signo previo."""
    out = []
    for j, f in enumerate(FEATURES):
        s = SIGNOS[f]
        v = [_auc(X[te, j], y[te]) - 0.5 for _, te in fl]
        v = [s * a if s else abs(a) for a in v]
        out.append(np.nanmean(v))
    return np.array(out)


def etapa1(E: Eventos, R: int = 1000, semilla: int = 0) -> dict:
    fl = folds(E.ev)
    U = estadistico_e1(E.X, E.y, fl)
    M = np.array([np.nanmax(estadistico_e1(sombra(E, int(k)), E.y, fl)) for k in offsets(len(E.cal), R, semilla)])
    terciles = {}
    for j, f in enumerate(FEATURES):
        x = E.X[:, j]
        lo, hi = np.nanpercentile(x, [33.3, 66.7])
        terciles[f] = float(E.y[x >= hi].mean() - E.y[x <= lo].mean()) if hi > lo else np.nan
    return dict(U=U, M=M, folds=fl, terciles=terciles)


def pasa_e1(e1: dict, p: float) -> list[str]:
    q = np.nanpercentile(e1["M"], p)
    return [f for f, u in zip(FEATURES, e1["U"]) if u > q]


# ------------------------------------------------------------------ etapa 2 (8.7)
def _std(A, tr):
    mu, sd = np.nanmean(A[tr], 0), np.nanstd(A[tr], 0)
    sd[sd == 0] = 1
    return (A - mu) / sd


def _c_logistica(Z, y, w):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        g = GridSearchCV(LogisticRegression(max_iter=2000), {"C": C_REJILLA}, cv=TimeSeriesSplit(3), scoring="neg_brier_score")
        g.fit(Z, y, sample_weight=w)
    return float(g.best_params_["C"])


def _ebm():
    from interpret.glassbox import ExplainableBoostingClassifier

    return ExplainableBoostingClassifier(interactions=0, min_samples_leaf=20, outer_bags=4, random_state=C.PREREG.get("semilla", 20261027), n_jobs=1)


def _ajuste(E, cols, Xall, fl, c_por_fold, con_ebm):
    """Coeficientes de la logística y G1/G3 del EBM por fold (columnas = reales + sombras)."""
    sg = np.array([SIGNOS[c.split("|")[0]] for c in cols], float)
    L, G1, G3 = [], [], []
    for f, (tr, te) in enumerate(fl):
        Z = _std(Xall, tr)
        ok_tr = tr[np.isfinite(Z[tr]).all(1)]
        ok_te = te[np.isfinite(Z[te]).all(1)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            lg = LogisticRegression(C=c_por_fold[f], max_iter=2000).fit(Z[ok_tr], E.y[ok_tr], sample_weight=E.w[ok_tr])
        L.append(lg.coef_[0])
        if con_ebm:
            m = _ebm().fit(Z[ok_tr], E.y[ok_tr], sample_weight=E.w[ok_tr])
            contrib = m.eval_terms(Z[ok_te])  # (n, términos) en log-odds
            g1, g3 = [], []
            for j in range(Z.shape[1]):
                x, s = Z[ok_te, j], contrib[:, j]
                g1.append(np.polyfit(x, s, 1)[0] if np.ptp(x) > 0 else np.nan)
                g3.append(np.abs(s).mean())
            G1.append(g1)
            G3.append(g3)
    con = sg != 0
    fsig = lambda a: np.where(con, sg * a, np.abs(a))  # noqa: E731
    out = dict(L=np.nanmean([fsig(np.array(l)) for l in L], 0), signo_L=np.sign(np.nanmean(L, 0)))
    if con_ebm:
        out.update(G1=np.nanmean([fsig(np.array(g)) for g in G1], 0), G3=np.nanmean(G3, 0), signo_G1=np.sign(np.nanmean(G1, 0)))
    return out


def etapa2(E: Eventos, feats: list[str], R: int = 100, semilla: int = 0, con_ebm: bool = True) -> dict:
    fl = folds(E.ev)
    j = [FEATURES.index(f) for f in feats]
    Xr = E.X[:, j]
    c_por_fold = []
    for tr, _ in fl:
        Z = _std(Xr, tr)
        ok = tr[np.isfinite(Z[tr]).all(1)]
        c_por_fold.append(_c_logistica(Z[ok], E.y[ok], E.w[ok]))
    k = len(feats)
    reps = []
    for kk in offsets(len(E.cal), R, semilla):
        Xs = sombra(E, int(kk))[:, j]
        reps.append(_ajuste(E, feats + [f + "|sombra" for f in feats], np.column_stack([Xr, Xs]), fl, c_por_fold, con_ebm))
    claves = ["L"] + (["G1", "G3"] if con_ebm else [])
    reales = {c: np.nanmedian([r[c][:k] for r in reps], 0) for c in claves}
    maximos = {c: np.array([np.nanmax(r[c][k:]) for r in reps]) for c in claves}
    signos = {c: np.sign(np.nanmedian([r[c][:k] for r in reps], 0)) for c in ["signo_L"] + (["signo_G1"] if con_ebm else [])}
    return dict(feats=feats, reales=reales, maximos=maximos, signos=signos, c_por_fold=c_por_fold, con_ebm=con_ebm)


def clasificar(e2: dict | None, p: float) -> tuple[list[str], pd.DataFrame]:
    """Features que entran al meta-modelo: cumplen L (y G1, G3 con el EBM) y la logística y el EBM coinciden en signo."""
    if e2 is None:
        return [], pd.DataFrame()
    filas, dentro = [], []
    for i, f in enumerate(e2["feats"]):
        fila = dict(feature=f)
        ok = True
        for c in e2["reales"]:
            u = float(np.nanpercentile(e2["maximos"][c], p))
            fila[c], fila[f"umbral_{c}"] = float(e2["reales"][c][i]), u
            ok &= fila[c] > u
        if e2["con_ebm"]:
            ok &= e2["signos"]["signo_L"][i] == e2["signos"]["signo_G1"][i]
        fila["entra"] = bool(ok)
        filas.append(fila)
        if ok:
            dentro.append(f)
    return dentro, pd.DataFrame(filas)


def seleccionar(E: Eventos, p: float, R1: int = 1000, R2: int = 100, semilla: int = 0, con_ebm: bool = True,
                e1: dict | None = None, cache_e2: dict | None = None) -> dict:
    e1 = e1 or etapa1(E, R=R1, semilla=semilla)
    pasan = pasa_e1(e1, p)
    e2 = None
    if pasan:
        clave = tuple(pasan)
        if cache_e2 is not None and clave in cache_e2:
            e2 = cache_e2[clave]
        else:
            e2 = etapa2(E, pasan, R=R2, semilla=semilla + 1, con_ebm=con_ebm)
            if cache_e2 is not None:
                cache_e2[clave] = e2
    dentro, tabla = clasificar(e2, p)
    return dict(e1=e1, pasan_e1=pasan, e2=e2, tabla_e2=tabla, meta_features=dentro)


# ------------------------------------------------------------------ meta-modelo final y predicción
def meta_modelo(X_tr, y_tr, w_tr, tipo: str):
    """Logística calibrada (Platt) o EBM calibrado, entrenados sobre los eventos de entrenamiento."""
    if X_tr.shape[1] == 0:
        return None
    if tipo == "logistica":
        c = _c_logistica(X_tr, y_tr, w_tr)
        base = LogisticRegression(C=c, max_iter=2000)
    else:
        base = _ebm()
    m = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m
