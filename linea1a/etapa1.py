"""Folds, sombras circulares, clusters y etapa 1 (cribado univariante)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from . import config as C
from .estadistica import monotonia_quintiles, nw_lags, pendiente_stambaugh, pendiente_t, spearman_enmascarado

DUMMIES = {"F13", "F14"}


@dataclass
class Periodo:
    """Datos de un periodo de análisis (por defecto, descubrimiento) en arrays alineados."""

    fechas: pd.DatetimeIndex
    X: np.ndarray  # (T, p) candidatas
    y: np.ndarray  # (T,)
    ym: np.ndarray  # (T,) máscara de y válida
    reg: pd.DataFrame  # R_vol, R_trend
    fin_etiqueta: pd.Series
    nombres: list[str]
    sigma5: np.ndarray | None = None  # volatilidad prevista a 5 días en t (para el objetivo normalizado)

    @classmethod
    def desde_panel(cls, panel: pd.DataFrame, ini=C.DESC_INI, fin=C.DESC_FIN) -> "Periodo":
        p = panel.loc[ini:fin]
        p = p[p["fin_etiqueta"] <= fin]  # purga en la frontera con el periodo siguiente
        y = p[C.OBJETIVO].to_numpy(float)
        return cls(
            fechas=p.index,
            X=p[C.CANDIDATAS].to_numpy(float),
            y=y,
            ym=np.isfinite(y),
            reg=p[["R_vol", "R_trend"]],
            fin_etiqueta=p["fin_etiqueta"],
            nombres=list(C.CANDIDATAS),
            sigma5=p["sigma5"].to_numpy(float),
        )

    def con_y(self, y: np.ndarray, ym: np.ndarray) -> "Periodo":
        return Periodo(self.fechas, self.X, y, ym, self.reg, self.fin_etiqueta, self.nombres, self.sigma5)


def folds(per: Periodo) -> list[tuple[np.ndarray, np.ndarray]]:
    """Walk-forward expansivo: 5 bloques de test de igual duración en la ventana de test, con purga."""
    bordes = pd.date_range(C.TEST_INI, C.TEST_FIN + pd.Timedelta(days=1), periods=C.N_FOLDS + 1)
    out = []
    for a, b in zip(bordes[:-1], bordes[1:]):
        test = np.where((per.fechas >= a) & (per.fechas < b))[0]
        t0 = per.fechas[test[0]]
        # Purga: fuera del entrenamiento toda etiqueta que termine después de la primera decisión de test.
        # (El embargo posterior al test no aplica: el walk-forward expansivo nunca entrena tras el test.)
        train = np.where((per.fechas < t0) & (per.fin_etiqueta.to_numpy() <= t0))[0]
        out.append((train, test))
    return out


def offsets(T: int, R: int, semilla: int) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    return rng.integers(C.K_MIN, T - C.K_MIN + 1, size=R)


def indices_sombra(T: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Índices fuente de la sombra con desplazamiento k y máscara que anula la unión (±1 semana)."""
    idx = (np.arange(T) + k) % T
    ok = np.ones(T, bool)
    ok[[T - k - 1, (T - k) % T]] = False
    return idx, ok


def mascara_corte(per: Periodo, corte: str) -> np.ndarray:
    if corte == "global":
        return np.ones(len(per.fechas), bool)
    var, val = corte.split("=")
    return (per.reg[var] == float(val)).to_numpy()


# ---------------------------------------------------------------- clusters (2.2)
def clusters(X: np.ndarray, nombres: list[str]) -> tuple[dict[str, int], pd.DataFrame]:
    rho = pd.DataFrame(X, columns=nombres).corr(method="spearman")
    dist = 1 - rho.abs().to_numpy()
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")
    etiquetas = fcluster(Z, t=0.4, criterion="distance")
    return dict(zip(nombres, map(int, etiquetas))), rho


def representantes(asignacion: dict[str, int], entran: list[str]) -> dict[str, str]:
    """Un representante por cluster entre las que entran, según la prioridad pre-registrada."""
    rep = {}
    for c in set(asignacion[v] for v in entran):
        miembros = [v for v in entran if asignacion[v] == c]
        elegido = min(miembros, key=C.PRIORIDAD.index)
        for v in miembros:
            rep[v] = elegido
    return rep


# ---------------------------------------------------------------- etapa 1
def _stats_celdas(per: Periodo, fl, X_src: np.ndarray, idx: np.ndarray, okm: np.ndarray, signos: np.ndarray):
    """U1/U2 por corte para un lote de réplicas. idx/okm: (R, T). Devuelve (R, cortes, p) x2 y signos de b."""
    R = idx.shape[0]
    p = X_src.shape[1]
    U1 = np.full((R, len(C.CORTES), p), np.nan)
    U2 = np.full((R, len(C.CORTES), p), np.nan)
    signo_ok = np.full((R, len(C.CORTES), p, len(fl)), np.nan)
    for ci, corte in enumerate(C.CORTES):
        mc = mascara_corte(per, corte)
        t_f, r_f, todas = [], [], []
        for fi, (_, test) in enumerate(fl):
            filas = test[mc[test] & per.ym[test]]
            n = len(filas)
            if n < C.MIN_CELDA:
                t_f.append(np.full((R, p), np.nan))
                r_f.append(np.full((R, p), np.nan))
                continue
            todas.append(filas)
            src = idx[:, filas]  # (R, n)
            x = np.transpose(X_src[src], (0, 2, 1))  # (R, p, n)
            m = np.isfinite(x) & okm[:, None, filas]
            y = per.y[filas]
            b, t, _ = pendiente_t(x, y, m, nw_lags(n))
            rho = monotonia_quintiles(x, y, m)
            t_f.append(t)
            r_f.append(rho)
            signo_ok[:, ci, :, fi] = np.where(signos == 0, np.nan, np.sign(b) == signos)
        t_f = np.stack(t_f, -1)  # (R, p, F)
        r_f = np.stack(r_f, -1)
        con = signos != 0
        st = np.where(con[None, :, None], signos[None, :, None] * t_f, np.abs(t_f))
        sr = np.where(con[None, :, None], signos[None, :, None] * r_f, np.abs(r_f))
        with np.errstate(all="ignore"):
            U1[:, ci, :] = C.agregar_folds(st, -1)
            if C.MODO_U2 == "por_fold":
                U2[:, ci, :] = C.agregar_folds(sr, -1)
        if C.MODO_U2 == "agregado" and todas:
            # Monotonía con todas las semanas de test evaluadas juntas (~400 en el corte global): correlación
            # de rangos x-y. Las medias por quintil agregadas saturan (el máximo del ruido llega a 1).
            filas = np.concatenate(todas)
            x = np.transpose(X_src[idx[:, filas]], (0, 2, 1))
            m = np.isfinite(x) & okm[:, None, filas]
            rho = spearman_enmascarado(x, per.y[filas], m)
            U2[:, ci, :] = np.where(con[None, :], signos[None, :] * rho, np.abs(rho))
    return U1, U2, signo_ok


def etapa1(per: Periodo, R: int = C.R1, semilla: int = C.SEMILLA, lote: int = 100) -> dict:
    T, p = per.X.shape
    fl = folds(per)
    signos = np.array([C.SIGNOS[v] for v in per.nombres])
    es_dummy = np.array([v in DUMMIES for v in per.nombres])

    idx0 = np.arange(T)[None, :]
    U1, U2, sok = _stats_celdas(per, fl, per.X, idx0, np.ones((1, T), bool), signos)
    U1, U2, sok = U1[0], U2[0], sok[0]
    U2[:, es_dummy] = np.nan  # forma no definida para binarias: U2 no aplica

    ks = offsets(T, R, semilla)
    M1, M2 = np.empty(R), np.empty(R)
    U1s_all = np.empty((R, len(C.CORTES), p))
    for a in range(0, R, lote):
        kk = ks[a : a + lote]
        pares = [indices_sombra(T, int(k)) for k in kk]
        idx = np.stack([q[0] for q in pares])
        okm = np.stack([q[1] for q in pares])
        u1, u2, _ = _stats_celdas(per, fl, per.X, idx, okm, signos)
        u2[:, :, es_dummy] = np.nan
        U1s_all[a : a + lote] = u1
        with np.errstate(all="ignore"):
            M1[a : a + lote] = np.nanmax(u1.reshape(len(kk), -1), -1)
            M2[a : a + lote] = np.nanmax(u2.reshape(len(kk), -1), -1)
    q1 = float(np.nanpercentile(M1, 95))
    q2 = float(np.nanpercentile(M2, 95)) if np.isfinite(M2).any() else np.nan

    filas = []
    for ci, corte in enumerate(C.CORTES):
        for j, v in enumerate(per.nombres):
            u1, u2 = U1[ci, j], U2[ci, j]
            forma_ok = es_dummy[j] or C.MODO_U2 == "ninguno" or (np.isfinite(u2) and u2 > q2)
            pasa = bool(np.isfinite(u1) and u1 > q1 and forma_ok)
            with np.errstate(all="ignore"):
                p_perm = float((np.sum(U1s_all[:, ci, j] >= u1) + 1) / (np.sum(np.isfinite(U1s_all[:, ci, j])) + 1))
            filas.append(
                dict(
                    variable=v,
                    corte=corte,
                    signo=int(signos[j]),
                    U1=u1,
                    U2=u2,
                    U3=float(np.nanmean(sok[ci, j])) if signos[j] != 0 else np.nan,
                    folds_evaluados=int(np.isfinite(sok[ci, j]).sum()) if signos[j] != 0 else np.nan,
                    p_perm_U1=p_perm,
                    q1=q1,
                    q2=q2,
                    pasa=pasa,
                )
            )
    tabla = pd.DataFrame(filas)

    # U4 analítico global (todo el periodo, NW + Stambaugh) para contraste con la permutación.
    u4 = []
    for j, v in enumerate(per.nombres):
        y = np.where(per.ym, per.y, np.nan)
        r = pendiente_stambaugh(per.X[:, j], y, nw_lags(int(per.ym.sum())))
        u4.append(dict(variable=v, **{f"U4_{k}": val for k, val in r.items()}))
    tabla = tabla.merge(pd.DataFrame(u4), on="variable")

    return dict(tabla=tabla, q1=q1, q2=q2, M1=M1, M2=M2, folds=fl, salida=salida_etapa1(tabla, per.nombres))


def salida_etapa1(tabla: pd.DataFrame, nombres: list[str]) -> pd.DataFrame:
    """Tabla 5.5: global / condicional (estado donde pasa) / excepción / descartada."""
    filas = []
    for v in nombres:
        t = tabla[tabla.variable == v]
        g = bool(t[t.corte == "global"].pasa.iloc[0])
        estados = t[(t.corte != "global") & t.pasa].corte.tolist()
        if g:
            clase = "pasa_global"
        elif estados:
            clase = "condicional"
        elif v in C.EXCEPCIONES:
            clase = "excepcion"
        else:
            clase = "descartada_e1"
        filas.append(dict(variable=v, clase_e1=clase, estados=";".join(estados)))
    return pd.DataFrame(filas)


def t_por_fold(per: Periodo) -> pd.DataFrame:
    """t Newey-West con signo pre-registrado (|t| sin signo previo) por variable y fold, corte global."""
    filas = []
    for f, (_, test) in enumerate(folds(per)):
        filas_ok = test[per.ym[test]]
        x = per.X[filas_ok].T
        b, t, n = pendiente_t(x, per.y[filas_ok], np.isfinite(x), nw_lags(len(filas_ok)))
        for j, v in enumerate(per.nombres):
            s = C.SIGNOS[v]
            filas.append(dict(variable=v, fold=f + 1, b=b[j], t=t[j], t_con_signo=s * t[j] if s else abs(t[j]), n=int(n[j])))
    return pd.DataFrame(filas)
