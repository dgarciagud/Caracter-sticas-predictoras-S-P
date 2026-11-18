"""Particiones: CPCV con purga y embargo (descubrimiento) y walk-forward expansivo (evaluación).

Todas las posiciones son de sesión dentro de la muestra válida, que es un tramo contiguo del
calendario del índice: la etiqueta de la fila p cubre los retornos de las sesiones p+1..p+h.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import config as C


def bloques(n: int, nb: int = C.N_BLOQUES) -> list[np.ndarray]:
    return np.array_split(np.arange(n), nb)


def cpcv(n: int, nb: int = C.N_BLOQUES, k: int = C.K_TEST, purga: int = C.PURGA, embargo: int = C.EMBARGO):
    """Lista de (combinación, idx_train, idx_test). Purga: fuera las filas cuyas etiquetas tocan un bloque
    de prueba (p en [ini - purga, ini - 1]); embargo: fuera las `embargo` filas posteriores a cada bloque."""
    B = bloques(n, nb)
    out = []
    for comb in itertools.combinations(range(nb), k):
        test = np.concatenate([B[c] for c in comb])
        fuera = np.zeros(n, bool)
        fuera[test] = True
        for c in comb:
            ini, fin = B[c][0], B[c][-1]
            fuera[max(0, ini - purga) : ini] = True
            fuera[fin + 1 : min(n, fin + 1 + embargo)] = True
        out.append((comb, np.where(~fuera)[0], test))
    return out


def trayectorias(nb: int = C.N_BLOQUES, k: int = C.K_TEST) -> list[dict[int, int]]:
    """Trayectorias CPCV: para la trayectoria j, bloque -> índice de la partición que lo predice."""
    combs = list(itertools.combinations(range(nb), k))
    usos = {b: [i for i, c in enumerate(combs) if b in c] for b in range(nb)}
    n_tray = len(usos[0])
    return [{b: usos[b][j] for b in range(nb)} for j in range(n_tray)]


def walk_forward(fechas: pd.DatetimeIndex, inicio=C.WF_TEST_INI, fin=None, purga: int = C.PURGA):
    """Lista de (año, idx_train, idx_test): ventana expansiva, reentreno anual, purga antes del test."""
    fin = fechas[-1] if fin is None else pd.Timestamp(fin)
    out = []
    for anio in range(pd.Timestamp(inicio).year, fin.year + 1):
        te = np.where((fechas.year == anio) & (fechas <= fin))[0]
        if not len(te):
            continue
        tr = np.arange(0, max(0, te[0] - purga))
        if len(tr) >= C.WF_MIN_TRAIN:
            out.append((anio, tr, te))
    return out
