"""Regla primaria (8.2) y barreras (8.3): eventos de reversión, salida por barreras y meta-label."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def fechas_decision(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Último día hábil de cada semana ISO."""
    iso = index.isocalendar()
    clave = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    ultimo = index.to_series().groupby(clave.to_numpy()).transform("max")
    return index[index == ultimo.to_numpy()]


def eventos(d: pd.DataFrame, sigma5: pd.Series, ini, fin) -> pd.DataFrame:
    """Eventos de la regla en [ini, fin] con salida por barreras; solo los que cierran antes de `fin`."""
    lp = np.log(d["close"])
    r5 = lp - lp.shift(C.H)
    pos = pd.Series(np.arange(len(d)), index=d.index)
    filas = []
    for t in fechas_decision(d.index):
        if not (ini <= t <= fin):
            continue
        s = sigma5.get(t, np.nan)
        if not np.isfinite(s) or not r5.loc[t] < -C.UMBRAL_REGLA * s:
            continue
        i = pos[t]
        if i + C.H >= len(d) or d.index[i + C.H] > fin:
            continue
        p0 = d["close"].iloc[i]
        tp, sl = p0 * np.exp(C.BARRERA * s), p0 * np.exp(-C.BARRERA * s)
        salida, motivo, k_sal = d["close"].iloc[i + C.H], "vertical", C.H
        for k in range(1, C.H + 1):
            o, h, lo = d["open"].iloc[i + k], d["high"].iloc[i + k], d["low"].iloc[i + k]
            if lo <= sl:  # stop primero si el mismo día toca las dos (supuesto conservador)
                salida, motivo, k_sal = (o if o < sl else sl), "stop", k
                break
            if h >= tp:
                salida, motivo, k_sal = (o if o > tp else tp), "toma_beneficio", k
                break
        ret = np.log(salida / p0) - C.COSTE_IDA_VUELTA
        filas.append(
            dict(
                fecha=t,
                r5=r5.loc[t],
                sigma5=s,
                entrada=p0,
                salida=salida,
                motivo=motivo,
                sesiones=k_sal,
                fin=d.index[i + k_sal],
                ret_neto=ret,
                meta_label=int(motivo == "toma_beneficio" or (motivo == "vertical" and ret > 0)),
            )
        )
    return pd.DataFrame(filas).set_index("fecha")
