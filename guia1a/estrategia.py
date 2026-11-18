"""Estrategia de timing diaria, referencias, métricas económicas, registro de intentos, PBO y DSR."""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from ml4t.diagnostic.evaluation.stats.backtest_overfitting import compute_pbo
from ml4t.diagnostic.evaluation.stats.deflated_sharpe_ratio import deflated_sharpe_ratio

from . import config as C
from .estad import dif_sharpe_hac, ols_hac, sharpe


def tramos(senal: pd.Series, h: int = C.H) -> pd.Series:
    """Peso efectivo: 5 tramos de 1/5 del capital, cada uno fijado en su día y mantenido h sesiones."""
    return senal.rolling(h, min_periods=1).mean()


def backtest(w: pd.Series, dia: pd.DataFrame, coste_pb: float = C.COSTE_PB) -> pd.DataFrame:
    """w fijado al cierre de t se aplica al retorno de t+1. dia: ret_d y rf_d diarios."""
    w = w.reindex(dia.index)
    wl = w.shift(1)
    rot = w.diff().abs().shift(1)
    bruto = wl * dia["ret_d"] + (1 - wl) * dia["rf_d"]
    neto = bruto - coste_pb / 1e4 * rot.fillna(0)
    out = pd.DataFrame({"w": wl, "bruto": bruto, "neto": neto, "rot": rot, "rf": dia["rf_d"]})
    return out.dropna(subset=["w"])


def referencias(dia: pd.DataFrame, mom12_1: pd.Series, sigma_d: pd.Series, media_hist: pd.Series) -> dict[str, pd.Series]:
    """Pesos de las referencias: buy & hold, media histórica, TSMOM 12-1 y B&H escalado por volatilidad."""
    uno = pd.Series(1.0, index=dia.index)
    ts = tramos((mom12_1 > 0).astype(float).where(mom12_1.notna()))
    obj = sigma_d.expanding(252).median()
    vs = (obj / sigma_d).clip(upper=2.0)
    mh = tramos((media_hist > 0).astype(float).where(media_hist.notna()))
    return {"buy_hold": uno, "media_historica": mh, "tsmom_12_1": ts, "bh_vol_escalado": vs}


def metricas(bt: pd.DataFrame, bt_bh: pd.DataFrame | None = None, bt_ts: pd.DataFrame | None = None) -> dict:
    ex = bt["neto"] - bt["rf"]
    curva = (1 + bt["neto"]).cumprod()
    m = dict(
        sharpe_neto=sharpe(ex),
        rent_anual=float(bt["neto"].mean() * 252),
        vol_anual=float(bt["neto"].std() * np.sqrt(252)),
        max_dd=float((curva / curva.cummax() - 1).min()),
        rotacion_anual=float(bt["rot"].mean() * 252),
        peso_medio=float(bt["w"].mean()),
    )
    if bt_bh is not None:
        idx = bt.index.intersection(bt_bh.index)
        exb = (bt_bh["neto"] - bt_bh["rf"]).loc[idx]
        d, t = dif_sharpe_hac(ex.loc[idx], exb)
        m.update(dif_sharpe_bh=d, t_dif_sharpe_bh=t)
        exceso_bruto = (bt["bruto"] - bt_bh["bruto"]).loc[idx].mean() * 252
        m["break_even_pb_vs_bh"] = float(exceso_bruto / max(m["rotacion_anual"], 1e-9) * 1e4)
        m["break_even_pb"] = float((bt["bruto"] - bt["rf"]).mean() * 252 / max(m["rotacion_anual"], 1e-9) * 1e4)
        if bt_ts is not None:
            ext = (bt_ts["neto"] - bt_ts["rf"]).loc[idx]
            b, t = ols_hac(ex.loc[idx].to_numpy(), np.column_stack([exb.to_numpy(), ext.to_numpy()]))
            m.update(alfa_anual=float(b[0] * 252), t_alfa=float(t[0]))
    return m


# ---------------------------------------------------------------- registro de intentos (A.2.4)
def registrar(id_: str, fase: str, descripcion: str, validacion: str, metr: dict, serie: pd.Series | None, **extra):
    ruta = None
    if serie is not None:
        ruta = C.SERIES_OOS / f"{id_}.parquet"
        serie.rename("neto_exceso").to_frame().to_parquet(ruta)
    fila = dict(
        id=id_,
        fecha=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        version=C.VERSION,
        hash_prereg=C.hash_prereg()[:16],
        fase=fase,
        descripcion=descripcion,
        validacion=validacion,
        serie=str(ruta.relative_to(C.RAIZ)) if ruta else "",
        metricas=json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in metr.items()}),
        **{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in extra.items()},
    )
    df = pd.DataFrame([fila])
    if C.REGISTRO.exists():
        prev = pd.read_csv(C.REGISTRO)
        prev = prev[prev["id"] != id_]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(C.REGISTRO, index=False)


def registro_conjunto() -> pd.DataFrame:
    """Registro de esta versión + el de las versiones previas (pre-registro: versiones_previas)."""
    regs = []
    for v in [*C.VERSIONES_PREVIAS, C.VERSION]:
        ruta = C.RAIZ / "resultados" / "guia1a" / v / "registro_intentos.csv"
        if ruta.exists():
            r = pd.read_csv(ruta)
            r["clave"] = v + ":" + r["id"]
            regs.append(r)
    return pd.concat(regs, ignore_index=True)


def matriz_registro(validacion: str | tuple = "walk_forward") -> tuple[pd.DataFrame, int]:
    """Matriz fechas x variantes (todas las versiones). Las series idénticas entre versiones (mismo intento
    repetido) se cuentan una sola vez. Devuelve la matriz y el número de intentos únicos con esa validación."""
    reg = registro_conjunto()
    vals = (validacion,) if isinstance(validacion, str) else validacion
    reg = reg[reg["validacion"].isin(vals) & reg["serie"].notna() & (reg["serie"] != "")]
    cols, vistos = {}, set()
    for r in reg.itertuples():
        s = pd.read_parquet(C.RAIZ / r.serie)["neto_exceso"]
        firma = hash(tuple(np.round(s.to_numpy(), 12)))
        if firma in vistos:
            continue
        vistos.add(firma)
        cols[r.clave] = s
    M = pd.DataFrame(cols).dropna()
    n = len(cols)
    # Variantes sin varianza (siempre en liquidez): cuentan como intento pero no tienen Sharpe.
    return M.loc[:, M.std() > 1e-12], n


def pbo_cscv(R: pd.DataFrame, S: int = 16) -> dict:
    T = len(R) // S * S
    B = np.array_split(np.arange(T), S)
    M = R.iloc[:T].to_numpy()
    sh = lambda a: a.mean(0) / a.std(0, ddof=1)  # noqa: E731
    is_, oos = [], []
    for comb in itertools.combinations(range(S), S // 2):
        i = np.concatenate([B[c] for c in comb])
        o = np.concatenate([B[c] for c in range(S) if c not in comb])
        is_.append(sh(M[i]))
        oos.append(sh(M[o]))
    res = compute_pbo(np.array(is_), np.array(oos))
    return dict(pbo=float(res.pbo), n_combinaciones=int(res.n_combinations), n_variantes=int(res.n_strategies))


def dsr(R: pd.DataFrame, n_intentos: int) -> dict:
    series = [R[c].to_numpy() for c in R]
    r = deflated_sharpe_ratio(series, frequency="daily", effective_trials=float(max(n_intentos, len(series))))
    shs = [sharpe(s) for s in series]
    return dict(
        dsr=float(r.probability),
        lider=R.columns[int(np.argmax(shs))],
        sharpe_lider=float(np.max(shs)),
        n_intentos=int(max(n_intentos, len(series))),
    )
