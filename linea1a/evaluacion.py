"""Evaluación fuera de muestra de 1A (sección 7): R2 OOS, Clark-West, métricas económicas, DSR y PBO."""

from __future__ import annotations

import itertools
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from ml4t.diagnostic.evaluation.stats.backtest_overfitting import compute_pbo
from ml4t.diagnostic.evaluation.stats.deflated_sharpe_ratio import deflated_sharpe_ratio
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from . import config as C
from .estadistica import clark_west, r2_oos
from .etapa2 import ALPHAS_RIDGE


def _diseno(p: pd.DataFrame, inc: list[str], cond: dict[str, str], lineal: bool) -> pd.DataFrame:
    X = p[inc].copy()
    for v, estado in cond.items():
        var, val = estado.split("=")
        X[v + "x" + estado] = p[v] * (p[var] == float(val)) if lineal else p[v]
    if not lineal:
        X[["R_vol", "R_trend"]] = p[["R_vol", "R_trend"]]
    return X


def media_historica(panel: pd.DataFrame, fechas: pd.DatetimeIndex) -> pd.Series:
    """ȳ_t: media expansiva desde 2007-04 de las etiquetas ya realizadas en t."""
    base = panel.loc[C.DESC_INI :].dropna(subset=["y"])
    out = {}
    for t in fechas:
        out[t] = base.loc[base.fin_etiqueta <= t, "y"].mean()
    return pd.Series(out)


def predecir(panel: pd.DataFrame, inc: list[str], cond: dict[str, str], ini=C.VAL_INI, fin=C.VAL_FIN) -> pd.DataFrame:
    """Reentrenamiento anual con ventana expansiva desde 2007-04: ridge (principal) y GBM (secundario)."""
    test_all = panel.loc[ini:fin]
    test_all = test_all[(test_all.fin_etiqueta <= fin) & test_all.y.notna()]
    out = pd.DataFrame(index=test_all.index)
    out["y"] = test_all.y
    out["bench"] = media_historica(panel, test_all.index)
    out["ridge"] = out["bench"]
    out["gbm"] = out["bench"]
    vars_ = inc + list(cond)
    for anio in sorted(set(test_all.index.year)):
        te = test_all[test_all.index.year == anio]
        t0 = te.index[0]
        tr = panel.loc[C.DESC_INI : t0]
        tr = tr[(tr.index < t0) & (tr.fin_etiqueta <= t0) & tr.y.notna()].dropna(subset=vars_ + ["R_vol", "R_trend"])
        if not vars_:
            continue
        for nombre, lineal in (("ridge", True), ("gbm", False)):
            Xtr, Xte = _diseno(tr, inc, cond, lineal), _diseno(te, inc, cond, lineal)
            mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
            Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
            if lineal:
                a = RidgeCV(alphas=ALPHAS_RIDGE, cv=TimeSeriesSplit(5)).fit(Ztr, tr.y).alpha_
                m = Ridge(alpha=a).fit(Ztr, tr.y)
            else:
                m = lgb.LGBMRegressor(**C.LGBM_PARAMS, random_state=C.SEMILLA).fit(Ztr, tr.y)
            ok = Zte.notna().all(1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out.loc[te.index[ok], nombre] = m.predict(Zte[ok])
    for nombre in ("ridge", "gbm"):
        out[nombre + "_ct"] = out[nombre].clip(lower=0)
    out["ret_simple"] = test_all.ret_simple
    out["rf"] = test_all.rf
    out["sigma5"] = test_all.sigma5
    return out


def estrategia(pred: pd.Series, d: pd.DataFrame, coste_pb: float = C.COSTE_PB) -> pd.DataFrame:
    w = (pred / (C.GAMMA * d.sigma5**2)).clip(0, C.W_MAX)
    dw = w.diff().abs().fillna(w.abs())
    r = d.rf + w * (d.ret_simple - d.rf) - coste_pb / 1e4 * dw
    return pd.DataFrame({"w": w, "r": r, "exceso": r - d.rf, "rotacion": dw})


def metricas_economicas(e: pd.DataFrame) -> dict:
    ex = e.exceso
    curva = (1 + e.r).cumprod()
    return dict(
        sharpe_neto=float(ex.mean() / ex.std() * np.sqrt(52)),
        cer_anual=float((e.r.mean() - C.GAMMA / 2 * e.r.var()) * 52),
        rent_anual=float(e.r.mean() * 52),
        vol_anual=float(e.r.std() * np.sqrt(52)),
        max_drawdown=float((curva / curva.cummax() - 1).min()),
        rotacion_anual=float(e.rotacion.mean() * 52),
        peso_medio=float(e.w.mean()),
    )


def informe(out: pd.DataFrame, modelos=("ridge", "ridge_ct", "gbm", "gbm_ct")) -> tuple[pd.DataFrame, dict]:
    filas, estrategias = [], {}
    bench_e = estrategia(out.bench, out)
    bh_e = estrategia(C.GAMMA * out.sigma5**2, out)  # predicción que da w = 1 en todas las semanas
    met_b, met_bh = metricas_economicas(bench_e), metricas_economicas(bh_e)
    estrategias["media_historica"], estrategias["comprar_mantener"] = bench_e, bh_e
    for m in modelos:
        e = estrategia(out[m], out)
        estrategias[m] = e
        t, p = clark_west(out.y.to_numpy(), out.bench.to_numpy(), out[m].to_numpy())
        me = metricas_economicas(e)
        filas.append(
            dict(
                modelo=m,
                r2_oos=r2_oos(out.y.to_numpy(), out[m].to_numpy(), out.bench.to_numpy()),
                cw_t=t,
                cw_p=p,
                **me,
                cer_vs_media=me["cer_anual"] - met_b["cer_anual"],
                cer_vs_bh=me["cer_anual"] - met_bh["cer_anual"],
                sharpe_coste_5pb=metricas_economicas(estrategia(out[m], out, 5))["sharpe_neto"],
                sharpe_coste_10pb=metricas_economicas(estrategia(out[m], out, 10))["sharpe_neto"],
            )
        )
    for nombre, met in (("media_historica", met_b), ("comprar_mantener", met_bh)):
        filas.append(dict(modelo=nombre, r2_oos=0.0 if nombre == "media_historica" else np.nan, **met))
    return pd.DataFrame(filas), estrategias


def cscv_pbo(R: pd.DataFrame, S: int = 16) -> dict:
    """PBO por CSCV (Bailey et al.) sobre la matriz T x N de retornos semanales; usa ml4t compute_pbo."""
    T = len(R) // S * S
    bloques = np.array_split(np.arange(T), S)
    M = R.iloc[:T].to_numpy()
    sh = lambda a: a.mean(0) / a.std(0, ddof=1)  # noqa: E731
    is_, oos = [], []
    for comb in itertools.combinations(range(S), S // 2):
        i = np.concatenate([bloques[c] for c in comb])
        o = np.concatenate([bloques[c] for c in range(S) if c not in comb])
        is_.append(sh(M[i]))
        oos.append(sh(M[o]))
    res = compute_pbo(np.array(is_), np.array(oos))
    return dict(pbo=res.pbo, n_combinaciones=res.n_combinations, n_estrategias=res.n_strategies)


def dsr(exceso: dict[str, pd.Series], n_ensayos: int) -> dict:
    series = [s.to_numpy() for s in exceso.values()]
    r = deflated_sharpe_ratio(series, frequency="weekly", effective_trials=float(max(n_ensayos, len(series))))
    return dict(
        dsr=float(r.probability),
        sharpe_lider_anual=float(r.sharpe_ratio_annualized),
        n_ensayos=int(max(n_ensayos, len(series))),
        lider=list(exceso)[int(np.argmax([np.nanmean(s) / np.nanstd(s) for s in series]))],
    )
