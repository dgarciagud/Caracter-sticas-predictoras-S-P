"""Orquestador de la línea 1B (sección 9.3, pasos 11-13).

Uso:
    python scripts/run_linea_1b.py volatilidad   # 8.1: HAR-RV con VIX frente a GJR-GARCH (QLIKE)
    python scripts/run_linea_1b.py regla         # 8.2: validez de la regla primaria en descubrimiento (puerta de 1B)
    python scripts/run_linea_1b.py calibrar      # percentil p* de los umbrales (nula con meta-labels desplazados)
    python scripts/run_linea_1b.py seleccion     # 8.6-8.7: etapas 1 y 2 del meta-label con p*
    python scripts/run_linea_1b.py validacion    # 8.8-8.9: 2019-2022, primario solo frente a filtrado, nula de filtros
    python scripts/run_linea_1b.py nula          # 8.9: nula del procedimiento (meta-labels desplazados, R = 100)
    python scripts/run_linea_1b.py criterios     # resumen frente a los criterios
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linea1a.estadistica import media_nw  # noqa: E402
from linea1b import config as C  # noqa: E402
from linea1b import registro, regla, volatilidad  # noqa: E402

warnings.filterwarnings("ignore")
R = C.RESULTADOS
VOL = C.CACHE / "1B_volatilidad.parquet"


def paso_volatilidad(_):
    d = volatilidad.cargar_diario()
    har = volatilidad.sigma5_har(d)
    gjr = volatilidad.sigma5_gjr(d)
    r2 = d["r"] ** 2
    rv5 = sum(r2.shift(-k) for k in range(1, C.H + 1))
    v = pd.DataFrame({"sigma5_har": har, "sigma5_gjr": gjr, "rv5_futura": rv5})
    v.to_parquet(VOL)
    sem = regla.fechas_decision(d.index)
    filas = []
    for nombre, (a, b) in {"descubrimiento": (C.DESC_INI, C.DESC_FIN), "validacion": (C.VAL_INI, C.VAL_FIN)}.items():
        w = v.loc[sem].loc[a:b].dropna()
        filas.append(dict(periodo=nombre, semanas=len(w),
                          qlike_har=volatilidad.qlike(w.rv5_futura.to_numpy(), w.sigma5_har.to_numpy() ** 2),
                          qlike_gjr=volatilidad.qlike(w.rv5_futura.to_numpy(), w.sigma5_gjr.to_numpy() ** 2),
                          corr_har_gjr=float(w.sigma5_har.corr(w.sigma5_gjr))))
    t = pd.DataFrame(filas)
    t.to_csv(R / "01_volatilidad_qlike.csv", index=False)
    print(t.round(4).to_string(index=False))


def _senal(d, v, panel=None):
    if C.REGLA == "reversion":
        return regla.senal_reversion(d, v["sigma5_har"])
    if C.REGLA == "reversion_laxa":
        return regla.senal_reversion_laxa(d, v["sigma5_har"])
    panel = pd.read_parquet(C.CACHE / "panel_1988.parquet") if panel is None else panel  # panel semanal de la 1A v4/v5
    return regla.senal_vix_alcista(panel)


def paso_regla(_):
    d = volatilidad.cargar_diario()
    v = pd.read_parquet(VOL)
    senal = _senal(d, v)
    ev = regla.eventos(d, v["sigma5_har"], C.DESC_INI, C.DESC_FIN, senal)
    ev.to_csv(R / "02_eventos_descubrimiento.csv")
    media, t = media_nw(ev.ret_neto.to_numpy())
    res = dict(
        eventos=len(ev),
        eventos_por_anio=float(len(ev) / ((C.DESC_FIN - C.DESC_INI).days / 365.25)),
        media_ret_neto=media,
        t_nw=t,
        tasa_acierto_meta_label=float(ev.meta_label.mean()),
        salidas=ev.motivo.value_counts().to_dict(),
        ret_medio_por_salida=ev.groupby("motivo").ret_neto.mean().to_dict(),
        cumple_requisito=bool(media > 0 and t > C.T_MIN_REGLA),
        puerta_obligatoria=C.PUERTA_OBLIGATORIA,
    )
    if C.REGLA == "reversion_laxa":
        sem = regla.fechas_decision(d.loc[C.DESC_INI : C.DESC_FIN].index)
        res["fraccion_semanas_con_evento"] = float(len(ev) / len(sem))
        from linea1b import meta as M

        res["eventos_por_fold_test"] = [int(len(te)) for _, te in M.folds(ev)]
    registro.registrar(periodo="descubrimiento", modelo=f"regla_primaria_{C.REGLA}", variables=[],
                       hiperparametros=dict(umbral=C.UMBRAL_REGLA, barrera=C.BARRERA, coste=C.COSTE_IDA_VUELTA),
                       metricas=res)
    json.dump(res, open(R / "02_regla_primaria.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


# ------------------------------------------------------------------ meta-label (1B-v2 y 1B-v3)
REJILLA_P = [95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 50.0]
SEMILLA = 20261027
SEL_CACHE = C.CACHE / f"1B_{C.VERSION.split('-')[1]}_seleccion.joblib"


def _eventos_desc():
    from linea1b import meta as M

    ev = pd.read_csv(R / "02_eventos_descubrimiento.csv", index_col=0, parse_dates=["fecha", "fin"])
    return M.Eventos.construir(ev, pd.read_parquet(C.CACHE / "panel_1988.parquet"))


def _nula_meta(E, r, k, R1, R2, rejilla):
    """Meta-labels (y sus pesos) desplazados k posiciones sobre la secuencia de eventos; selección con cada p."""
    from linea1b import meta as M

    En = M.Eventos(E.ev, E.X, np.roll(E.y, k), np.roll(E.w, k), E.cal, E.pos)
    e1 = M.etapa1(En, R=R1, semilla=SEMILLA + 100 + r)
    cache, filas = {}, []
    for p in rejilla:
        sel = M.seleccionar(En, p, R2=R2, semilla=SEMILLA + 100 + r, e1=e1, cache_e2=cache)
        filas.append(dict(replica=r, k=int(k), percentil=p, pasan_e1=";".join(sel["pasan_e1"]),
                          meta_features=";".join(sel["meta_features"]), n=len(sel["meta_features"])))
    return filas


def paso_calibrar(a):
    from joblib import Parallel, delayed

    E = _eventos_desc()
    n = len(E.y)
    ks = np.random.default_rng(SEMILLA + 3333).integers(10, n - 10 + 1, size=a.runs)
    res = Parallel(n_jobs=4)(delayed(_nula_meta)(E, r, k, a.r1, a.r2, REJILLA_P) for r, k in enumerate(ks))
    d = pd.DataFrame([f for fs in res for f in fs])
    d.to_csv(R / "03_calibracion_detalle.csv", index=False)
    t = d.groupby("percentil").n.apply(lambda s: (s > 0).mean()).rename("tasa_falsos").sort_index(ascending=False)
    validos = t[t <= 0.05].index
    p_star = float(min(validos)) if len(validos) else 95.0
    t.to_csv(R / "03_calibracion_resumen.csv")
    (R / "03_p_estrella.txt").write_text(f"{p_star}\n")
    print(t.to_string(), f"\np* = {p_star}")


def _p_estrella() -> float:
    return float((R / "03_p_estrella.txt").read_text())


def paso_seleccion(_):
    import joblib

    from linea1b import meta as M

    E = _eventos_desc()
    p = _p_estrella()
    sel = M.seleccionar(E, p, R1=1000, R2=100, semilla=SEMILLA)
    joblib.dump(sel, SEL_CACHE)
    e1 = sel["e1"]
    t1 = pd.DataFrame(dict(feature=M.FEATURES, U=e1["U"], umbral=float(np.nanpercentile(e1["M"], p)),
                           p_perm=[float((e1["M"] >= u).mean()) for u in e1["U"]],
                           dif_tercil_alto_bajo=[e1["terciles"][f] for f in M.FEATURES]))
    t1["pasa"] = t1.U > t1.umbral
    t1.to_csv(R / "04_etapa1.csv", index=False)
    if len(sel["tabla_e2"]):
        sel["tabla_e2"].to_csv(R / "04_etapa2.csv", index=False)
    registro.registrar(periodo="descubrimiento", modelo="meta_label_seleccion", variables=M.FEATURES,
                       hiperparametros=dict(percentil=p, R1=1000, R2=100),
                       metricas=dict(pasan_e1=sel["pasan_e1"], meta_features=sel["meta_features"]))
    print(t1.round(4).to_string(index=False))
    print(sel["tabla_e2"].round(4).to_string(index=False) if len(sel["tabla_e2"]) else "sin etapa 2")
    print("meta-features:", sel["meta_features"])


def _validar(E, meta_features, d, v, panel):
    """Predicción anual en 2019-2022 y P&L semanal de las tres estrategias (8.8)."""
    from linea1b import meta as M

    senal = _senal(d, v, panel)
    ev_val = regla.eventos(d, v["sigma5_har"], C.VAL_INI, C.VAL_FIN, senal)
    Xv = panel.loc[ev_val.index, M.FEATURES].to_numpy(float)
    yv = ev_val.meta_label.to_numpy(int)
    todos = pd.concat([E.ev, ev_val])
    Xt = np.vstack([E.X, Xv])
    yt = np.r_[E.y, yv]
    wt = M.pesos(todos)
    j = [M.FEATURES.index(f) for f in meta_features]
    pred = {"logistica": np.full(len(ev_val), np.nan), "ebm": np.full(len(ev_val), np.nan)}
    for anio in sorted(set(ev_val.index.year)):
        te = np.where(ev_val.index.year == anio)[0]
        t0 = ev_val.index[te[0]]
        tr = np.where((todos.index < t0) & (pd.to_datetime(todos["fin"]).to_numpy() < t0.to_datetime64()))[0]
        for tipo in pred:
            if not j:
                pred[tipo][te] = np.average(yt[tr], weights=wt[tr])  # tasa base: sin meta-features
                continue
            mu, sd = Xt[tr][:, j].mean(0), Xt[tr][:, j].std(0)
            m = M.meta_modelo((Xt[tr][:, j] - mu) / sd, yt[tr], wt[tr], tipo)
            pred[tipo][te] = m.predict_proba((Xv[te][:, j] - mu) / sd)[:, 1]
    objetivo = float(np.median(E.ev["sigma5"]))
    base = objetivo / ev_val["sigma5"].to_numpy()
    tam = {"primario": base}
    for tipo, p_ in pred.items():
        tam[f"filtrado_{tipo}"] = np.where(p_ > 0.55, np.minimum(2 * p_ - 1, 1) * base, 0.0)
    semanas = regla.fechas_decision(d.loc[C.VAL_INI : C.VAL_FIN].index)
    pnl = pd.DataFrame(0.0, index=semanas, columns=list(tam))
    for c, t_ in tam.items():
        pnl.loc[ev_val.index, c] = t_ * ev_val["ret_neto"].to_numpy()
    return ev_val, pred, tam, pnl


def _sharpe(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(52)) if x.std() > 0 else np.nan


def paso_validacion(_):
    import joblib
    from sklearn.metrics import brier_score_loss, roc_auc_score

    from linea1a.evaluacion import cscv_pbo, dsr

    E = _eventos_desc()
    sel = joblib.load(SEL_CACHE)
    d = volatilidad.cargar_diario()
    v = pd.read_parquet(VOL)
    panel = pd.read_parquet(C.CACHE / "panel_1988.parquet")
    ev_val, pred, tam, pnl = _validar(E, sel["meta_features"], d, v, panel)
    ev_val.assign(p_logistica=pred["logistica"], p_ebm=pred["ebm"]).to_csv(R / "05_eventos_validacion.csv")
    pnl.to_csv(R / "05_pnl_semanal_validacion.csv")
    y = ev_val.meta_label.to_numpy()
    media, t = media_nw(ev_val.ret_neto.to_numpy())
    filas = []
    for c in pnl:
        fila = dict(estrategia=c, operaciones=int((tam[c] > 0).sum()), sharpe_neto=_sharpe(pnl[c]),
                    rent_anual=float(pnl[c].mean() * 52), vol_anual=float(pnl[c].std() * np.sqrt(52)))
        tipo = c.replace("filtrado_", "")
        if tipo in pred:
            p_ = pred[tipo]
            fila.update(auc=float(roc_auc_score(y, p_)) if np.ptp(p_) > 0 else np.nan, brier=float(brier_score_loss(y, p_)),
                        precision_p055=float(y[p_ > 0.55].mean()) if (p_ > 0.55).any() else np.nan)
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    # Nula de filtros aleatorios (8.9): mismo número de eventos por año, tamaños del filtro real permutados.
    rng = np.random.default_rng(SEMILLA + 77)
    nula = {}
    for c in [x for x in tam if x.startswith("filtrado_")]:
        elegidos = tam[c] > 0
        sh = []
        for _ in range(1000):
            sel_r = np.zeros(len(ev_val), bool)
            for anio in set(ev_val.index.year):
                idx = np.where(ev_val.index.year == anio)[0]
                nsel = int(elegidos[idx].sum())
                sel_r[rng.choice(idx, nsel, replace=False)] = True
            t_r = np.zeros(len(ev_val))
            t_r[sel_r] = rng.permutation(tam[c][elegidos])
            p_r = pd.Series(0.0, index=pnl.index)
            p_r.loc[ev_val.index] = t_r * ev_val["ret_neto"].to_numpy()
            sh.append(_sharpe(p_r))
        nula[c] = dict(p95=float(np.nanpercentile(sh, 95)), p_valor=float((np.array(sh) >= _sharpe(pnl[c])).mean()))
    for c in pnl:
        registro.registrar(periodo="validacion", modelo=f"1B_{c}", variables=sel["meta_features"],
                           hiperparametros=dict(umbral_p=0.55), metricas=tabla.set_index("estrategia").loc[c].to_dict())
    from linea1a import registro as R1A

    n_ensayos = len(R1A.leer("1A")) + len(R1A.leer("1B"))
    exceso = {c: pnl[c] for c in pnl}
    distintas = pd.DataFrame(exceso).loc[:, ~pd.DataFrame(exceso).T.duplicated()]
    res = dict(eventos_validacion=len(ev_val), ret_neto_medio_primario=media, t_nw_primario=t,
               acierto_primario=float(y.mean()), meta_features=sel["meta_features"],
               nula_filtros=nula, dsr=dsr(exceso, n_ensayos),
               pbo=cscv_pbo(distintas, S=16) if distintas.shape[1] >= 2 else None,
               estrategias_distintas=list(distintas.columns))
    tabla.to_csv(R / "05_validacion.csv", index=False)
    json.dump(res, open(R / "05_validacion.json", "w"), indent=2, default=float)
    print(tabla.round(4).to_string(index=False))
    print(json.dumps(res, indent=2, default=float))


def paso_nula(a):
    """Nula del procedimiento (8.9): meta-labels desplazados, selección con p* y validación con los reales."""
    from joblib import Parallel, delayed

    E = _eventos_desc()
    p = _p_estrella()
    n = len(E.y)
    ks = np.random.default_rng(SEMILLA + 999).integers(10, n - 10 + 1, size=a.runs)
    res = Parallel(n_jobs=4)(delayed(_nula_meta)(E, r, k, a.r1, a.r2, [p]) for r, k in enumerate(ks))
    d = pd.DataFrame([f for fs in res for f in fs])
    d.to_csv(R / "06_nula_procedimiento.csv", index=False)
    print(d.n.value_counts().to_dict(), d.meta_features.value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paso", choices=["volatilidad", "regla", "calibrar", "seleccion", "validacion", "nula"])
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--r1", type=int, default=1000)
    ap.add_argument("--r2", type=int, default=20)
    a = ap.parse_args()
    dict(volatilidad=paso_volatilidad, regla=paso_regla, calibrar=paso_calibrar, seleccion=paso_seleccion,
         validacion=paso_validacion, nula=paso_nula)[a.paso](a)
