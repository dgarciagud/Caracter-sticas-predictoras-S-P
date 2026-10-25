"""Orquestador de la línea 1A (sección 9.3, pasos 1-10).

Uso:
    python scripts/run_linea_1a.py datos           # pasos 1-2: panel semanal point-in-time
    python scripts/run_linea_1a.py descubrimiento  # pasos 4-7: clusters, etapa 1, etapa 2, catálogo
    python scripts/run_linea_1a.py validacion      # paso 10: R2 OOS, Clark-West, económicas, DSR, PBO
    python scripts/run_linea_1a.py nula [--runs N] # paso 8: nula del procedimiento (sombra del objetivo)
    python scripts/run_linea_1a.py muestra_corta   # 7.8: amplitud real de componentes (AlgoSeek, 2017-2021)
    python scripts/run_linea_1a.py criterios       # 7.6: criterios para abrir el holdout
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linea1a import config as C  # noqa: E402
from linea1a import evaluacion as EV  # noqa: E402
from linea1a import figuras, registro  # noqa: E402
from linea1a.etapa1 import Periodo, indices_sombra, offsets, t_por_fold  # noqa: E402
from linea1a.seleccion import n_estables, seleccionar_catalogo, variables_modelo  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
PANEL = C.CACHE / "panel.parquet"
SEL = C.CACHE / ("seleccion_real.joblib" if C.VERSION == "v1" else f"seleccion_real_{C.VERSION}.joblib")
R = C.RESULTADOS


def paso_datos(_):
    from linea1a import datos

    panel = datos.construir_panel(datos.cargar_diario())
    panel.to_parquet(PANEL)
    desc = panel.loc[C.DESC_INI :, C.CANDIDATAS + ["R_vol", "R_trend"]]
    resumen = desc.describe().T[["count", "mean", "std", "min", "max"]]
    resumen.to_csv(R / "01_resumen_candidatas.csv")
    print(f"Panel: {panel.shape}, {panel.index.min().date()} a {panel.index.max().date()}")
    print(f"hash del pre-registro: {C.hash_prereg()}")


def paso_descubrimiento(args):
    panel = pd.read_parquet(PANEL)
    per = Periodo.desde_panel(panel)
    sel = seleccionar_catalogo(per, n_jobs=args.n_jobs)
    joblib.dump(sel, SEL)
    e1, e2, cat = sel["etapa1"], sel["etapa2"], sel["catalogo"]

    e1["tabla"].to_csv(R / "02_etapa1_estadisticos.csv", index=False)
    pd.DataFrame({"M1": e1["M1"], "M2": e1["M2"]}).to_csv(R / "02_etapa1_nula_maximos.csv", index=False)
    sel["rho"].to_csv(R / "02_spearman_descubrimiento.csv")
    registro.registrar(
        periodo="descubrimiento",
        variables=C.CANDIDATAS,
        modelo="etapa1_univariante",
        hiperparametros=dict(R=C.R1, cortes=C.CORTES),
        metricas=dict(q1=e1["q1"], q2=e1["q2"], pasan=e1["tabla"].query("pasa")[["variable", "corte"]].values.tolist()),
        semilla=C.SEMILLA,
    )
    if e2 is not None:
        t2 = e2["tabla"]
        t2.to_csv(R / "03_etapa2_estadisticos.csv", index=False)
        for m in ("ridge", "elastic_net", "lightgbm"):
            registro.registrar(
                periodo="descubrimiento",
                variables=t2.variable.tolist(),
                modelo=f"etapa2_{m}",
                hiperparametros=dict(lineal=e2["hp"]) if m != "lightgbm" else C.LGBM_PARAMS,
                metricas=dict(cumplen=t2[t2.cumple_lineal if m != "lightgbm" else t2.cumple_gbm].variable.tolist()),
                semilla=C.SEMILLA + 2,
            )
        figuras.dependencia(e2, t2, C.FIGURAS / "dependencia_shap.png")
    cat.to_csv(R / "04_catalogo_1A.csv", index=False)
    tf = t_por_fold(per)
    tf.to_csv(R / "02_etapa1_t_por_fold.csv", index=False)
    figuras.etapa1_folds(tf, e1["tabla"], e1["q1"], C.FIGURAS / "etapa1_t_por_fold.png")
    print(e1["salida"].to_string(index=False))
    print(cat[["variable", "clase_e1", "clase", "motivo"]].to_string(index=False))


def _validar(panel, cat) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    inc, cond = variables_modelo(cat)
    out = EV.predecir(panel, inc, cond)
    tabla, estr = EV.informe(out)
    return out, tabla, estr


def paso_validacion(_):
    panel = pd.read_parquet(PANEL)
    sel = joblib.load(SEL)
    cat = sel["catalogo"]
    inc, cond = variables_modelo(cat)
    out, tabla, estr = _validar(panel, cat)
    out.to_csv(R / "05_predicciones_validacion.csv")
    if not (inc or cond):
        # Catálogo vacío: el modelo final se reduce a la media histórica (R2 OOS = 0 por construcción).
        tabla = tabla[tabla.modelo.isin(["media_historica", "comprar_mantener"])]
        tabla.to_csv(R / "05_validacion_metricas.csv", index=False)
        registro.registrar(
            periodo="validacion",
            variables=[],
            modelo="catalogo_vacio_media_historica",
            hiperparametros=dict(gamma=C.GAMMA, w_max=C.W_MAX, coste_pb=C.COSTE_PB),
            metricas=tabla.set_index("modelo")[["sharpe_neto", "cer_anual"]].to_dict(),
        )
        nota = "catálogo vacío: sin configuraciones que comparar; DSR y PBO no aplican"
        json.dump(dict(dsr=None, pbo=None, nota=nota, variables=[], condicionales={}), open(R / "05_dsr_pbo.json", "w"), indent=2)
        pd.DataFrame({m: (1 + e.r).cumprod() for m, e in estr.items() if m in tabla.modelo.values}).to_csv(
            R / "05_curvas_capital_validacion.csv"
        )
        print(tabla.round(4).to_string(index=False))
        print(nota)
        return
    configs = ["ridge", "ridge_ct", "gbm", "gbm_ct"]
    variantes = {}
    sospechosas = cat[cat.clase == "supresion_sospechosa"].variable.tolist()
    if sospechosas and (inc or cond):
        out_s = EV.predecir(panel, inc + sospechosas, cond)
        tab_s, est_s = EV.informe(out_s, modelos=("ridge",))
        tab_s["modelo"] = "ridge+sospechosas"
        tabla = pd.concat([tabla, tab_s[tab_s.modelo == "ridge+sospechosas"]])
        variantes["ridge+sospechosas"] = est_s["ridge"]
    for _, f in tabla[tabla.modelo.isin(configs + list(variantes))].iterrows():
        registro.registrar(
            periodo="validacion",
            variables=inc + [f"{v}|{s}" for v, s in cond.items()],
            modelo=f.modelo,
            hiperparametros=dict(gamma=C.GAMMA, w_max=C.W_MAX, coste_pb=C.COSTE_PB),
            metricas={k: f[k] for k in ("r2_oos", "cw_p", "sharpe_neto", "cer_vs_media")},
        )
    tabla.to_csv(R / "05_validacion_metricas.csv", index=False)

    exceso = {m: estr[m].exceso for m in configs}
    exceso.update({m: e.exceso for m, e in variantes.items()})
    n_trials = len(registro.leer("1A"))
    d = EV.dsr(exceso, n_trials)
    pbo = EV.cscv_pbo(pd.DataFrame(exceso).dropna(), S=16)
    json.dump(dict(dsr=d, pbo=pbo, variables=inc, condicionales=cond), open(R / "05_dsr_pbo.json", "w"), indent=2)
    pd.DataFrame({m: (1 + e.r).cumprod() for m, e in estr.items()}).to_csv(R / "05_curvas_capital_validacion.csv")
    print(tabla.round(4).to_string(index=False))
    print(json.dumps(dict(dsr=d, pbo=pbo), indent=2))


def _una_nula(r, panel, per, k, n_jobs, R1, R2):
    T = len(per.fechas)
    idx, ok = indices_sombra(T, int(k))
    y = per.y[idx].copy()
    ym = per.ym[idx] & ok
    sel = seleccionar_catalogo(per.con_y(y, ym), R1=R1, R2=R2, semilla=C.SEMILLA + 1000 + r, n_jobs=n_jobs)
    cat = sel["catalogo"]
    out, tabla, _ = _validar(panel, cat)
    fila = tabla.set_index("modelo").loc["ridge"]
    return dict(
        replica=r,
        k=int(k),
        n_estables=n_estables(cat),
        estables=";".join(cat[cat.clase.str.startswith("estable")].variable),
        r2_oos=float(fila.r2_oos),
        sharpe_neto=float(fila.sharpe_neto),
    )


def paso_nula(args):
    panel = pd.read_parquet(PANEL)
    per = Periodo.desde_panel(panel)
    ruta = R / "06_nula_procedimiento.csv"
    hechas = pd.read_csv(ruta) if ruta.exists() else pd.DataFrame(columns=["replica"])
    ks = offsets(len(per.fechas), C.RY, C.SEMILLA + 999)
    for r in range(args.runs):
        if r in set(hechas.replica):
            continue
        fila = _una_nula(r, panel, per, ks[r], args.n_jobs, args.r1, args.r2)
        hechas = pd.concat([hechas, pd.DataFrame([fila])], ignore_index=True)
        hechas.to_csv(ruta, index=False)
        print(fila, flush=True)


def paso_muestra_corta(_):
    """7.8: F09 (proxy con SPDR sectoriales) frente a la amplitud real de componentes (AlgoSeek)."""
    C.usar_repo_ml4t()
    from data import load_sp500_daily_bars

    from linea1a.estadistica import monotonia_quintiles, pendiente_t

    bars = load_sp500_daily_bars(start_date="2016-09-01", end_date="2021-12-31").to_pandas()
    # Cierre ajustado por splits (close * adj_factor) y por sec_id, como en el caso de estudio de Jansen.
    bars["ajustado"] = bars["close"] * bars["adj_factor"]
    px = bars.pivot_table(index="timestamp", columns="sec_id", values="ajustado")
    px.index = pd.to_datetime(px.index).astype("datetime64[ns]")
    ma = px.rolling(50, min_periods=50).mean()
    amplitud = ((px > ma) & ma.notna()).sum(1) / ma.notna().sum(1).replace(0, np.nan)

    panel = pd.read_parquet(PANEL)
    p = panel.loc["2017-01-01":"2021-12-31"].copy()
    p["F09_real"] = amplitud.reindex(p.index)
    p = p.dropna(subset=["F09", "F09_real", "y"])
    filas = []
    for v in ("F09", "F09_real"):
        x = p[v].to_numpy()[None, :]
        m = np.ones_like(x, bool)
        b, t, n = pendiente_t(x, p.y.to_numpy(), m, 3)
        rho = monotonia_quintiles(x, p.y.to_numpy(), m)
        filas.append(dict(variable=v, pendiente=float(b[0]), t_nw=float(t[0]), monotonia_quintiles=float(rho[0]), n=int(n[0])))
    res = pd.DataFrame(filas)
    res["corr_spearman_proxy_real"] = p[["F09", "F09_real"]].corr("spearman").iloc[0, 1]
    res.to_csv(R / "07_muestra_corta_F09.csv", index=False)
    print(res.to_string(index=False))


def paso_criterios(_):
    val = pd.read_csv(R / "05_validacion_metricas.csv").set_index("modelo")
    dp = json.load(open(R / "05_dsr_pbo.json"))
    sel = joblib.load(SEL)
    fila = val.loc["ridge"] if "ridge" in val.index else val.loc["media_historica"]
    real = dict(n_estables=n_estables(sel["catalogo"]), r2_oos=float(fila.r2_oos), sharpe_neto=float(fila.sharpe_neto))
    ruta_n = R / "06_nula_procedimiento.csv"
    nula = pd.read_csv(ruta_n) if ruta_n.exists() else pd.DataFrame()
    crit = {
        "R2_OOS>0 y CW p<0.10 (ridge)": bool(real["r2_oos"] > 0 and "cw_p" in fila and fila.cw_p < 0.10),
        "DSR>0.95": bool(dp["dsr"] is not None and dp["dsr"]["dsr"] > 0.95),
        "PBO<0.5": bool(dp["pbo"] is not None and dp["pbo"]["pbo"] < 0.5),
    }
    if len(nula):
        p95 = {c: float(np.nanpercentile(nula[c], 95)) for c in ("n_estables", "r2_oos", "sharpe_neto")}
        crit["n_estables > p95 nula"] = bool(real["n_estables"] > p95["n_estables"])
        crit["R2_OOS > p95 nula"] = bool(real["r2_oos"] > p95["r2_oos"])
        figuras.nula_procedimiento(nula, real, C.FIGURAS / "nula_procedimiento.png")
    else:
        p95 = {}
    resumen = dict(real=real, p95_nula=p95, n_nula=len(nula), criterios=crit, abrir_holdout=all(crit.values()) and len(nula) > 0)
    json.dump(resumen, open(R / "08_criterios_holdout.json", "w"), indent=2, default=float)
    print(json.dumps(resumen, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paso", choices=["datos", "descubrimiento", "validacion", "nula", "muestra_corta", "criterios"])
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--runs", type=int, default=C.RY)
    ap.add_argument("--r1", type=int, default=C.R1)
    ap.add_argument("--r2", type=int, default=C.R2)
    a = ap.parse_args()
    dict(
        datos=paso_datos,
        descubrimiento=paso_descubrimiento,
        validacion=paso_validacion,
        nula=paso_nula,
        muestra_corta=paso_muestra_corta,
        criterios=paso_criterios,
    )[a.paso](a)
