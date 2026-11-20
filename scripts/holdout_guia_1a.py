"""Holdout 2013-2026 de la línea 1A (guía): UNA sola ejecución con el pipeline congelado.

Requisitos: GUIA1A_HOLDOUT=ABRIR-HOLDOUT-UNA-SOLA-VEZ y que no exista resultados/guia1a/<v>/holdout/ABIERTO.
Congelado del descubrimiento: variables estables del catálogo (08_catalogo.csv; si está vacío, los supervivientes
del MDA), modelo elegido por la escalera (04_modelos.json) y el GBM, hiperparámetros del pre-registro.
Walk-forward expansivo con reentreno anual desde 1992 (purga de 5 sesiones) para 2013-2026; se reportan 2013-2026,
2013-2022 (contaminado por la línea 1A anterior) y 2023-2026 (virgen).

Criterio de la guía (Fase 5): mismo signo; magnitud dentro del intervalo del descubrimiento (encoger 30-50 % es
normal); Sharpe aceptable. Se reporta la pendiente de y sobre cada variable estable con t NW frente al intervalo
del 95 % del descubrimiento, el IC, el R2 OOS, Clark-West y la estrategia largo/neutral neta de 2 pb.

`--prueba`: ejecuta lo mismo dentro del descubrimiento (walk-forward 2008-2012) para comprobar el código sin
leer el holdout ni dejar la marca.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guia1a import config as C  # noqa: E402
from guia1a import datos, estad, figuras, variables  # noqa: E402
from guia1a.cv import walk_forward  # noqa: E402
from guia1a.estrategia import backtest, metricas, referencias, tramos  # noqa: E402
from guia1a.modelos import ESCALERA  # noqa: E402
from guia1a.oos import media_historica  # noqa: E402
from guia1a.variables import z_expanding  # noqa: E402

R = C.RESULTADOS


def psr(ex: pd.Series) -> float:
    """Probabilidad de que el Sharpe verdadero sea > 0 (Bailey y López de Prado), con asimetría y curtosis."""
    from ml4t.diagnostic.evaluation.stats.deflated_sharpe_ratio import deflated_sharpe_ratio

    return float(deflated_sharpe_ratio(ex.dropna().to_numpy(), frequency="daily").probability)


def evaluar(fin, inicio, subperiodos: dict, out: Path, cols_forzadas: list[str] | None = None):
    cl = json.loads((R / "02_clusters.json").read_text())
    mods = json.loads((R / "04_modelos.json").read_text())
    tm = pd.read_csv(R / "02_mda.csv")
    superv = tm.loc[tm.sobrevive, "cluster"].tolist()
    elegido = mods["elegido"]
    cat = pd.read_csv(R / "08_catalogo.csv")
    estables = cat.loc[cat.decision == "estable", "cluster"].unique().tolist()
    cols = cols_forzadas or estables or superv or list(cl["alineacion"])
    print("variables:", cols, "| modelos:", sorted({elegido, "3_gbm"}))

    d = datos.cargar(fin)
    P = variables.panel(d)
    X = pd.DataFrame(index=P.index)
    for nombre, al in cl["alineacion"].items():
        X[nombre] = P[nombre] if len(al) == 1 else z_expanding(sum(s * P[m] for m, s in al.items()) / len(al))
    ok = pd.concat([X[cols], P["y"]], axis=1).notna().all(axis=1) & (P.index >= pd.Timestamp("1992-01-03"))
    idx = P.index[ok]
    Xv, y = X.loc[idx, cols].to_numpy(), P.loc[idx, "y"].to_numpy()
    w = variables.unicidad(len(idx))

    preds, anios_fit = {}, []
    for nombre in sorted({elegido, "3_gbm"}):
        p = pd.Series(np.nan, index=idx)
        for anio, tr, te in walk_forward(idx, inicio=inicio, fin=fin):
            m = ESCALERA[nombre]().fit(Xv[tr], y[tr], w[tr])
            p.iloc[te] = m.predict(Xv[te])
            if nombre == elegido and hasattr(m, "m") and hasattr(m.m, "coef_"):
                anios_fit.append(dict(anio=anio, n_train=len(tr), a=float(m.m.intercept_), b=[float(v) for v in np.ravel(m.m.coef_)]))
        preds[nombre] = p.dropna()

    dia = pd.DataFrame({"ret_d": d["close"].pct_change(), "rf_d": d["DFF"].ffill() / 100 / 252}).loc[idx[0]:]
    r = np.log(d["close"]).diff()
    sigma_d = r.ewm(span=C.SPAN_SIGMA, min_periods=C.SPAN_SIGMA).std()
    mom = np.log(d["close"]).shift(21) - np.log(d["close"]).shift(252)
    mh = media_historica(P.loc[idx, "y"])
    pesos = referencias(dia, mom.reindex(dia.index), sigma_d.reindex(dia.index), mh.reindex(dia.index))
    refs = {k: backtest(v.where(v.index <= idx[-1]), dia) for k, v in pesos.items()}
    # Estrategias sobre todo el periodo de prueba (los tramos no se reinician en cada subperiodo).
    bts = {n: backtest(tramos((p > 0).astype(float)).reindex(dia.index).where(lambda s, p=p: (s.index >= p.index[0]) & (s.index <= p.index[-1])), dia)
           for n, p in preds.items()}

    # Pendiente del descubrimiento (fase 1) para el criterio de magnitud.
    f1 = pd.read_csv(R / "01_fase1_univariante.csv").set_index("variable")
    filas_b, filas, series = [], [], {}
    for sub, (a, b) in subperiodos.items():
        sel = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
        for c in cols:
            if c in f1.index:
                bd, td = f1.loc[c, "b_nw"] if "b_nw" in f1 else f1.loc[c, "b"], f1.loc[c, "t_nw"]
                se_d = abs(bd / td)
                bh_, th_ = estad.pendiente_t(X.loc[idx[sel], c].to_numpy()[None], y[sel], np.ones((1, sel.sum()), bool))
                filas_b.append(dict(subperiodo=sub, variable=c, n=int(sel.sum()), b=float(bh_[0]), t_nw=float(th_[0]),
                                    b_descubrimiento=bd, t_descubrimiento=td, ic95_desc=[bd - 1.96 * se_d, bd + 1.96 * se_d],
                                    mismo_signo=bool(np.sign(bh_[0]) == np.sign(bd)),
                                    dentro_ic95_desc=bool(bd - 1.96 * se_d <= bh_[0] <= bd + 1.96 * se_d),
                                    ratio_b=float(bh_[0] / bd)))
        for nombre, p in preds.items():
            p = p.loc[a:b]
            yy = P.loc[p.index, "y"].to_numpy()
            bench = mh.reindex(p.index).to_numpy()
            bt = bts[nombre].loc[a:b]
            m = metricas(bt, refs["buy_hold"].loc[a:b], refs["tsmom_12_1"].loc[a:b])
            ic, tic = estad.ic_t(p.to_numpy(), yy)
            filas.append(dict(subperiodo=sub, modelo=nombre, n=len(p), r2_oos=estad.r2_oos(yy, p.to_numpy(), bench),
                              cw_t=estad.clark_west(yy, bench, p.to_numpy()), ic=ic, t_ic=tic,
                              psr=psr(bt["neto"] - bt["rf"]), **m))
        for k, bt in refs.items():
            btk = bt.loc[a:b]
            filas.append(dict(subperiodo=sub, modelo=k, psr=psr(btk["neto"] - btk["rf"]), **metricas(btk)))
    series = dict(bts)  # series completas del periodo de prueba (no las del último subperiodo)
    # Año a año (modelo elegido frente al buy & hold).
    anual = []
    be = bts[elegido]
    for anio in sorted(set(preds[elegido].index.year)):
        s = str(anio)
        ex_m = (be["neto"] - be["rf"]).loc[s]
        ex_b = (refs["buy_hold"]["neto"] - refs["buy_hold"]["rf"]).loc[s]
        pa = preds[elegido].loc[s]
        anual.append(dict(anio=anio, ret_modelo=float((1 + be["neto"].loc[s]).prod() - 1),
                          ret_bh=float((1 + refs["buy_hold"]["neto"].loc[s]).prod() - 1),
                          exceso_vs_bh=float(ex_m.sum() - ex_b.sum()), peso_medio=float(be["w"].loc[s].mean()),
                          ic=estad.ic_t(pa.to_numpy(), P.loc[pa.index, "y"].to_numpy())[0]))
    out.mkdir(parents=True, exist_ok=True)
    tb, tm_, ta = pd.DataFrame(filas_b), pd.DataFrame(filas), pd.DataFrame(anual)
    tb.to_csv(out / "holdout_pendiente.csv", index=False)
    tm_.to_csv(out / "holdout_metricas.csv", index=False)
    ta.to_csv(out / "holdout_anual.csv", index=False)
    (out / "holdout_coeficientes.json").write_text(json.dumps(anios_fit, indent=1))
    pd.DataFrame({k: v["neto"] for k, v in {**series, **{f"ref_{k}": v.loc[preds[elegido].index[0]:] for k, v in refs.items()}}.items()}).to_parquet(out / "holdout_series_diarias.parquet")
    nombre_el = {"0_combo": "Combinación", "1_ols": "OLS", "2_ridge": "Ridge", "3_gbm": "GBM"}[elegido]
    figuras.curvas_capital({f"{nombre_el} ({', '.join(cols)})": series[elegido], "GBM": series["3_gbm"],
                            "Buy & hold": refs["buy_hold"], "TSMOM 12-1": refs["tsmom_12_1"]},
                           out / "holdout_curvas.png",
                           f"Holdout walk-forward {preds[elegido].index[0].year}-{preds[elegido].index[-1].year} (neto de 2 pb)")
    pd.set_option("display.width", 250)
    print(tb.round(4).to_string())
    print(tm_[["subperiodo", "modelo", "n", "r2_oos", "cw_t", "ic", "t_ic", "sharpe_neto", "dif_sharpe_bh", "t_dif_sharpe_bh",
               "alfa_anual", "t_alfa", "max_dd", "rotacion_anual", "psr"]].round(4).to_string())
    print(ta.round(4).to_string())


def main():
    if "--exploratorio" in sys.argv:
        # Análisis A POSTERIORI sobre el holdout ya abierto: no confirma nada, cuenta como intento y se guarda aparte.
        if not (R / "holdout" / "ABIERTO").exists() or not C.holdout_abierto():
            sys.exit("El modo exploratorio solo se usa con el holdout ya abierto y la llave puesta.")
        cols = sys.argv[sys.argv.index("--exploratorio") + 1].split(",")
        out = R / "holdout" / ("exploratorio_" + "_".join(cols))
        out.mkdir(parents=True, exist_ok=True)
        (out / "EXPLORATORIO").write_text(datetime.now(timezone.utc).isoformat(timespec="seconds")
                                          + " · análisis a posteriori, no confirmatorio; cuenta como intento\n")
        evaluar(C.HOLD_FIN, C.HOLD_INI, {"2013-2026": ("2013-01-01", "2026-12-31"),
                                          "2013-2022 (contaminado)": ("2013-01-01", "2022-12-31"),
                                          "2023-2026 (virgen)": ("2023-01-01", "2026-12-31")}, out, cols)
        return
    if "--prueba" in sys.argv:
        evaluar(C.DESC_FIN, "2008-01-01", {"2008-2012 (prueba en descubrimiento)": ("2008-01-01", "2012-12-31")},
                C.CACHE / "prueba_holdout")
        return
    if not C.holdout_abierto():
        sys.exit(f"Holdout sellado. Para abrirlo una sola vez: GUIA1A_HOLDOUT={C.LLAVE_HOLDOUT}")
    out = R / "holdout"
    marca = out / "ABIERTO"
    if marca.exists():
        sys.exit(f"El holdout ya se abrió ({marca.read_text().strip()}); repetirlo lo convierte en un intento más.")
    out.mkdir(parents=True, exist_ok=True)
    marca.write_text(datetime.now(timezone.utc).isoformat(timespec="seconds") + f" · {C.VERSION} · prereg {C.hash_prereg()[:16]}\n")
    evaluar(C.HOLD_FIN, C.HOLD_INI, {"2013-2026": ("2013-01-01", "2026-12-31"),
                                      "2013-2022 (contaminado)": ("2013-01-01", "2022-12-31"),
                                      "2023-2026 (virgen)": ("2023-01-01", "2026-12-31")}, out)


if __name__ == "__main__":
    main()
