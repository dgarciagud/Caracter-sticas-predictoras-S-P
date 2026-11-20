"""Holdout 2013-2026 de la línea 1A (guía): UNA sola ejecución con el pipeline congelado.

Requisitos: GUIA1A_HOLDOUT=ABRIR-HOLDOUT-UNA-SOLA-VEZ y que no exista resultados/guia1a/<v>/holdout/ABIERTO.
Congelado del descubrimiento: clusters y su alineación (02_clusters.json), variables estables del catálogo
(08_catalogo.csv; si está vacío, los supervivientes del MDA),
modelo elegido por la escalera (04_modelos.json), hiperparámetros del pre-registro.
Evalúa en walk-forward expansivo con reentreno anual desde 1992 (purga de 5 sesiones) para 2013-2026 y
reporta 2013-2022 (contaminado por la línea 1A anterior) y 2023-2026 (virgen) por separado.
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
from guia1a import datos, estad, variables  # noqa: E402
from guia1a.cv import walk_forward  # noqa: E402
from guia1a.estrategia import backtest, metricas, referencias, tramos  # noqa: E402
from guia1a.modelos import ESCALERA  # noqa: E402
from guia1a.oos import media_historica  # noqa: E402
from guia1a.variables import z_expanding  # noqa: E402

OUT = C.RESULTADOS / "holdout"
MARCA = OUT / "ABIERTO"
SUBPERIODOS = {"2013-2026": ("2013-01-01", "2026-12-31"), "2013-2022 (contaminado)": ("2013-01-01", "2022-12-31"),
               "2023-2026 (virgen)": ("2023-01-01", "2026-12-31")}


def main():
    if not C.holdout_abierto():
        sys.exit(f"Holdout sellado. Para abrirlo una sola vez: GUIA1A_HOLDOUT={C.LLAVE_HOLDOUT}")
    if MARCA.exists():
        sys.exit(f"El holdout ya se abrió ({MARCA.read_text().strip()}); repetirlo lo convierte en un intento más.")
    OUT.mkdir(parents=True, exist_ok=True)
    MARCA.write_text(datetime.now(timezone.utc).isoformat(timespec="seconds"))
    R = C.RESULTADOS
    cl = json.loads((R / "02_clusters.json").read_text())
    mods = json.loads((R / "04_modelos.json").read_text())
    tm = pd.read_csv(R / "02_mda.csv")
    superv = tm.loc[tm.sobrevive, "cluster"].tolist()
    elegido = mods["elegido"]

    d = datos.cargar(C.HOLD_FIN)
    P = variables.panel(d)
    X = pd.DataFrame(index=P.index)
    for nombre, al in cl["alineacion"].items():
        X[nombre] = P[nombre] if len(al) == 1 else z_expanding(sum(s * P[m] for m, s in al.items()) / len(al))
    # A.6, paso 4: holdout con las variables estables del catálogo; si está vacío, con los supervivientes del MDA.
    cat = pd.read_csv(R / "08_catalogo.csv")
    estables = cat.loc[cat.decision == "estable", "cluster"].unique().tolist()
    cols = estables or superv or list(cl["alineacion"])
    print("variables del holdout:", cols)
    ok = pd.concat([X[cols], P["y"]], axis=1).notna().all(axis=1) & (P.index >= pd.Timestamp("1992-01-03"))
    idx = P.index[ok]
    Xv, y = X.loc[idx, cols], P.loc[idx, "y"].to_numpy()
    w = variables.unicidad(len(idx))
    preds = {}
    for nombre in sorted({elegido, "3_gbm"}):
        fab = ESCALERA[nombre]
        p = pd.Series(np.nan, index=idx)
        for _, tr, te in walk_forward(idx, inicio=C.HOLD_INI, fin=C.HOLD_FIN):
            p.iloc[te] = fab().fit(Xv.to_numpy()[tr], y[tr], w[tr]).predict(Xv.to_numpy()[te])
        preds[nombre] = p.dropna()
    dia = pd.DataFrame({"ret_d": d["close"].pct_change(), "rf_d": d["DFF"].ffill() / 100 / 252}).loc[idx[0]:]
    r = np.log(d["close"]).diff()
    sigma_d = r.ewm(span=C.SPAN_SIGMA, min_periods=C.SPAN_SIGMA).std()
    mom = np.log(d["close"]).shift(21) - np.log(d["close"]).shift(252)
    mh = media_historica(P.loc[idx, "y"])
    refs = {k: backtest(v.where(v.index <= idx[-1]), dia) for k, v in
            referencias(dia, mom.reindex(dia.index), sigma_d.reindex(dia.index), mh.reindex(dia.index)).items()}
    filas = []
    for sub, (a, b) in SUBPERIODOS.items():
        for nombre, p in preds.items():
            p = p.loc[a:b]
            yy = P.loc[p.index, "y"].to_numpy()
            bench = mh.reindex(p.index).to_numpy()
            bt = backtest(tramos((p > 0).astype(float)).reindex(dia.index).where(lambda s: s.index <= p.index[-1]), dia).loc[a:b]
            m = metricas(bt, refs["buy_hold"].loc[a:b], refs["tsmom_12_1"].loc[a:b])
            filas.append(dict(subperiodo=sub, modelo=nombre, r2_oos=estad.r2_oos(yy, p.to_numpy(), bench),
                              cw_t=estad.clark_west(yy, bench, p.to_numpy()), ic=estad.ic_t(p.to_numpy(), yy)[0], **m))
        for k, bt in refs.items():
            filas.append(dict(subperiodo=sub, modelo=k, **metricas(bt.loc[a:b])))
    pd.DataFrame(filas).to_csv(OUT / "holdout_metricas.csv", index=False)
    print(pd.DataFrame(filas).round(3).to_string())


if __name__ == "__main__":
    main()
