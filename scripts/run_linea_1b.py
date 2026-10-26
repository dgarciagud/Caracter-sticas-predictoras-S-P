"""Orquestador de la línea 1B (sección 9.3, pasos 11-13).

Uso:
    python scripts/run_linea_1b.py volatilidad   # 8.1: HAR-RV con VIX frente a GJR-GARCH (QLIKE)
    python scripts/run_linea_1b.py regla         # 8.2: validez de la regla primaria en descubrimiento (puerta de 1B)
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


def paso_regla(_):
    d = volatilidad.cargar_diario()
    v = pd.read_parquet(VOL)
    if C.REGLA == "reversion":
        senal = regla.senal_reversion(d, v["sigma5_har"])
    else:
        panel = pd.read_parquet(C.CACHE / "panel_1988.parquet")  # panel semanal de la 1A v4/v5 (F05, R_trend)
        senal = regla.senal_vix_alcista(panel)
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
    )
    registro.registrar(periodo="descubrimiento", modelo=f"regla_primaria_{C.REGLA}", variables=[],
                       hiperparametros=dict(umbral=C.UMBRAL_REGLA, barrera=C.BARRERA, coste=C.COSTE_IDA_VUELTA),
                       metricas=res)
    json.dump(res, open(R / "02_regla_primaria.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paso", choices=["volatilidad", "regla"])
    a = ap.parse_args()
    dict(volatilidad=paso_volatilidad, regla=paso_regla)[a.paso](a)
