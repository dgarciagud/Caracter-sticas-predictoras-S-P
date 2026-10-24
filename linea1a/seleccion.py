"""Selección anidada (3.3): función que recibe un periodo y devuelve un catálogo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .etapa1 import Periodo, clusters, etapa1, representantes
from .etapa2 import Diseno, clasificar, etapa2

ESTABLES = ("estable_incondicional", "estable_condicional")


def seleccionar_catalogo(per: Periodo, R1: int = C.R1, R2: int = C.R2, semilla: int = C.SEMILLA, n_jobs: int = 4) -> dict:
    e1 = etapa1(per, R=R1, semilla=semilla)
    sal = e1["salida"]
    asign, rho = clusters(per.X[np.isfinite(per.X).all(1)], per.nombres)
    entran = sal[sal.clase_e1 != "descartada_e1"].variable.tolist()
    rep = representantes(asign, entran)
    vars_e2 = sorted({rep[v] for v in entran}, key=per.nombres.index)

    condicionales = {}
    for v in vars_e2:
        fila = sal.set_index("variable").loc[v]
        if fila.clase_e1 == "condicional":
            t = e1["tabla"]
            t = t[(t.variable == v) & (t.corte != "global") & t.pasa]
            condicionales[v] = t.sort_values("U1", ascending=False).corte.iloc[0]

    e2 = None
    if vars_e2:
        e2 = etapa2(per, Diseno(vars_e2, condicionales), R=R2, semilla=semilla + 2, n_jobs=n_jobs)
    Xdf = pd.DataFrame(per.X, columns=per.nombres)
    cat = clasificar(sal, None if e2 is None else e2["tabla"], rep, Xdf)
    cat["cluster"] = cat.variable.map(asign)
    cat["estado_condicional"] = cat.variable.map(condicionales).fillna("")
    return dict(etapa1=e1, etapa2=e2, catalogo=cat, clusters=asign, rho=rho, condicionales=condicionales)


def variables_modelo(cat: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Variables del modelo final: estables incondicionales + condicionales (con su estado)."""
    inc = cat[cat.clase == "estable_incondicional"].variable.tolist()
    cond = dict(cat[cat.clase == "estable_condicional"][["variable", "estado_condicional"]].values)
    return inc, cond


def n_estables(cat: pd.DataFrame) -> int:
    return int(cat.clase.isin(ESTABLES).sum())
