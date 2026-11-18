"""Test anti look-ahead (guía, A.10): recortar los datos en una fecha d, recalcular todas las variables
y comprobar que todo lo anterior a d sale idéntico. Las etiquetas y el calendario (t, t+5] de las
últimas h sesiones dependen por definición del futuro y quedan fuera de la comparación."""

import numpy as np
import pandas as pd
import pytest

from guia1a import config as C
from guia1a import datos, variables

CORTES = ["1998-06-30", "2005-03-15", "2010-11-30"]
FUTURAS = ["ret_h", "y", "CAL_fomc", "CAL_tom", "CAL_opex"]


@pytest.fixture(scope="module")
def completo():
    return variables.panel(datos.cargar(C.DESC_FIN))


@pytest.mark.parametrize("corte", CORTES)
def test_sin_informacion_futura(completo, corte):
    rec = variables.panel(datos.cargar(corte))
    a = completo.loc[rec.index]
    pasado = [c for c in rec.columns if c not in FUTURAS]
    np.testing.assert_allclose(rec[pasado].to_numpy(), a[pasado].to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)
    # Columnas que miran (t, t+h]: idénticas salvo en las últimas h sesiones, que el recorte deja en NaN.
    cab = rec.index[: -C.H]
    np.testing.assert_allclose(rec.loc[cab, FUTURAS].to_numpy(), a.loc[cab, FUTURAS].to_numpy(), equal_nan=True)
    assert rec[FUTURAS].iloc[-C.H :].isna().all().all()


def test_holdout_sellado():
    with pytest.raises(PermissionError):
        datos.cargar("2013-01-02")
