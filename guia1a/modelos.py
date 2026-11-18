"""Escalera de modelos (A.5): 0 combinación de univariantes, 1 OLS, 2 ridge, 3 GBM (5 semillas).

Las cuatro piezas fijas: variables estandarizadas con el pasado, objetivo escalado por volatilidad,
pesos por unicidad y las mismas particiones.
"""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from . import config as C


class Combo:
    """Media de los pronósticos de regresiones univariantes (Rapach, Strauss y Zhou, 2010)."""

    def fit(self, X, y, w):
        self.m = [LinearRegression().fit(X[:, [j]], y, sample_weight=w) for j in range(X.shape[1])]
        return self

    def predict(self, X):
        return np.mean([m.predict(X[:, [j]]) for j, m in enumerate(self.m)], axis=0)


class OLS:
    def fit(self, X, y, w):
        self.m = LinearRegression().fit(X, y, sample_weight=w)
        return self

    def predict(self, X):
        return self.m.predict(X)


class RidgeTS:
    """StandardScaler ajustado con el entrenamiento + ridge con alpha por TimeSeriesSplit(5) interno."""

    def fit(self, X, y, w):
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        a = RidgeCV(alphas=C.ALPHAS_RIDGE, cv=TimeSeriesSplit(5)).fit(Z, y).alpha_
        self.alpha = float(a)
        self.m = Ridge(alpha=a).fit(Z, y, sample_weight=w)
        return self

    def predict(self, X):
        return self.m.predict(self.sc.transform(X))

    @property
    def coef(self):
        return self.m.coef_


class GBM:
    """LightGBM pequeño y restringido; predicción = media de las semillas."""

    def __init__(self, semillas=None, interaction_constraints=None, **extra):
        self.semillas = C.SEMILLAS_GBM if semillas is None else semillas
        self.params = {**C.LGBM, **extra}
        if interaction_constraints is not None:
            self.params["interaction_constraints"] = interaction_constraints

    def fit(self, X, y, w):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ms = [lgb.LGBMRegressor(**self.params, random_state=s).fit(X, y, sample_weight=w) for s in self.semillas]
        return self

    def predict(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.mean([m.predict(X) for m in self.ms], axis=0)


ESCALERA = {"0_combo": Combo, "1_ols": OLS, "2_ridge": RidgeTS, "3_gbm": GBM}
