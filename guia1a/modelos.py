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
    """LightGBM pequeño y restringido; predicción = media de las semillas.

    g1: hiperparámetros fijos. g2 (config.GBM_G2): número de árboles por early stopping y min_child_samples
    elegido en una validación interna (último tramo del entrenamiento, separado con purga); después se reentrena
    con todo el entrenamiento. `rejilla` fija la lista de min_child_samples (un solo valor = variante fija).
    """

    def __init__(self, semillas=None, interaction_constraints=None, rejilla=None, **extra):
        self.semillas = C.SEMILLAS_GBM if semillas is None else semillas
        self.params = {**C.LGBM, **extra}
        if interaction_constraints is not None:
            self.params["interaction_constraints"] = interaction_constraints
        self.g2 = C.GBM_G2
        self.rejilla = rejilla if rejilla is not None else (self.g2["min_child_samples_rejilla"] if self.g2 else None)

    def _ajustar(self, X, y, w, params):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return [lgb.LGBMRegressor(**params, random_state=s).fit(X, y, sample_weight=w) for s in self.semillas]

    def fit(self, X, y, w):
        if not self.g2:
            self.ms = self._ajustar(X, y, w, self.params)
            return self
        es = self.g2["early_stopping"]
        n = len(y)
        ini_val = int(n * (1 - es["fraccion_validacion"]))
        tr, va = np.arange(0, max(1, ini_val - es["purga"])), np.arange(ini_val, n)
        mejor = None
        for mcs in self.rejilla:
            p = {**self.params, "min_child_samples": int(mcs), "n_estimators": int(es["max_arboles"])}
            preds, iters = [], []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for s in self.semillas:
                    m = lgb.LGBMRegressor(**p, random_state=s).fit(
                        X[tr], y[tr], sample_weight=w[tr], eval_set=[(X[va], y[va])], eval_sample_weight=[w[va]],
                        callbacks=[lgb.early_stopping(int(es["paciencia"]), verbose=False)],
                    )
                    iters.append(max(1, int(m.best_iteration_ or p["n_estimators"])))
                    preds.append(m.predict(X[va], num_iteration=iters[-1]))
            mse = float(np.mean([np.average((y[va] - q) ** 2, weights=w[va]) for q in preds]))
            # Empate (o mejora < 1e-12) -> el valor mayor, más conservador.
            if mejor is None or mse <= mejor[0] + 1e-12:
                mejor = (mse, int(mcs), int(np.median(iters)))
        self.min_child, self.n_arboles = mejor[1], mejor[2]
        p = {**self.params, "min_child_samples": self.min_child, "n_estimators": self.n_arboles}
        self.ms = self._ajustar(X, y, w, p)
        return self

    def predict(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.mean([m.predict(X) for m in self.ms], axis=0)


ESCALERA = {"0_combo": Combo, "1_ols": OLS, "2_ridge": RidgeTS, "3_gbm": GBM}
