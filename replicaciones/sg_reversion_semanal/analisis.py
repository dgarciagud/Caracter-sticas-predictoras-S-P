"""Tablas y figuras de la réplica, en el orden del paper (secciones y gráficos de las págs. 1-11)."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config as C
from .caracteristicas import ETIQUETAS, FEATURES, retorno_entre
from .cartera import backtest, fechas_mensuales, fechas_semanales, metricas, pesos

COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # paleta de linea1a.figuras
DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
# Factores lineales para exposiciones y atribución (los de precio/volumen de la lista de SG).
FACTORES = {
    "1-week Return (reversal)": ("ret_1w", -1),
    "12-month Price Momentum": ("mom_12m1m", 1),
    "3-month Price Momentum": ("mom_3m1m", 1),
    "Low Vol": ("vol_6m", -1),
    "Low Beta": ("beta_12m", -1),
    "Low Size (ADV)": ("log_adv_6m", -1),
}
# Paneles de exposiciones (como los dos gráficos de la pág. 4) y serie de atribución.
PANEL_A = ["Low Vol", "Low Beta", "Low Size (ADV)"]
PANEL_B = ["1-week Return (reversal)", "12-month Price Momentum", "3-month Price Momentum"]
SERIE_ATR = ["Residual", "1-week Return (reversal)", "12-month Price Momentum", "3-month Price Momentum", "Low Vol"]
if C.VERSION != "v2":
    FACTORES.pop("Low Size (ADV)")
    FACTORES.update({
        "Low Size": ("log_mcap", -1),
        "Value": ("valor", 1),
        "Quality": ("calidad", 1),
        "Growth": ("crecimiento", 1),
        "Dividend Yield": ("div_yield", 1),
        "SUE (≈ revisiones BPA)": ("sue", 1),
        "EAR (≈ revisiones BPA)": ("ear", 1),
    })
    PANEL_A = ["Value", "Quality", "Growth", "Low Vol", "Dividend Yield"]
    PANEL_B = PANEL_B + ["SUE (≈ revisiones BPA)", "EAR (≈ revisiones BPA)"]
    SERIE_ATR = ["Residual", "1-week Return (reversal)", "12-month Price Momentum", "SUE (≈ revisiones BPA)", "EAR (≈ revisiones BPA)"]


def _guardar(fig, nombre: str) -> None:
    fig.tight_layout()
    fig.savefig(C.FIGURAS / f"{nombre}.png", dpi=130)
    plt.close(fig)


def _curva(r: pd.Series) -> pd.Series:
    return 100 * (1 + r.fillna(0)).cumprod()


def _z(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = df.groupby("date")[cols]
    return ((df[cols] - g.transform("mean")) / g.transform("std")).clip(-3, 3).fillna(0.0)


class Replica:
    def __init__(self, panel: pd.DataFrame, pred: pd.DataFrame, pxf: pd.DataFrame, etf: pd.DataFrame, extra: dict):
        self.df = panel.merge(pred, on=["date", "symbol"])
        self.df["rev"] = -self.df["ret_1w"]
        self.pxf, self.extra = pxf, extra
        self.spy = etf["SPY"].ffill().reindex(pxf.index, method="ffill")
        self.dias = pd.DatetimeIndex(np.sort(self.df["date"].unique()))
        self.res: dict = {}

    # -- utilidades ---------------------------------------------------------------------------------
    def _bt(self, col: str, fechas, lag=C.LAG_INVERTIBLE, coste=C.COSTE_PB, df=None, **kw) -> pd.DataFrame:
        d = (self.df if df is None else df).copy()
        d = d[d["date"].isin(fechas)]
        d["w"] = pesos(d, col, **kw)
        return backtest(d, self.pxf, fechas, "w", lag=lag, coste_pb=coste)

    def _spy_entre(self, bt: pd.DataFrame) -> pd.Series:
        return pd.Series(self.spy.reindex(bt["fin"]).to_numpy() / self.spy.reindex(bt["inicio"]).to_numpy() - 1, index=bt.index)

    # -- 1. rendimiento principal ----------------------------------------------------------------
    def principal(self) -> None:
        fs = fechas_semanales(self.dias)
        self.bt = {(s, lag): self._bt(s, fs, lag=lag) for s in ("score", "rev") for lag in (0, 1)}
        filas = []
        for (s, lag), b in self.bt.items():
            for coste in C.COSTES_SENSIBILIDAD:
                r = b["ret_bruto"] - b["rotacion"] * coste / 1e4
                filas.append({"estrategia": {"score": "ML semanal", "rev": "Reversión 1 semana"}[s], "lag": lag,
                              "coste_pb": coste, "rotacion_media": b["rotacion"].mean(), **metricas(r)})
        tabla = pd.DataFrame(filas)
        tabla.to_csv(C.RESULTADOS / "01_rendimiento.csv", index=False, float_format="%.4f")
        ml, rev = self.bt[("score", 1)], self.bt[("rev", 1)]
        anual = pd.DataFrame({
            "ML neto": ml["ret_neto"].groupby(ml.index.year).sum(),
            "Reversión neto": rev["ret_neto"].groupby(rev.index.year).sum(),
            "ML bruto lag 0": self.bt[("score", 0)]["ret_bruto"].groupby(ml.index.year).sum(),
            "Reversión bruto lag 0": self.bt[("rev", 0)]["ret_bruto"].groupby(ml.index.year).sum(),
        })
        anual.to_csv(C.RESULTADOS / "01_rendimiento_anual.csv", float_format="%.4f")
        pd.concat({"ml": ml, "rev": rev}, names=["estrategia"]).to_csv(C.RESULTADOS / "01_series_semanales.csv", float_format="%.6f")

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(_curva(ml["ret_neto"]), color=COLORES[0], label="ML semanal (neto, lag 1)")
        ax[0].plot(_curva(rev["ret_neto"]), color=COLORES[1], label="Reversión 1 semana (neto, lag 1)")
        ax[0].set_title(f"L/S neutral a sector, neto de {C.COSTE_PB:.0f} pb por lado")
        ax[1].plot(_curva(self.bt[("score", 0)]["ret_bruto"]), color=COLORES[0], label="ML semanal (bruto, lag 0)")
        ax[1].plot(_curva(self.bt[("rev", 0)]["ret_bruto"]), color=COLORES[1], label="Reversión (bruto, lag 0)")
        ax[1].set_title("Rendimiento teórico (bruto, ejecución al cierre de la señal)")
        for a in ax:
            a.legend(fontsize=8)
            a.axhline(100, color="0.6", lw=0.6)
        _guardar(fig, "01_rendimiento")
        self.res["principal"] = tabla[tabla.coste_pb.isin([0, C.COSTE_PB])].round(4).to_dict("records")

    # -- 2. regímenes: VIX y retorno del mercado -------------------------------------------------
    def regimenes(self) -> None:
        ml = self.bt[("score", 1)].copy()
        vix = self.df.groupby("date")["vix"].first().reindex(ml.index)
        ml["vix_q"] = pd.qcut(vix, 5, labels=["Lowest", "Low", "Middle", "High", "Highest"])
        ml["spy"] = self._spy_entre(ml)
        ml["spy_d"] = pd.qcut(ml["spy"], 10, labels=range(1, 11))
        por_vix = ml.groupby("vix_q", observed=True)["ret_neto"].mean() * C.SEMANAS_ANIO
        por_mkt = ml.groupby("spy_d", observed=True)["ret_neto"].mean() * C.SEMANAS_ANIO
        pd.DataFrame({"ret_anual": por_vix}).to_csv(C.RESULTADOS / "02_por_quintil_vix.csv", float_format="%.4f")
        pd.DataFrame({"ret_anual": por_mkt}).to_csv(C.RESULTADOS / "02_por_decil_spy.csv", float_format="%.4f")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        por_vix[::-1].plot.bar(ax=ax[0], color=COLORES[0])
        ax[0].set_title("ML neto: retorno anualizado por quintil del VIX")
        por_mkt.plot.bar(ax=ax[1], color=COLORES[0])
        ax[1].set_title("ML neto: retorno anualizado por decil del retorno semanal de SPY")
        for a in ax:
            a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            a.tick_params(axis="x", rotation=0)
        _guardar(fig, "02_regimenes")
        self.res["por_vix"] = por_vix.round(4).to_dict()
        self.res["por_decil_spy"] = {int(k): v for k, v in por_mkt.round(4).items()}

    # -- 3. SHAP -------------------------------------------------------------------------------
    def shap(self) -> None:
        nombres = [ETIQUETAS[f] for f in FEATURES]
        shap.summary_plot(self.extra["s_shap"], self.extra["x_shap"], feature_names=nombres, max_display=10, show=False)
        plt.title("SHAP del modelo semanal (último reentreno, oct-2021)")
        _guardar(plt.gcf(), "03_shap")
        imp = pd.DataFrame(self.extra["importancias"][:, 1:], columns=FEATURES,
                           index=pd.to_datetime(self.extra["importancias"][:, 0].astype("int64")))
        imp.to_csv(C.RESULTADOS / "03_shap_medio_abs.csv", float_format="%.5f")
        orden = imp.mean().sort_values(ascending=False)
        self.res["shap_orden"] = [ETIQUETAS[f] for f in orden.index[:10]]
        # Dirección: correlación entre el valor (rango) de la variable y su SHAP en la última muestra.
        x, s = self.extra["x_shap"], self.extra["s_shap"]
        self.res["shap_signo"] = {ETIQUETAS[f]: round(float(pd.Series(x[:, j]).corr(pd.Series(s[:, j]))), 2)
                                  for j, f in enumerate(FEATURES) if f in orden.index[:10]}

    # -- 4. exposiciones y atribución ----------------------------------------------------------
    def exposiciones_y_atribucion(self) -> None:
        fs = fechas_semanales(self.dias)
        d = self.df[self.df["date"].isin(fs)].copy()
        d["w"] = pesos(d, "score")
        cols = [c for c, _ in FACTORES.values()]
        z = _z(d, cols)
        for nombre, (c, signo) in FACTORES.items():
            d[nombre] = signo * z[c]
        expo = d[list(FACTORES)].mul(d["w"], axis=0).groupby(d["date"]).sum()
        expo.to_csv(C.RESULTADOS / "04_exposiciones.csv", float_format="%.4f")

        # Retorno de cada acción en el periodo de tenencia (lag 1) y regresión transversal por semana.
        ml = self.bt[("score", 1)]
        d["r"] = retorno_entre(self.pxf, d["date"].map(ml["inicio"]), d["date"].map(ml["fin"]), d["symbol"])
        filas = []
        for f, g in d.dropna(subset=["r"]).groupby("date"):
            X = np.column_stack([pd.get_dummies(g["sector"]).to_numpy(float), g[list(FACTORES)].to_numpy()])
            beta, *_ = np.linalg.lstsq(X, g["r"].to_numpy(), rcond=None)
            k = len(FACTORES)
            w = g["w"].to_numpy()
            contrib = {n: (w * g[n].to_numpy()).sum() * b for n, b in zip(FACTORES, beta[-k:])}
            contrib["Sectors"] = (w * (X[:, :-k] @ beta[:-k])).sum()
            contrib["total"] = (w * g["r"].to_numpy()).sum()
            contrib["Residual"] = contrib["total"] - sum(v for n, v in contrib.items() if n != "total")
            filas.append(pd.Series(contrib, name=f))
        atr = pd.DataFrame(filas)
        atr.to_csv(C.RESULTADOS / "04_atribucion_semanal.csv", float_format="%.6f")
        anual = (atr.drop(columns="total").mean() * C.SEMANAS_ANIO).sort_values()
        anual.to_frame("ret_anual").to_csv(C.RESULTADOS / "04_atribucion_anual.csv", float_format="%.4f")
        anio_fin = atr.index.year == atr.index.year.max()
        ultimo = atr[anio_fin].drop(columns="total").sum().sort_values()

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        rev_mom, resto = PANEL_B, PANEL_A
        for j, n in enumerate(resto):
            ax[0, 0].plot(expo[n].rolling(12).mean(), color=COLORES[j], label=n)
        for j, n in enumerate(rev_mom):
            ax[0, 1].plot(expo[n].rolling(12).mean(), color=COLORES[j], label=n)
        ax[0, 0].set_title("Exposiciones del ML semanal (media 12 semanas, z)")
        ax[0, 1].set_title("Reversión y momentum de precio/beneficios (media 12 semanas, z)")
        (anual * 100).plot.barh(ax=ax[1, 0], color=COLORES[0])
        ax[1, 0].set_title("Atribución (% anualizado, bruto lag 1, 2018-2021)")
        (ultimo * 100).plot.barh(ax=ax[1, 1], color=COLORES[1])
        ax[1, 1].set_title(f"Atribución {atr.index.year.max()} (%)")
        for a in ax[0]:
            a.legend(fontsize=8)
            a.axhline(0, color="0.6", lw=0.6)
            a.tick_params(axis="x", rotation=30)
        _guardar(fig, "04_exposiciones_atribucion")

        fig, ax = plt.subplots(figsize=(7, 4))
        for j, n in enumerate(SERIE_ATR):
            ax.plot(_curva(atr[n]), color=COLORES[j], label=n)
        ax.set_title("Serie temporal de la atribución (bruto, lag 1)")
        ax.legend(fontsize=8)
        _guardar(fig, "04_atribucion_serie")
        self.res["exposicion_media"] = expo.mean().round(3).to_dict()
        self.res["atribucion_anual"] = anual.round(4).to_dict()

    # -- 5. reversión: patas larga/corta, día de la semana y lags --------------------------------
    def estacionalidad_y_lags(self) -> None:
        fs = fechas_semanales(self.dias)
        uni = self.df[self.df["date"].isin(fs)].copy()
        uni["w"] = 1.0 / uni.groupby("date")["symbol"].transform("count")
        mercado = backtest(uni, self.pxf, fs, "w", lag=0)["ret_bruto"]
        rev0 = self.bt[("rev", 0)]
        patas = pd.DataFrame({"Larga vs mercado": rev0["larga"] - mercado, "Mercado vs corta": mercado - rev0["corta"]})
        dow = {s: pd.DataFrame({DIAS_SEMANA[k]: self._bt(s, fechas_semanales(self.dias, k), lag=0)["ret_bruto"]
                               for k in range(5)}) for s in ("rev", "score")}
        lags = pd.DataFrame({f"lag {k}": self._bt("score", fs, lag=k)["ret_bruto"] for k in range(5)})
        tabla = pd.concat({
            "patas_reversion": patas.mean() * C.SEMANAS_ANIO,
            "dia_reversion": dow["rev"].mean() * C.SEMANAS_ANIO,
            "dia_ml": dow["score"].mean() * C.SEMANAS_ANIO,
            "lags_ml_miercoles": lags.mean() * C.SEMANAS_ANIO,
        })
        tabla.to_frame("ret_anual_bruto").to_csv(C.RESULTADOS / "05_estacionalidad_lags.csv", float_format="%.4f")

        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for j, c in enumerate(patas):
            ax[0, 0].plot(_curva(patas[c]), color=COLORES[j], label=c)
        ax[0, 0].set_title("Patas de la reversión 1 semana (bruto, lag 0)")
        for j, c in enumerate(DIAS_SEMANA):
            ax[0, 1].plot(_curva(dow["rev"][c].dropna()), color=COLORES[j], label=c)
            ax[1, 0].plot(_curva(dow["score"][c].dropna()), color=COLORES[j], label=c)
        ax[0, 1].set_title("Reversión según el día de la señal (bruto, lag 0)")
        ax[1, 0].set_title("ML según el día de la señal (bruto, lag 0)")
        for j, c in enumerate(lags):
            ax[1, 1].plot(_curva(lags[c]), color=COLORES[j], label=c)
        ax[1, 1].set_yscale("log")
        ax[1, 1].set_title("ML con señal el miércoles, por lag de ejecución (bruto, log)")
        for a in ax.ravel():
            a.legend(fontsize=8)
        _guardar(fig, "05_estacionalidad_lags")
        self.res["estacionalidad_lags"] = {f"{a}|{b}": round(v, 4) for (a, b), v in tabla.items()}

    # -- 6. liquidez -----------------------------------------------------------------------------
    def liquidez(self) -> None:
        fs = fechas_semanales(self.dias)
        d = self.df[self.df["date"].isin(fs)].copy()
        d["q_adv"] = d.groupby("date")["log_adv_6m"].transform(lambda s: pd.qcut(s, 5, labels=False))
        etq = {4: "Top", 3: "2", 2: "3", 1: "4", 0: "Bottom"}
        filas = {}
        for q, n in etq.items():
            b = self._bt("score", fs, df=d[d["q_adv"] == q], por_sector=False)
            filas[n] = {"completo": b["ret_neto"].mean() * C.SEMANAS_ANIO,
                        "ultimo_anio": b.loc[b.index.year == b.index.year.max(), "ret_neto"].mean() * C.SEMANAS_ANIO}
        t = pd.DataFrame(filas).T
        t.to_csv(C.RESULTADOS / "06_por_quintil_adv.csv", float_format="%.4f")
        fig, ax = plt.subplots(figsize=(7, 4))
        t.plot.bar(ax=ax, color=COLORES[:2])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title("ML L/S neto por quintil de ADV a 6 meses (2018-2021 y 2021)")
        ax.tick_params(axis="x", rotation=0)
        _guardar(fig, "06_liquidez")
        self.res["por_quintil_adv"] = t.round(4).to_dict()

    # -- 7. dispersión ---------------------------------------------------------------------------
    def dispersion(self) -> None:
        p = self.pxf
        fs = fechas_semanales(p.index)
        r = (p.reindex(fs) / p.reindex(fs).shift(1) - 1).iloc[1:]
        dis = pd.DataFrame({"mediana_abs": r.abs().median(axis=1), "desv_transversal": r.std(axis=1)})
        dis.to_csv(C.RESULTADOS / "07_dispersion.csv", float_format="%.5f")
        ml = self.bt[("score", 1)]["ret_neto"]
        terc = pd.qcut(dis["desv_transversal"].shift(1).reindex(ml.index), 3, labels=["Baja", "Media", "Alta"])
        por_disp = ml.groupby(terc, observed=True).mean() * C.SEMANAS_ANIO
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(dis["mediana_abs"].rolling(52).mean() * 100, color=COLORES[0], label="mediana |r semanal|")
        ax[0].plot(dis["desv_transversal"].rolling(52).mean() * 100, color=COLORES[1], label="dispersión transversal")
        ax[0].set_title("S&P 500: variación de retornos semanales (%, media 52 semanas)")
        ax[0].legend(fontsize=8)
        por_disp.plot.bar(ax=ax[1], color=COLORES[0])
        ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax[1].set_title("ML neto por tercil de dispersión de la semana previa")
        ax[1].tick_params(axis="x", rotation=0)
        _guardar(fig, "07_dispersion")
        self.res["ml_por_dispersion"] = por_disp.round(4).to_dict()

    # -- 8. ciclo reversión-momentum por periodo de formación ------------------------------------
    def ciclo_momentum(self) -> None:
        ventanas = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 18), (18, 24), (24, 48)]
        fs = fechas_semanales(self.dias)
        fs = fs[fs >= self.pxf.index[48 * 5]]
        d = self.df[self.df["date"].isin(fs)][["date", "symbol", "sector"]].copy()
        out = {}
        for a, b in ventanas:
            sig = (self.pxf.shift(5 * a) / self.pxf.shift(5 * b) - 1).stack(future_stack=True)
            d["sig"] = sig.reindex(pd.MultiIndex.from_frame(d[["date", "symbol"]])).to_numpy()
            out["1" if a == 0 else f"{a}-{b}"] = self._bt("sig", fs, lag=0, df=d.dropna(subset=["sig"]), cuantil=0.1, por_sector=False)["ret_bruto"].mean() * C.SEMANAS_ANIO
        s = pd.Series(out)
        s.to_frame("ret_anual_bruto").to_csv(C.RESULTADOS / "08_ciclo_momentum.csv", float_format="%.4f")
        fig, ax = plt.subplots(figsize=(8, 4))
        s.plot.bar(ax=ax, color=[COLORES[1] if v < 0 else COLORES[0] for v in s])
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title("Momentum por periodo de formación (semanas), decil sup.-inf., 1 semana, 2018-2021")
        ax.tick_params(axis="x", rotation=0)
        _guardar(fig, "08_ciclo_momentum")
        self.res["ciclo_momentum"] = s.round(4).to_dict()

    # -- 9. el ML como filtro de ejecución en carteras factoriales mensuales -----------------------
    def filtro_factores(self) -> None:
        fm = fechas_mensuales(self.dias)
        d = self.df[self.df["date"].isin(fm)].copy()
        z = _z(d, ["mom_12m1m", "vol_6m", "beta_12m", "mom_3m1m"])
        d["Momentum"], d["Low Vol"], d["Low Beta"], d["3m Momentum"] = z["mom_12m1m"], -z["vol_6m"], -z["beta_12m"], z["mom_3m1m"]
        d["Multi-Factor"] = (d["Momentum"] + d["Low Vol"]) / 2
        factores = ["Multi-Factor", "Momentum", "3m Momentum", "Low Vol", "Low Beta"]
        if C.VERSION != "v2":
            # Los factores del paper (pág. 11); su multifactor combina valor, momentum y calidad.
            zf = _z(d, ["valor", "calidad", "crecimiento"])
            d["Value"], d["Quality"], d["Growth"] = zf["valor"], zf["calidad"], zf["crecimiento"]
            d["Multi-Factor"] = (d["Value"] + d["Momentum"] + d["Quality"]) / 3
            factores = ["Multi-Factor", "Value", "Momentum", "Quality", "Low Vol", "Growth"]
        d["ml_pct"] = d.groupby("date")["score"].rank(pct=True)
        filas, curvas = [], {}
        for fac in factores:
            d["w"] = pesos(d, fac)
            d["wf"] = self._filtrar(d)
            res = {}
            for col, n in (("w", "Sin filtro"), ("wf", "Con filtro ML")):
                b = backtest(d, self.pxf, fm, col, lag=C.LAG_INVERTIBLE, coste_pb=C.COSTE_PB)
                expo = (d[col] * d[fac]).groupby(d["date"]).sum().mean()
                res[n] = b["ret_neto"]
                filas.append({"factor": fac, "version": n, **metricas(b["ret_neto"], C.MESES_ANIO),
                              "exposicion_z": expo, "rotacion": b["rotacion"].mean()})
            curvas[fac] = res
        t = pd.DataFrame(filas)
        t.to_csv(C.RESULTADOS / "09_filtro_ml_factores.csv", index=False, float_format="%.4f")
        piv = t.pivot(index="factor", columns="version", values="ret_anual")
        expo = t.pivot(index="factor", columns="version", values="exposicion_z")
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        piv[["Sin filtro", "Con filtro ML"]].plot.bar(ax=ax[0, 0], color=COLORES[:2])
        ax[0, 0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax[0, 0].set_title("Retorno anualizado neto: el filtro ML en carteras mensuales")
        expo[["Sin filtro", "Con filtro ML"]].plot.bar(ax=ax[0, 1], color=COLORES[:2])
        ax[0, 1].set_title("Exposición al factor (z larga - corta)")
        mf = curvas["Multi-Factor"]
        ax[1, 0].plot(_curva(mf["Sin filtro"]), color=COLORES[0], label="Multi-Factor")
        ax[1, 0].plot(_curva(mf["Con filtro ML"]), color=COLORES[1], label="Multi-Factor con filtro ML")
        ax[1, 0].legend(fontsize=8)
        ax[1, 1].plot(100 * (1 + mf["Con filtro ML"]).cumprod() / (1 + mf["Sin filtro"]).cumprod(), color=COLORES[0])
        ax[1, 1].set_title("Rendimiento relativo (con filtro / sin filtro)")
        for a in ax[0]:
            a.tick_params(axis="x", rotation=0)
        _guardar(fig, "09_filtro_ml")
        self.res["filtro_ml"] = piv.round(4).to_dict()

    @staticmethod
    def _filtrar(d: pd.DataFrame) -> pd.Series:
        """No comprar lo que el ML ve con reversión a la baja ni vender lo que ve con reversión al alza:
        una alta larga nueva con ML en el quintil inferior no entra; una larga que sale con ML en el
        superior se queda (simétrico en la corta). Luego se reponderan las patas a pesos iguales."""
        wf = pd.Series(0.0, index=d.index)
        largos_prev, cortos_prev = set(), set()
        for f, g in d.groupby("date", sort=True):
            larga, corta = set(g.symbol[g.w > 0]), set(g.symbol[g.w < 0])
            ml = g.set_index("symbol")["ml_pct"]
            bajo, alto = set(ml.index[ml <= C.CUANTIL]), set(ml.index[ml > 1 - C.CUANTIL])
            vivos = set(ml.index)
            nl = (larga - ((larga - largos_prev) & bajo)) | ((largos_prev - larga) & alto & vivos)
            nc = (corta - ((corta - cortos_prev) & alto)) | ((cortos_prev - corta) & bajo & vivos)
            nl, nc = nl - nc, nc - nl
            wf[g.index[g.symbol.isin(nl)]] = 1.0 / max(len(nl), 1)
            wf[g.index[g.symbol.isin(nc)]] = -1.0 / max(len(nc), 1)
            largos_prev, cortos_prev = nl, nc
        return wf

    # -- 10. diversificación: beta móvil frente al mercado ---------------------------------------
    def beta_movil(self) -> None:
        ml = self.bt[("score", 1)]
        spy = self._spy_entre(ml)
        beta = ml["ret_neto"].rolling(26).cov(spy) / spy.rolling(26).var()
        beta.to_frame("beta_26s").to_csv(C.RESULTADOS / "10_beta_movil.csv", float_format="%.4f")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(beta, color=COLORES[0], label="ML semanal L/S")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title("Beta móvil a 6 meses frente a SPY")
        ax.legend(fontsize=8)
        _guardar(fig, "10_beta_movil")
        self.res["beta_total"] = round(float(np.cov(ml["ret_neto"], spy)[0, 1] / spy.var()), 3)
        self.res["corr_spy"] = round(float(ml["ret_neto"].corr(spy)), 3)

    # -- 11. atribución frente a factores académicos (Fama-French 5 + momentum; QMJ de AQR mensual) ------
    def atribucion_academica(self) -> None:
        import statsmodels.api as sm
        from linea1a import config as C1

        C1.usar_repo_ml4t()
        from data import load_aqr_factors, load_ff_factors

        ff = load_ff_factors("ff5", "daily").to_pandas().set_index("timestamp")
        ff = ff.join(load_ff_factors("mom", "daily").to_pandas().set_index("timestamp"))
        ff.index = pd.to_datetime(ff.index)
        cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
        acum = np.log1p(ff[cols]).cumsum()
        filas = []
        for s in ("score", "rev"):
            b = self.bt[(s, 1)]
            ini = acum.reindex(b["inicio"], method="ffill").to_numpy()
            fin = acum.reindex(b["fin"], method="ffill").to_numpy()
            X = pd.DataFrame(np.expm1(fin - ini), index=b.index, columns=cols)
            m = sm.OLS(b["ret_neto"], sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
            filas.append(self._fila_reg(m, s, "semanal FF5+MOM", C.SEMANAS_ANIO))
            # Mensual con QMJ (AQR sólo publica QMJ mensual en los datos de Jansen).
            qmj = load_aqr_factors("qmj").to_pandas().set_index("timestamp")["USA"]
            qmj.index = pd.to_datetime(qmj.index).to_period("M")
            mens = (1 + b.set_index("fin")["ret_neto"]).groupby(pd.to_datetime(b["fin"]).dt.to_period("M").to_numpy()).prod() - 1
            ffm = (1 + ff[cols]).groupby(ff.index.to_period("M")).prod() - 1
            Xm = ffm.join(qmj.rename("QMJ")).reindex(mens.index)
            mm = sm.OLS(mens, sm.add_constant(Xm), missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            filas.append(self._fila_reg(mm, s, "mensual FF5+MOM+QMJ", C.MESES_ANIO))
        t = pd.DataFrame(filas)
        t.to_csv(C.RESULTADOS / "11_atribucion_academica.csv", index=False, float_format="%.4f")
        self.res["atribucion_academica"] = t.round(3).to_dict("records")

    @staticmethod
    def _fila_reg(m, s: str, modelo: str, periodos: int) -> dict:
        fila = {"estrategia": {"score": "ML semanal", "rev": "Reversión 1 semana"}[s], "modelo": modelo,
                "alfa_anual": m.params["const"] * periodos, "t_alfa": m.tvalues["const"], "r2": m.rsquared, "n": int(m.nobs)}
        for k in m.params.index.drop("const"):
            fila[f"b_{k}"], fila[f"t_{k}"] = m.params[k], m.tvalues[k]
        return fila

    def todo(self) -> dict:
        for paso in (self.principal, self.regimenes, self.shap, self.exposiciones_y_atribucion,
                     self.estacionalidad_y_lags, self.liquidez, self.dispersion, self.ciclo_momentum,
                     self.filtro_factores, self.beta_movil, self.atribucion_academica):
            print(f"  {paso.__name__}")
            paso()
        with open(C.RESULTADOS / "resumen.json", "w", encoding="utf-8") as f:
            json.dump(self.res, f, indent=2, ensure_ascii=False, default=float)
        return self.res
