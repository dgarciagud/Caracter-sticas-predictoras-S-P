"""Fundamentales point-in-time de la SEC (XBRL `companyfacts`) para la v3.

Por qué no `data/equities/fundamentals/xbrl_download.py` de Jansen: usa la API Frames, que devuelve cada
dato tal como aparece en la *última* presentación que lo incluye, y fecha la fila por la última de ellas.
Para las cuentas de resultados eso suele ser el 10-Q del año siguiente (el comparativo), así que el dato
llega hasta 12 meses tarde: inservible para una sorpresa de beneficios. `companyfacts` trae cada hecho con
su fecha de presentación (`filed`); aquí se conserva la **primera** presentación de cada periodo, que es
lo que el mercado conocía en esa fecha.

Mapa ticker → CIK (`data_ref/cik_sp500.csv`, versionado; `construir_mapa_cik()` lo regenera): candidatos del
histórico git de github.com/datasets/s-and-p-500-companies (columna CIK) y de `company_tickers.json` de la
SEC; si ninguno es válido, búsqueda por nombre en EDGAR. Un CIK es válido si la empresa presentó 10-Q/10-K
al principio y al final del periodo de la acción en el índice, con activos ≥ 500 M$ (los tickers se reutilizan).
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import subprocess
import time

import numpy as np
import pandas as pd
import requests

from . import config as C

DATA_REF = C.RAIZ / "data_ref"
MAPA_CIK = DATA_REF / "cik_sp500.csv"
SEC_DIR = C.CACHE / "sec"
# La SEC pide un User-Agent descriptivo con contacto; se configura con SEC_USER_AGENT.
UA = os.environ.get("SEC_USER_AGENT", "caracteristicas-predictoras-sp research admin@example.org")
PAUSA = 0.15  # < 10 peticiones/s

DURACION = {
    "ni": ["NetIncomeLoss", "ProfitLoss"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "gp": ["GrossProfit"],
}
INSTANTE = {
    "activos": ["Assets"],
    "patrimonio": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "deuda_lp": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "fondo_comercio": ["Goodwill"],
    "intangibles": ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"],
}
FORMULARIOS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT"}


def _get(url: str, **kw) -> requests.Response:
    time.sleep(PAUSA)
    return requests.get(url, headers={"User-Agent": UA}, timeout=60, **kw)


# -- mapa ticker → CIK ---------------------------------------------------------------------------
def _nombre_limpio(n: str, palabras: int = 3) -> str:
    n = re.sub(r"\(.*?\)|-[A-Z]$|\b(inc|corp|corporation|co|company|ltd|plc|nv|holdings?|group|the)\b\.?", " ", n, flags=re.I)
    return " ".join(re.sub(r"[^A-Za-z0-9& ]", " ", n).split()[:palabras])


def _buscar_por_nombre(nombre: str) -> list[tuple[int, str]]:
    """CIK candidatos por nombre en EDGAR. Con una sola coincidencia, se exige que todas las palabras del
    nombre limpio estén en el nombre registrado o en los anteriores; con una lista (EDGAR no da nombres en
    ese modo) se devuelven todos y decide `_valido`."""
    limpio = _nombre_limpio(nombre)
    r = _get("https://www.sec.gov/cgi-bin/browse-edgar", params={
        "action": "getcompany", "company": limpio, "type": "10-Q", "dateb": "20220101",
        "owner": "include", "count": "20", "output": "atom"})
    ciks = [int(c) for c in re.findall(r"<cik>(\d+)</cik>", r.text)]
    if "<conformed-name>" not in r.text:
        return [(c, "edgar_lista") for c in ciks]
    nombres = re.sub(r"[^a-z0-9&| ]", " ", " | ".join(re.findall(r"<(?:conformed-name|name)>([^<]+)</", r.text)).lower())
    if ciks and all(w.lower() in nombres.split() for w in limpio.split()):
        return [(ciks[0], "edgar_nombre")]
    return []


def _candidatos(simbolos: list[str]) -> tuple[dict[str, list], dict[str, str]]:
    cand: dict[str, list] = {}
    nombre: dict[str, str] = {}
    repo = C.CACHE / "s-and-p-500-companies"
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", "--filter=blob:none",
                        "https://github.com/datasets/s-and-p-500-companies.git", str(repo)], check=True)
    commits = subprocess.run(["git", "-C", str(repo), "log", "--format=%h", "--", "data/constituents.csv"],
                             capture_output=True, text=True, check=True).stdout.split()
    for c in reversed(commits):  # del más antiguo al más reciente: primero el CIK más cercano a 2017-2021
        txt = subprocess.run(["git", "-C", str(repo), "show", f"{c}:data/constituents.csv"], capture_output=True, text=True).stdout
        try:
            d = pd.read_csv(io.StringIO(txt))
        except Exception:
            continue
        if "Symbol" not in d:
            continue
        col = next((x for x in ("Security", "Company", "Name") if x in d), None)
        for i, s in enumerate(d["Symbol"].astype(str).str.upper().str.strip()):
            if col:
                nombre.setdefault(s, d[col].iloc[i])
            if "CIK" in d and (m := re.search(r"\d+", str(d["CIK"].iloc[i]))):
                if (k := (int(m.group()), "datasets_sp500_git")) not in cand.setdefault(s, []):
                    cand[s].append(k)
    for v in _get("https://www.sec.gov/files/company_tickers.json").json().values():
        cand.setdefault(v["ticker"].upper(), []).append((int(v["cik_str"]), "sec_company_tickers"))
    out = {}
    for s in simbolos:
        n = s.replace(".", "-")
        lista = list(cand.get(n, [])) + [x for x in cand.get(s, []) if s != n]
        out[s] = lista
    return out, {s: nombre.get(s.replace(".", "-")) or nombre.get(s) for s in simbolos}


def _valido(cik: int, ini: pd.Timestamp, fin: pd.Timestamp) -> tuple[bool, str | None]:
    """La empresa presentó 10-Q/10-K al principio y al final del periodo en que la acción estaba en el
    índice (±150 días), con activos medianos ≥ 500 M$. Descarta SPAC, empresas nuevas que heredan un
    ticker y escisiones (p. ej. HCP → Quality Care Properties); mejor sin dato que con otra empresa."""
    cf = _companyfacts(cik) or {}
    h = _hechos(cf, ["Assets"], "USD")
    if h.empty:
        return False, cf.get("entityName")
    margen = pd.Timedelta(days=150)
    en_rango = h[(h["filed"] >= ini - margen) & (h["filed"] <= fin + margen)]
    ok = (len(en_rango) > 0 and en_rango["filed"].min() <= ini + margen and en_rango["filed"].max() >= fin - margen
          and en_rango["val"].median() >= 5e8)
    return bool(ok), cf.get("entityName")


def construir_mapa_cik(px: pd.DataFrame) -> pd.DataFrame:
    """Para cada símbolo, el primer CIK candidato válido: históricos (datasets), actuales (SEC) y, si
    ninguno vale, búsqueda por nombre en EDGAR."""
    cand, nombre = _candidatos(list(px.columns))
    filas = []
    for s in px.columns:
        ini, fin = px[s].first_valid_index(), px[s].last_valid_index()
        elegido = None
        for k, fuente in cand[s]:
            ok, entidad = _valido(k, ini, fin)
            if ok:
                elegido = (k, fuente, entidad)
                break
        if elegido is None and nombre[s]:
            for k, fuente in _buscar_por_nombre(nombre[s]):
                ok, entidad = _valido(k, ini, fin)
                if ok:
                    elegido = (k, fuente, entidad)
                    break
        filas.append({"symbol": s, "nombre": nombre[s], "cik": elegido[0] if elegido else None,
                      "fuente": elegido[1] if elegido else None, "entidad": elegido[2] if elegido else None,
                      "candidatos": ";".join(str(k) for k, _ in cand[s])})
    out = pd.DataFrame(filas)
    DATA_REF.mkdir(exist_ok=True)
    out.to_csv(MAPA_CIK, index=False)
    return out


# -- descarga y hechos PIT --------------------------------------------------------------------
def _companyfacts(cik: int) -> dict | None:
    SEC_DIR.mkdir(parents=True, exist_ok=True)
    ruta = SEC_DIR / f"CIK{cik:010d}.json.gz"
    if not ruta.exists():
        r = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
        if r.status_code != 200:
            return None
        ruta.write_bytes(gzip.compress(r.content))
    return json.loads(gzip.decompress(ruta.read_bytes()))


def _hechos(cf: dict, conceptos: list[str], unidad: str) -> pd.DataFrame:
    """Primer concepto disponible de la lista; un hecho por (inicio, fin): el primero presentado."""
    gaap = cf.get("facts", {}).get("us-gaap", {})
    partes = []
    for k in conceptos:
        u = gaap.get(k, {}).get("units", {}).get(unidad)
        if u:
            partes.append(pd.DataFrame(u).assign(prioridad=len(partes)))
    if not partes:
        return pd.DataFrame()
    d = pd.concat(partes)
    d = d[d["form"].isin(FORMULARIOS)].copy()
    if "start" not in d:
        d["start"] = np.nan
    d["filed"] = pd.to_datetime(d["filed"])
    d["end"] = pd.to_datetime(d["end"])
    d["start"] = pd.to_datetime(d["start"])
    d = d.sort_values(["filed", "prioridad"]).drop_duplicates(["start", "end"], keep="first")
    return d[["start", "end", "val", "filed"]]


def _trimestres(d: pd.DataFrame) -> pd.DataFrame:
    """Valores trimestrales de un concepto de flujo: directos (80-100 días) o diferencia de acumulados
    (YTD) con el mismo inicio de ejercicio; la fecha conocida es la de la presentación más tardía."""
    if d.empty:
        return d
    d = d.dropna(subset=["start"]).copy()
    d["dias"] = (d["end"] - d["start"]).dt.days
    directos = d[d["dias"].between(80, 100)][["end", "val", "filed"]]
    deriv = []
    for _, g in d[d["dias"] > 100].groupby("start"):
        g = g.sort_values("end")
        previo = d[(d["start"] == g["start"].iloc[0]) & (d["end"] < g["end"].iloc[0])].sort_values("end")
        cadena = pd.concat([previo.tail(1), g])
        for a, b in zip(cadena.iloc[:-1].itertuples(), cadena.iloc[1:].itertuples()):
            if 80 <= (b.end - a.end).days <= 100:
                deriv.append({"end": b.end, "val": b.val - a.val, "filed": max(a.filed, b.filed)})
    t = pd.concat([directos, pd.DataFrame(deriv)]).sort_values("filed").drop_duplicates("end", keep="first")
    return t.sort_values("end").reset_index(drop=True)


def _ttm(t: pd.DataFrame) -> pd.DataFrame:
    """Suma de los 4 últimos trimestres consecutivos (el último terminado 330-400 días tras el primero)."""
    if len(t) < 4:
        return pd.DataFrame(columns=["end", "val", "filed"])
    v = t["val"].rolling(4).sum()
    ok = (t["end"] - t["end"].shift(3)).dt.days.between(250, 300)
    f = pd.Series([t["filed"].iloc[max(0, i - 3): i + 1].max() for i in range(len(t))])
    return pd.DataFrame({"end": t["end"], "val": v.where(ok), "filed": f}).dropna()


def eventos_empresa(cik: int) -> pd.DataFrame:
    """Una fila por fecha de presentación con los últimos valores conocidos de cada métrica."""
    cf = _companyfacts(cik)
    if cf is None:
        return pd.DataFrame()
    series = {}
    for k, cs in DURACION.items():
        q = _trimestres(_hechos(cf, cs, "USD/shares" if k == "eps" else "USD"))
        if q.empty:
            continue
        series[f"{k}_q"] = q
        if k != "eps":
            series[f"{k}_ttm"] = _ttm(q)
        else:
            series["eps_ttm"] = _ttm(q)
            # SUE: cambio interanual del BPA trimestral / su desviación en los 8 trimestres previos.
            q = q.copy()
            lag = q.set_index("end")["val"]
            q["dif"] = [v - _hace_un_anio(lag, e) for e, v in zip(q["end"], q["val"])]
            q["sue_val"] = q["dif"] / q["dif"].shift(1).rolling(8, min_periods=4).std()
            series["sue"] = q[["end", "sue_val", "filed"]].rename(columns={"sue_val": "val"}).dropna()
    for k, cs in INSTANTE.items():
        h = _hechos(cf, cs, "USD")
        if not h.empty:
            series[k] = h[["end", "val", "filed"]].sort_values("end")
    dei = cf.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares")
    if dei:
        s = pd.DataFrame(dei)
        s = s[s["form"].isin(FORMULARIOS)]
        s["filed"], s["end"] = pd.to_datetime(s["filed"]), pd.to_datetime(s["end"])
        series["acciones"] = s.sort_values("filed").drop_duplicates("end")[["end", "val", "filed"]]
    if "activos" in series:
        a = series["activos"].set_index("end")["val"]
        g = series["activos"].copy()
        g["val"] = [v / previo - 1 if (previo := _hace_un_anio(a, e)) > 0 else np.nan for e, v in zip(g["end"], g["val"])]
        series["crec_activos"] = g.dropna()
    if "eps_ttm" in series:
        e = series["eps_ttm"].set_index("end")["val"]
        g = series["eps_ttm"].copy()
        g["val"] = [v - _hace_un_anio(e, f) for f, v in zip(g["end"], g["val"])]
        series["dif_eps_ttm"] = g.dropna()
    fechas = sorted({f for s in series.values() for f in s["filed"]})
    if not fechas:
        return pd.DataFrame()
    out = pd.DataFrame(index=pd.DatetimeIndex(fechas, name="filed"))
    for k, s in series.items():
        # valor del periodo más reciente ya presentado en cada fecha
        s = s.sort_values(["filed", "end"])
        s = s[s["end"] == s["end"].cummax()]
        g = s.groupby("filed").last()
        val = g["val"].reindex(out.index).ffill()
        fin = pd.to_datetime(g["end"].reindex(out.index).ffill())
        # un concepto que la empresa deja de usar no debe quedarse congelado
        edad = (out.index.to_series() - fin).dt.days
        out[k] = val.where(edad <= 400)
    out["cik"] = cik
    return out.reset_index()


def _hace_un_anio(serie: pd.Series, fecha: pd.Timestamp) -> float:
    """Valor de `serie` en el periodo que terminó ~1 año antes de `fecha` (±20 días)."""
    objetivo = fecha - pd.DateOffset(years=1)
    cand = serie[(serie.index >= objetivo - pd.Timedelta(days=20)) & (serie.index <= objetivo + pd.Timedelta(days=20))]
    return float(cand.iloc[-1]) if len(cand) else np.nan


def panel_fundamental(px: pd.DataFrame) -> pd.DataFrame:
    """Eventos (fecha de presentación × símbolo) de todas las empresas del universo, cacheado."""
    ruta = C.CACHE / "fundamentales_eventos.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta)
    mapa = pd.read_csv(MAPA_CIK) if MAPA_CIK.exists() else construir_mapa_cik(px)
    mapa = mapa.dropna(subset=["cik"])
    partes = []
    for i, (s, c) in enumerate(zip(mapa["symbol"], mapa["cik"].astype(int)), 1):
        ev = eventos_empresa(c)
        if not ev.empty:
            partes.append(ev.assign(symbol=s))
        if i % 50 == 0:
            print(f"    SEC {i}/{len(mapa)}")
    out = pd.concat(partes, ignore_index=True)
    out.to_parquet(ruta)
    return out


def dividendos() -> pd.DataFrame:
    """Dividendo en efectivo por acción en su fecha ex: (1 - adjustment_factor) × cierre previo (AlgoSeek)."""
    from linea1a import config as C1

    C1.usar_repo_ml4t()
    from data import load_sp500_daily_bars

    b = load_sp500_daily_bars().to_pandas()
    if "date" not in b:
        b["date"] = b["timestamp"]
    b = b.sort_values(["symbol", "date"])
    b["prev"] = b.groupby("symbol")["close"].shift(1)
    d = b[b["adjustment_reason"] == "CashDiv"].copy()
    d["div"] = (1 - d["adjustment_factor"]) * d["prev"]
    d["date"] = pd.to_datetime(d["date"]).astype("datetime64[ns]")
    return d.pivot_table(index="date", columns="symbol", values="div", aggfunc="sum")

