"""Orquestador de la línea 1A según la guía del TFM (pre-registro preregistro/preregistro_1A_guia.yaml).

Uso: python scripts/run_guia_1a.py [datos|fase1|fase2|modelos|shap|regimen|sobreajuste|todo]
Cada paso guarda sus tablas en resultados/guia1a/<version>/ y el siguiente las lee.
El holdout (2013-2026) no se toca aquí: ver scripts/holdout_guia_1a.py.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guia1a import config as C  # noqa: E402
from guia1a import datos, estad, fase1, fase2, figuras, variables  # noqa: E402
from guia1a.estrategia import backtest, dsr, matriz_registro, metricas, pbo_cscv, referencias, registrar, tramos  # noqa: E402
from guia1a.modelos import ESCALERA, GBM, RidgeTS  # noqa: E402
from guia1a.oos import cpcv_pred, media_historica, puerta, wf_pred  # noqa: E402
from guia1a.shap_estab import estabilidad, metricas_bloque, shap_por_bloque  # noqa: E402

R = C.RESULTADOS
ESTADO = R / "estado.pkl"


def guardar(**kw):
    est = pickle.loads(ESTADO.read_bytes()) if ESTADO.exists() else {}
    est.update(kw)
    ESTADO.write_bytes(pickle.dumps(est))


def estado() -> dict:
    return pickle.loads(ESTADO.read_bytes())


def js(obj, nombre):
    def conv(v):
        if isinstance(v, dict):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, (np.floating, float)):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v

    (R / nombre).write_text(json.dumps(conv(obj), ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ datos
def paso_datos():
    d = datos.cargar(C.DESC_FIN)
    P = variables.panel(d)
    cols = list(C.CANDIDATAS) + list(C.REGIMENES) + list(C.INTERACCIONES)
    D = P.loc[C.DESC_INI : C.DESC_FIN]
    ok = D[cols + ["y"]].notna().all(axis=1)
    idx = D.index[ok]
    # La muestra válida debe ser un tramo contiguo del calendario del índice.
    pos = d.index.get_indexer(idx)
    assert np.all(np.diff(pos) == 1), "la muestra válida no es contigua"
    r = np.log(d["close"]).diff()
    dia = pd.DataFrame({"ret_d": d["close"].pct_change(), "rf_d": d["DFF"].ffill() / 100 / 252}).loc[idx[0] :]
    sigma_d = r.ewm(span=C.SPAN_SIGMA, min_periods=C.SPAN_SIGMA).std()
    mom = np.log(d["close"]).shift(21) - np.log(d["close"]).shift(252)
    w = variables.unicidad(len(idx))
    n_eff = len(idx) / C.H
    info = dict(
        primera=str(idx[0].date()), ultima=str(idx[-1].date()), sesiones=len(idx), n_eff=n_eff,
        t_ic005=0.05 * np.sqrt(n_eff), se_auc=np.sqrt(1 / (3 * n_eff)), unicidad_media=float(w.mean()),
    )
    js(info, "00_muestra.json")
    guardar(d=d, P=P, idx=idx, dia=dia, sigma_d=sigma_d, mom=mom, w=w)
    print(info)


def _refs(est):
    P, idx, dia = est["P"], est["idx"], est["dia"]
    mh = media_historica(P.loc[idx, "y"])
    pesos = referencias(dia, est["mom"].reindex(dia.index), est["sigma_d"].reindex(dia.index), mh.reindex(dia.index))
    return {k: backtest(v.where(v.index <= idx[-1]), dia) for k, v in pesos.items()}


# ------------------------------------------------------------------ fase 1
def paso_fase1():
    est = estado()
    P, idx = est["P"], est["idx"]
    refs = _refs(est)
    for k, bt in refs.items():
        registrar(f"REF_{k}", "referencia", k, "referencia", metricas(bt), bt["neto"] - bt["rf"])
    t1 = fase1.ejecutar(P.loc[idx], est["d"], est["dia"], refs["buy_hold"], refs["tsmom_12_1"])
    t1.to_csv(R / "01_fase1_univariante.csv", index=False)
    print(t1[["variable", "signo_esperado", "t_nw", "p95_nula", "p_nula", "consistencia_bloques", "t_placebo", "pasa_fase1", "bh_10"]].round(3).to_string())


# ------------------------------------------------------------------ fase 2
def paso_fase2():
    est = estado()
    P, idx, w = est["P"], est["idx"], est["w"]
    Pd = P.loc[idx]
    grupos, info = fase2.clusters(Pd)
    Xall, signo, alin = fase2.compuestos(P, grupos, info["rho"])
    X = Xall.loc[idx]
    y = Pd["y"].to_numpy()
    t1 = pd.read_csv(R / "01_fase1_univariante.csv").set_index("variable")
    filas = []
    for g, miembros in grupos.items():
        tc = fase2.t_simple(X[g].to_numpy(), y)
        mejor = max(miembros, key=lambda m: abs(t1.loc[m, "t_nw"]))
        filas.append(dict(cluster=g, miembros=" ".join(miembros), n=len(miembros), signo=signo[g],
                          t_compuesto=tc, mejor_miembro=mejor, t_mejor=t1.loc[mejor, "t_nw"],
                          compuesto_ge_mejor=abs(tc) >= abs(t1.loc[mejor, "t_nw"])))
    tcl = pd.DataFrame(filas)
    tcl["vif"] = fase2.vif(X).reindex(tcl.cluster).to_numpy()
    tcl.to_csv(R / "02_clusters.csv", index=False)
    info["rho"].to_csv(R / "02_spearman.csv")
    js(dict(k=info["k"], modo=info.get("modo", "silueta"), k_silueta=info.get("k_silueta", info["k"]), silueta=info["silueta"], siluetas=info["siluetas"], alineacion=alin), "02_clusters.json")
    # Clustered MDA: compuestos + 1 sombra por compuesto.
    Sh = fase2.sombras(X, 1, C.SEMILLA + 1)
    XM = pd.concat([X, Sh], axis=1)
    n_perm = int(C.PREREG["clustered_mda"]["permutaciones"])
    mda_g = fase2.clustered_mda(XM, y, w, GBM, n_perm, C.SEMILLA)
    mda_r = fase2.clustered_mda(XM, y, w, RidgeTS, n_perm, C.SEMILLA)
    mda_g.to_csv(R / "02_mda_gbm_particiones.csv")
    mda_r.to_csv(R / "02_mda_ridge_particiones.csv")
    filas = []
    sombras_cols = list(Sh.columns)
    p95g = np.percentile(mda_g[sombras_cols].mean(), 95)
    p95r = np.percentile(mda_r[sombras_cols].mean(), 95)
    for g in X.columns:
        filas.append(dict(cluster=g, mda_gbm=mda_g[g].mean(), mda_gbm_pos=float((mda_g[g] > 0).mean()), p95_sombras_gbm=p95g,
                          sobrevive=bool(mda_g[g].mean() > p95g), mda_ridge=mda_r[g].mean(), p95_sombras_ridge=p95r,
                          sobrevive_ridge=bool(mda_r[g].mean() > p95r)))
    tm = pd.DataFrame(filas)
    tm.to_csv(R / "02_mda.csv", index=False)
    pd.DataFrame({"gbm": mda_g[sombras_cols].mean(), "ridge": mda_r[sombras_cols].mean()}).to_csv(R / "02_mda_sombras.csv")
    superv = tm.loc[tm.sobrevive, "cluster"].tolist()
    guardar(grupos=grupos, Xall=Xall, signo=signo, superv=superv, info_cl=info)
    figuras.dendrograma(info, R / "figuras" / "02_dendrograma.png")
    figuras.mda(tm, mda_g[sombras_cols].mean(), R / "figuras" / "02_mda.png")
    print(tcl.round(3).to_string())
    print(tm.round(5).to_string())
    print("supervivientes:", superv)


# ------------------------------------------------------------------ escalera, puerta y walk-forward
def _conjunto(est):
    superv = est["superv"]
    exploratorio = len(superv) == 0
    cols = superv if superv else list(est["grupos"])
    return cols, exploratorio


def _eval_wf(nombre, desc, pred: pd.Series, est, refs, extra):
    idx = est["idx"]
    y = est["P"].loc[pred.index, "y"].to_numpy()
    bench = media_historica(est["P"].loc[idx, "y"]).reindex(pred.index).to_numpy()
    ok = np.isfinite(bench)
    senal = tramos((pred > 0).astype(float))
    bt = backtest(senal.reindex(est["dia"].index).where(lambda s: s.index <= idx[-1]), est["dia"])
    m = dict(r2_oos=estad.r2_oos(y[ok], pred.to_numpy()[ok], bench[ok]), cw_t=estad.clark_west(y[ok], bench[ok], pred.to_numpy()[ok]),
             **dict(zip(("ic", "t_ic"), estad.ic_t(pred.to_numpy(), y))),
             **metricas(bt, refs["buy_hold"], refs["tsmom_12_1"]))
    m["dif_sharpe_volesc"], m["t_dif_sharpe_volesc"] = estad.dif_sharpe_hac(
        (bt["neto"] - bt["rf"]), (refs["bh_vol_escalado"]["neto"] - refs["bh_vol_escalado"]["rf"]).reindex(bt.index))
    registrar(nombre, extra.pop("fase", "fase4"), desc, "walk_forward", m, bt["neto"] - bt["rf"], **extra)
    return m, bt


def paso_modelos():
    est = estado()
    P, idx, w = est["P"], est["idx"], est["w"]
    y = P.loc[idx, "y"].to_numpy()
    cols, explor = _conjunto(est)
    X = est["Xall"].loc[idx]
    refs = _refs(est)
    res_cpcv, res_wf, preds_wf, curvas = {}, {}, {}, {}
    variantes = {"sel": cols} if explor else {"sel": cols, "todos": list(est["grupos"])}
    for vn, vc in variantes.items():
        for nombre, fab in ESCALERA.items():
            o = cpcv_pred(X[vc], y, w, fab)
            g = puerta(o["agregada"], y, o["bench_agregado"])
            g["cw_t"] = estad.clark_west(y, o["bench_agregado"], o["agregada"])
            # Sharpe por trayectoria CPCV (estrategia largo/neutral).
            shs = []
            for j in range(o["tray"].shape[1]):
                s = pd.Series(o["tray"][:, j], index=idx)
                bt = backtest(tramos((s > 0).astype(float)).reindex(est["dia"].index).where(lambda q: q.index <= idx[-1]), est["dia"])
                shs.append(metricas(bt)["sharpe_neto"])
            g["sharpe_trayectorias"] = shs
            res_cpcv[f"{vn}|{nombre}"] = g
            curvas[f"{vn}|{nombre}"] = o["agregada"]
            p = wf_pred(X[vc], y, w, fab)
            preds_wf[f"{vn}|{nombre}"] = p
            m, bt = _eval_wf(f"WF_{vn}_{nombre}", f"{nombre} sobre {vn} ({len(vc)} compuestos)", p, est, refs,
                             dict(variables=vc, modelo=nombre))
            res_wf[f"{vn}|{nombre}"] = m
            print(vn, nombre, {k: round(v, 3) for k, v in g.items() if not isinstance(v, list)}, "| WF", round(m["r2_oos"], 4), round(m["sharpe_neto"], 3))
    # Comparación entre escalones (pérdida cuadrática, t NW) en CPCV agregado y WF.
    nombres = list(ESCALERA)
    comp = []
    for vn in variantes:
        for a, b in zip(nombres[:-1], nombres[1:]):
            dc, tc = estad.dm_t(y, curvas[f"{vn}|{a}"], curvas[f"{vn}|{b}"])
            pa, pb = preds_wf[f"{vn}|{a}"], preds_wf[f"{vn}|{b}"]
            yy = P.loc[pa.index, "y"].to_numpy()
            dw, tw = estad.dm_t(yy, pa.to_numpy(), pb.to_numpy())
            comp.append(dict(variante=vn, de=a, a=b, dif_cpcv=dc, t_cpcv=tc, dif_wf=dw, t_wf=tw, sube=bool(tc >= 2 and tw >= 2)))
    # g2: hiperparámetros elegidos por la validación interna y variantes fijas de min_child_samples.
    if C.GBM_G2:
        from guia1a.cv import cpcv as _cpcv, walk_forward as _wf

        Xs = X[cols].to_numpy()
        elec = []
        for nombre_v, parts in (("walk_forward", [(str(a), tr, te) for a, tr, te in _wf(idx)]),
                                ("cpcv", [(str(c), tr, te) for c, tr, te in _cpcv(len(idx))])):
            for et, tr, te in parts:
                m_ = GBM().fit(Xs[tr], y[tr], w[tr])
                elec.append(dict(validacion=nombre_v, particion=et, min_child_samples=m_.min_child, n_arboles=m_.n_arboles))
        pd.DataFrame(elec).to_csv(R / "04_gbm_hiperparametros_elegidos.csv", index=False)
        fijas = []
        for mcs in C.GBM_G2["min_child_samples_rejilla"]:
            fab = lambda mcs=mcs: GBM(rejilla=[mcs])  # noqa: E731
            o = cpcv_pred(X[cols], y, w, fab)
            g = puerta(o["agregada"], y, o["bench_agregado"])
            p = wf_pred(X[cols], y, w, fab)
            m, _ = _eval_wf(f"WF_sel_gbm_mcs{mcs}", f"GBM con early stopping y min_child_samples={mcs} fijo", p, est, refs,
                            dict(variables=cols, modelo="3_gbm", fase="fase4_g2"))
            fijas.append(dict(min_child_samples=mcs, cpcv_ic=g["ic"], cpcv_t_ic=g["t_ic"], cpcv_r2_oos=g["r2_oos"],
                              wf_r2_oos=m["r2_oos"], wf_cw_t=m["cw_t"], wf_sharpe=m["sharpe_neto"],
                              wf_dif_sharpe_bh=m["dif_sharpe_bh"], wf_t_dif=m["t_dif_sharpe_bh"]))
            print("mcs", mcs, {k: round(v, 4) for k, v in fijas[-1].items()})
        pd.DataFrame(fijas).to_csv(R / "04_gbm_rejilla_min_child.csv", index=False)
    tcomp = pd.DataFrame(comp)
    tcomp.to_csv(R / "04_escalera_comparacion.csv", index=False)
    elegido = nombres[0]
    for r_ in comp:
        if r_["variante"] == "sel" and r_["de"] == elegido and r_["sube"]:
            elegido = r_["a"]
    # Puerta de skill: GBM del conjunto seleccionado frente a 20 modelos con todo sustituido por sombras.
    g_real = res_cpcv["sel|3_gbm"]
    nul = []
    for r_ in range(C.N_NULA_SKILL):
        Sh = fase2.sombras(X[cols], 1, C.SEMILLA + 1000 + r_)
        o = cpcv_pred(Sh, y, w, GBM)
        nul.append(estad.ic_t(o["agregada"], y)[0])
    gate = dict(ic=g_real["ic"], t_ic=g_real["t_ic"], r2_oos=g_real["r2_oos"], ic_p95_sombras=float(np.percentile(nul, 95)),
                ic_sombras=nul)
    gate["pasa"] = bool(gate["t_ic"] >= 2 and gate["r2_oos"] > 0 and gate["ic"] > gate["ic_p95_sombras"])
    gate["ridge"] = dict(ic=res_cpcv["sel|2_ridge"]["ic"], t_ic=res_cpcv["sel|2_ridge"]["t_ic"], r2_oos=res_cpcv["sel|2_ridge"]["r2_oos"])
    js(dict(cpcv=res_cpcv, walk_forward=res_wf, elegido=elegido, puerta=gate, exploratorio=explor), "04_modelos.json")
    pd.DataFrame(res_wf).T.to_csv(R / "04_walk_forward.csv")
    pd.DataFrame({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in res_cpcv.items()}).T.to_csv(R / "04_cpcv.csv")
    guardar(elegido=elegido, gate=gate, preds_wf=preds_wf)
    figuras.curvas_capital({"Modelo elegido (" + elegido + ")": _bt_de(preds_wf[f"sel|{elegido}"], est),
                            "GBM": _bt_de(preds_wf["sel|3_gbm"], est),
                            "Buy & hold": refs["buy_hold"], "TSMOM 12-1": refs["tsmom_12_1"]},
                           R / "figuras" / "04_curvas_walk_forward.png", "Walk-forward en descubrimiento 1997-2012 (neto de 2 pb)")
    figuras.sharpe_trayectorias(res_cpcv, R / "figuras" / "04_sharpe_trayectorias_cpcv.png")
    print(tcomp.round(3).to_string())
    print("elegido:", elegido, "| puerta:", {k: v for k, v in gate.items() if k != "ic_sombras"})


def _bt_de(pred, est):
    idx = est["idx"]
    return backtest(tramos((pred > 0).astype(float)).reindex(est["dia"].index).where(lambda s: s.index <= idx[-1]), est["dia"])


# ------------------------------------------------------------------ estabilidad SHAP
def paso_shap():
    est = estado()
    P, idx, w = est["P"], est["idx"], est["w"]
    y = P.loc[idx, "y"].to_numpy()
    cols, explor = _conjunto(est)
    X = est["Xall"].loc[idx, cols]
    Sh = fase2.sombras(X, C.N_SOMBRAS_SHAP, C.SEMILLA + 2)
    XS = pd.concat([X, Sh], axis=1)
    bloques = shap_por_bloque(XS, y, w)
    pb = [metricas_bloque(XS.loc[S.index], S) for S in bloques]
    filas = {c: estabilidad(pb, c) for c in XS.columns}
    T = pd.DataFrame(filas).T
    T["sombra"] = T.index.str.contains("__sh")
    T["binaria"] = [pb[0][c]["binaria"] for c in T.index]
    out = []
    for c in cols:
        sh = T[T.sombra & (T.binaria == T.loc[c, "binaria"])]
        if not len(sh):
            sh = T[T.sombra]
        s = est["signo"][c]
        r = T.loc[c]
        signo_ok = r.signo_constante and (r.signo_dominante == s if s else True)
        p95 = {k: float(np.percentile(sh[k].astype(float), 95)) for k in ("monotonia_min", "forma_media", "relevancia")}
        umbral_forma = 1.0 if r.binaria else 0.7
        out.append(dict(
            cluster=c, signo_esperado=s, signos_bloques=r.signo_bloques, signo_ok=bool(signo_ok),
            monotonia_min=r.monotonia_min, p95_monotonia_sombras=p95["monotonia_min"],
            forma_media=r.forma_media, p95_forma_sombras=p95["forma_media"],
            relevancia=r.relevancia, p95_relevancia_sombras=p95["relevancia"],
            monotonia_ok=bool(r.monotonia_min >= 0.7 and r.monotonia_min > p95["monotonia_min"]),
            forma_ok=bool(r.forma_media >= umbral_forma and r.forma_media > p95["forma_media"]),
            relevancia_ok=bool(r.relevancia > p95["relevancia"]),
        ))
    TS = pd.DataFrame(out)
    TS["estable_shap"] = TS.signo_ok & TS.monotonia_ok & TS.forma_ok & TS.relevancia_ok
    # Ridge: estabilidad del signo del coeficiente en las 15 particiones.
    o = cpcv_pred(X, y, w, RidgeTS, devolver_modelos=True)
    coefs = np.array([m.coef for m in o["modelos"]])
    TS["ridge_signos_pos"] = (coefs > 0).sum(0)
    TS["ridge_signo_estable"] = [bool(((coefs[:, j] > 0).all() and (est["signo"][c] >= 0)) or ((coefs[:, j] < 0).all() and (est["signo"][c] <= 0)))
                                 for j, c in enumerate(cols)]
    TS.to_csv(R / "05_estabilidad_shap.csv", index=False)
    T.drop(columns=["signo_bloques"]).to_csv(R / "05_estabilidad_shap_todas.csv")
    guardar(shap_bloques=bloques, shap_metricas=pb, XS=XS)
    figuras.dependencia_shap(XS, bloques, cols, R / "figuras" / "05_dependencia_shap.png", explor)
    print(TS.drop(columns=["signos_bloques"]).round(3).to_string())


# ------------------------------------------------------------------ regímenes (A.12)
def paso_regimen():
    est = estado()
    P, idx, w = est["P"], est["idx"], est["w"]
    y = P.loc[idx, "y"].to_numpy()
    cols, _ = _conjunto(est)
    X = est["Xall"].loc[idx, cols]
    Rg = P.loc[idx, list(C.REGIMENES)]
    It = P.loc[idx, list(C.INTERACCIONES)]
    refs = _refs(est)
    M = {"M1": X, "M2": pd.concat([X, Rg], axis=1), "M3": pd.concat([X, Rg, It], axis=1)}
    res, preds_c, preds_w = {}, {}, {}
    for k, XX in M.items():
        o = cpcv_pred(XX, y, w, RidgeTS)
        preds_c[k] = o
        p = wf_pred(XX, y, w, RidgeTS)
        preds_w[k] = p
        m, _ = _eval_wf(f"WF_reg_{k}_ridge", f"ridge {k}", p, est, refs, dict(variables=list(XX.columns), modelo="2_ridge", fase="fase3"))
        res[k] = dict(cpcv=puerta(o["agregada"], y, o["bench_agregado"]), wf=m)
    # GBM: M2 aditivo frente a M3 con la interacción permitida solo en los pares de la hipótesis.
    XG = pd.concat([X, Rg, P.loc[idx, [s["x"] for s in C.INTERACCIONES.values() if s["x"] not in X]]], axis=1)
    XG = XG.loc[:, ~XG.columns.duplicated()]
    cc = list(XG.columns)
    aditivo = [[i] for i in range(len(cc))]
    pares = aditivo + [[cc.index(s["x"]), cc.index(s["r"])] for s in C.INTERACCIONES.values()]
    for k, ic in (("M2", aditivo), ("M3", pares)):
        fab = lambda ic=ic: GBM(interaction_constraints=ic)  # noqa: E731
        o = cpcv_pred(XG, y, w, fab)
        p = wf_pred(XG, y, w, fab)
        m, _ = _eval_wf(f"WF_reg_{k}_gbm", f"GBM {k} (restricciones de interacción)", p, est, refs, dict(variables=cc, modelo="3_gbm", fase="fase3"))
        res[k + "_gbm"] = dict(cpcv=puerta(o["agregada"], y, o["bench_agregado"]), wf=m)
        preds_c[k + "_gbm"], preds_w[k + "_gbm"] = o, p
    # Contrastes M1->M2, M2->M3 (ridge): DM, folds con mejora y nulo con regímenes sombra.
    comp = []
    for a, b, añadidos in (("M1", "M2", list(C.REGIMENES)), ("M2", "M3", list(C.INTERACCIONES)), ("M2_gbm", "M3_gbm", None)):
        oc_a, oc_b = preds_c[a], preds_c[b]
        dc, tc = estad.dm_t(y, oc_a["agregada"], oc_b["agregada"])
        pa, pbw = preds_w[a], preds_w[b]
        yy = P.loc[pa.index, "y"].to_numpy()
        dw, tw = estad.dm_t(yy, pa.to_numpy(), pbw.to_numpy())
        # Folds con mejora: particiones CPCV y años del walk-forward.
        fc = np.mean([np.nanmean((y - oc_a["M"][:, i]) ** 2) > np.nanmean((y - oc_b["M"][:, i]) ** 2) for i in range(oc_a["M"].shape[1])])
        anios = sorted(set(pa.index.year))
        fw = np.mean([((yy[pa.index.year == a_] - pa[pa.index.year == a_]) ** 2).mean() > ((yy[pa.index.year == a_] - pbw[pa.index.year == a_]) ** 2).mean() for a_ in anios])
        fila = dict(de=a, a=b, dr2_cpcv=res[b]["cpcv"]["r2_oos"] - res[a]["cpcv"]["r2_oos"], t_cpcv=tc,
                    dr2_wf=res[b]["wf"]["r2_oos"] - res[a]["wf"]["r2_oos"], t_wf=tw, folds_mejora_cpcv=fc, anios_mejora_wf=fw,
                    sharpe_a=res[a]["wf"]["sharpe_neto"], sharpe_b=res[b]["wf"]["sharpe_neto"],
                    sharpe_volesc=metricas(refs["bh_vol_escalado"].loc[pa.index[0]:])["sharpe_neto"])
        if añadidos is not None:
            base = M[a]
            nul = []
            for r_ in range(C.N_NULA_REG):
                Sh = fase2.sombras(P.loc[idx, añadidos], 1, C.SEMILLA + 5000 + r_)
                Sh.columns = añadidos
                XX = pd.concat([base, Sh], axis=1)
                o = cpcv_pred(XX, y, w, RidgeTS, n_jobs=4)
                nul.append(estad.r2_oos(y, o["agregada"], o["bench_agregado"]) - res[a]["cpcv"]["r2_oos"])
            fila["p95_dr2_sombras_cpcv"] = float(np.percentile(nul, 95))
            fila["supera_sombras"] = bool(fila["dr2_cpcv"] > fila["p95_dr2_sombras_cpcv"])
        fila["acepta"] = bool(
            fila["dr2_cpcv"] > 0 and fila["dr2_wf"] > 0 and tc >= 2 and tw >= 2 and fila.get("supera_sombras", True)
            and fc > 0.5 and fw > 0.5 and fila["sharpe_b"] > fila["sharpe_a"] and fila["sharpe_b"] > fila["sharpe_volesc"]
        )
        comp.append(fila)
    tc_ = pd.DataFrame(comp)
    tc_.to_csv(R / "06_regimen_comparacion.csv", index=False)
    # Coeficiente d de la interacción con HAC sobre el descubrimiento.
    dd = []
    for nombre, s in C.INTERACCIONES.items():
        b, t = estad.ols_hac(y, P.loc[idx, [s["x"], s["r"], nombre]].to_numpy())
        dd.append(dict(interaccion=nombre, x=s["x"], r=s["r"], b=b[1], c=b[2], d=b[3], t_d=t[3], signo_d=s["signo_d"],
                       pasa=bool(t[3] * s["signo_d"] >= 2.5)))
    pd.DataFrame(dd).to_csv(R / "06_interacciones.csv", index=False)
    js(res, "06_regimen.json")
    print(tc_.round(4).to_string())
    print(pd.DataFrame(dd).round(4).to_string())


# ------------------------------------------------------------------ sobreajuste y catálogo
def paso_sobreajuste():
    Mwf, n_wf = matriz_registro("walk_forward")
    Mtodo, n_todo = matriz_registro(("walk_forward", "regla_fija"))
    res = dict(versiones=[*C.VERSIONES_PREVIAS, C.VERSION], n_intentos_unicos=n_todo, n_intentos_wf=n_wf,
               wf=dict(**pbo_cscv(Mwf), **dsr(Mwf, n_todo)),
               todo=dict(**pbo_cscv(Mtodo), **dsr(Mtodo, n_todo)), periodo=[str(Mtodo.index[0].date()), str(Mtodo.index[-1].date())])
    js(res, "07_dsr_pbo.json")
    print(res)
    catalogo()


def catalogo():
    est = estado()
    t1 = pd.read_csv(R / "01_fase1_univariante.csv").set_index("variable")
    cl = pd.read_csv(R / "02_clusters.csv")
    tm = pd.read_csv(R / "02_mda.csv").set_index("cluster")
    ts = pd.read_csv(R / "05_estabilidad_shap.csv").set_index("cluster") if (R / "05_estabilidad_shap.csv").exists() else None
    gate = est["gate"]
    filas = []
    for r in cl.itertuples():
        for v in r.miembros.split():
            f = dict(variable=v, familia=C.FAMILIA[v], signo_esperado=C.SIGNOS[v], cluster=r.cluster,
                     t_nw=t1.loc[v, "t_nw"], p_nula=t1.loc[v, "p_nula"], pasa_fase1=bool(t1.loc[v, "pasa_fase1"]),
                     mda_gbm=tm.loc[r.cluster, "mda_gbm"], p95_sombras=tm.loc[r.cluster, "p95_sombras_gbm"],
                     sobrevive_mda=bool(tm.loc[r.cluster, "sobrevive"]))
            if ts is not None and r.cluster in ts.index:
                s = ts.loc[r.cluster]
                f.update(estable_shap=bool(s.estable_shap), signo_ok=bool(s.signo_ok), monotonia_min=s.monotonia_min,
                         forma_media=s.forma_media, ridge_signo_estable=bool(s.ridge_signo_estable))
            motivo = None
            if not f["sobrevive_mda"]:
                motivo = "su cluster no empeora la pérdida OOS más que las sombras (clustered MDA)"
            elif not gate["pasa"]:
                motivo = "el modelo no pasa la puerta de skill: su SHAP describe ruido ajustado"
            elif not f.get("estable_shap", False):
                s = ts.loc[r.cluster]
                fallos = []
                if not s.signo_ok:
                    fallos.append("signo no constante en los 6 bloques")
                if not s.monotonia_ok:
                    fallos.append(f"monotonía mínima {s.monotonia_min:.2f} (< 0,7 o <= p95 sombras {s.p95_monotonia_sombras:.2f})")
                if not s.forma_ok:
                    fallos.append(f"forma {s.forma_media:.2f} (p95 sombras {s.p95_forma_sombras:.2f})")
                if not s.relevancia_ok:
                    fallos.append("relevancia <= p95 de las sombras")
                motivo = "SHAP inestable: " + "; ".join(fallos)
            f["decision"] = "estable" if motivo is None else "descartada"
            f["motivo"] = motivo or "cumple MDA, puerta de skill y estabilidad SHAP"
            uni = "Sí" if f["pasa_fase1"] else "No"
            shp = "Sí" if f.get("estable_shap") else "No"
            f["lectura_A6"] = {("Sí", "Sí"): "efecto propio y robusto", ("Sí", "No"): "probable proxy",
                               ("No", "Sí"): "efecto condicional o de supresión", ("No", "No"): "sin efecto"}[(uni, shp)]
            filas.append(f)
    cat = pd.DataFrame(filas)
    cat.to_csv(R / "08_catalogo.csv", index=False)
    print(cat[["variable", "cluster", "t_nw", "pasa_fase1", "sobrevive_mda", "decision", "motivo"]].to_string())


# ------------------------------------------------------------------ walk-forward con las variables estables (A.6, paso 4)
def paso_estables():
    est = estado()
    cat = pd.read_csv(R / "08_catalogo.csv")
    estables = cat.loc[cat.decision == "estable", "cluster"].unique().tolist()
    if not estables:
        print("catálogo vacío: no hay walk-forward con variables estables")
        return
    P, idx, w = est["P"], est["idx"], est["w"]
    y = P.loc[idx, "y"].to_numpy()
    X = est["Xall"].loc[idx, estables]
    refs = _refs(est)
    res, bts = {}, {}
    for nombre, fab in ESCALERA.items():
        o = cpcv_pred(X, y, w, fab)
        p = wf_pred(X, y, w, fab)
        m, bt = _eval_wf(f"WF_estables_{nombre}", f"{nombre} sobre las variables estables {estables}", p, est, refs,
                         dict(variables=estables, modelo=nombre, fase="fase5_estables"))
        res[nombre] = dict(cpcv=puerta(o["agregada"], y, o["bench_agregado"]), wf=m)
        bts[nombre] = bt
        print(nombre, "CPCV", {k: round(v, 4) for k, v in res[nombre]["cpcv"].items()}, "| WF",
              {k: round(m[k], 4) for k in ("r2_oos", "cw_t", "sharpe_neto", "dif_sharpe_bh", "t_dif_sharpe_bh", "alfa_anual", "t_alfa", "rotacion_anual")})
    js(dict(estables=estables, resultados=res), "09_walk_forward_estables.json")
    figuras.curvas_capital({f"Combinación ({', '.join(estables)})": bts["0_combo"], "GBM": bts["3_gbm"],
                            "Buy & hold": refs["buy_hold"], "TSMOM 12-1": refs["tsmom_12_1"]},
                           R / "figuras" / "09_curvas_estables.png", "Walk-forward 1998-2012 con las variables estables (neto de 2 pb)")


PASOS = {"datos": paso_datos, "fase1": paso_fase1, "fase2": paso_fase2, "modelos": paso_modelos, "shap": paso_shap,
         "regimen": paso_regimen, "estables": paso_estables, "sobreajuste": paso_sobreajuste, "catalogo": catalogo}

if __name__ == "__main__":
    paso = sys.argv[1] if len(sys.argv) > 1 else "todo"
    for nombre, f in PASOS.items():
        if paso in (nombre, "todo") and not (paso == "todo" and nombre == "catalogo"):
            print(f"==== {nombre}")
            f()
