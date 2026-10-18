"""All experiments for the paper. Results go to ../results/*.csv; rerunning skips finished rows.

E1  Reproduction of the dissertation protocol on the recruitment data (25-run CV, baseline ESS).
E2  Dose-response: ESS, its components, input drift (PSI) and accuracy under graded drift, 3 datasets x 5 seeds.
E3  Estimator variance: spread of ESS at baseline across sampling seeds and sample sizes.
E4  Positive control: does ESS fall when the model itself becomes less stable (label noise, overfitting)?
E5  Calibrated variant: a held-out reference for G and a bootstrap null threshold, compared with PSI.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder

from datasets import ALL
from esslib import (Explainer, compute_ess, covariate_shift, missingness, prepare, prior_shift, psi,
                    train_catboost)

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)
SEEDS = [11, 22, 33, 44, 55]
LOCAL_N, LIME_N = 80, 40


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def append(name: str, row: dict) -> None:
    path = RESULTS / f"{name}.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def done(name: str) -> pd.DataFrame:
    path = RESULTS / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def setup(key: str, seed: int):
    ds = ALL[key]()
    X_tr, X_te, y_tr, y_te = train_test_split(ds.X, ds.y, test_size=0.3, random_state=seed, stratify=ds.y)
    model = train_catboost(ds, X_tr, y_tr, seed=seed)
    # Reference window and a separate monitoring window, both from unseen data.
    X_ref, X_mon, y_ref, y_mon = train_test_split(X_te, y_te, test_size=0.5, random_state=seed, stratify=y_te)
    n = min(500, len(X_mon))
    X_mon, y_mon = X_mon.iloc[:n].reset_index(drop=True), y_mon.iloc[:n].reset_index(drop=True)
    X_ref, y_ref = X_ref.iloc[:n].reset_index(drop=True), y_ref.iloc[:n].reset_index(drop=True)
    ex = Explainer(model, ds, X_tr, seed=seed)
    return ds, model, ex, X_ref, y_ref, X_mon, y_mon


def perf(model, ds, X, y) -> dict:
    p = model.predict_proba(prepare(X, ds))[:, 1]
    pred = (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "auc": roc_auc_score(y, p) if y.nunique() > 1 else np.nan}


def mean_psi(ds, X_ref, X_cur) -> float:
    return float(np.mean([psi(X_ref[c], X_cur[c]) for c in ds.features]))


# ------------------------------------------------------------------ E1

def e1():
    if (RESULTS / "e1_cv.csv").exists():
        return
    ds = ALL["recruitment"]()
    X, y = ds.X, ds.y
    rows = []
    for seed in SEEDS:
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y), 1):
            m = train_catboost(ds, X.iloc[tr], y.iloc[tr], seed=seed)
            pred = m.predict(prepare(X.iloc[te], ds)).astype(int)
            rows.append({"model": "CatBoost", "seed": seed, "fold": fold, **_scores(y.iloc[te], pred)})
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
            Xtr[ds.categorical] = enc.fit_transform(Xtr[ds.categorical])
            Xte[ds.categorical] = enc.transform(Xte[ds.categorical])
            rf = RandomForestClassifier(200, random_state=seed, n_jobs=2, class_weight="balanced").fit(Xtr, y.iloc[tr])
            rows.append({"model": "Random Forest", "seed": seed, "fold": fold, **_scores(y.iloc[te], rf.predict(Xte))})
        log("E1 seed", seed)
    pd.DataFrame(rows).to_csv(RESULTS / "e1_cv.csv", index=False)
    # Baseline ESS exactly as in the dissertation (500-row audit set, 100 local, 500 LIME).
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = train_catboost(ds, X_tr, y_tr, seed=42)
    ex = Explainer(m, ds, X, seed=42)
    audit = X_te.sample(n=500, random_state=42)
    r = compute_ess(ex, audit, local_sample=100, lime_sample=500, seed=42)
    json.dump({k: v for k, v in r.items() if k in ("G", "L", "F", "ESS", "gate")} | {"top5": r["rank"][:5]},
              open(RESULTS / "e1_baseline_ess.json", "w"), indent=2)
    log("E1 baseline", r["ESS"])


def _scores(y, pred):
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"), "minority_f1": f1_score(y, pred, pos_label=0)}


# ------------------------------------------------------------------ E2

def e2():
    prev = done("e2_dose")
    for key in ALL:
        for seed in SEEDS:
            if not prev.empty and ((prev.dataset == key) & (prev.seed == seed)).sum() >= 16:
                continue
            ds, model, ex, X_ref, y_ref, X_mon, y_mon = setup(key, seed)
            ref = compute_ess(ex, X_ref, local_sample=LOCAL_N, lime_sample=LIME_N, seed=seed)
            base = compute_ess(ex, X_mon, local_sample=LOCAL_N, lime_sample=LIME_N, seed=seed)
            base_rank = base["rank"]
            top_num = [f for f in base_rank if f in ds.numeric][:5]
            top_any = base_rank[:5]
            base_perf = perf(model, ds, X_mon, y_mon)
            scenarios = [("none", 0.0, X_mon, y_mon)]
            scenarios += [("covariate", m, covariate_shift(X_mon, top_num, m, seed), y_mon) for m in (0.25, 0.5, 1.0, 1.5, 2.0)]
            scenarios += [("missingness", r, missingness(X_mon, top_any, r, seed), y_mon) for r in (0.1, 0.2, 0.4, 0.6, 0.8)]
            share0 = float(y_mon.mean())
            for s in (0.1, 0.3, 0.5, 0.9, 0.97):
                Xp, yp = prior_shift(X_mon, y_mon, s, len(X_mon), seed)
                scenarios.append(("prior", s, Xp, yp))
            for kind, level, Xs, ys in scenarios:
                if kind == "none":
                    r_self = base
                else:
                    r_self = compute_ess(ex, Xs, baseline_rank=base_rank, local_sample=LOCAL_N, lime_sample=LIME_N, seed=seed)
                # Variant: G against the held-out reference window instead of the monitored batch itself.
                r_ref_G = float(np.mean([1 - abs(ref["rank"].index(f) - r_self["rank"].index(f)) / max(1, len(ds.features) - 1)
                                         for f in ds.features]))
                p = perf(model, ds, Xs, ys)
                append("e2_dose", {
                    "dataset": key, "seed": seed, "kind": kind, "level": level, "base_positive_share": share0,
                    "G": r_self["G"], "L": r_self["L"], "F": r_self["F"], "ESS": r_self["ESS"], "gate": r_self["gate"],
                    "G_heldout_ref": r_ref_G, "psi_mean": mean_psi(ds, X_mon, Xs),
                    **{f"{k}": v for k, v in p.items()},
                    **{f"delta_{k}": p[k] - base_perf[k] for k in p},
                })
            log("E2", key, seed)


# ------------------------------------------------------------------ E3

def e3():
    if (RESULTS / "e3_variance.csv").exists():
        return
    for key in ALL:
        ds, model, ex, X_ref, y_ref, X_mon, y_mon = setup(key, 11)
        for local_n, lime_n in ((20, 20), (80, 40), (200, 100)):
            for rep in range(10):
                r = compute_ess(ex, X_mon, local_sample=local_n, lime_sample=lime_n, seed=1000 + rep)
                append("e3_variance", {"dataset": key, "local_n": local_n, "lime_n": lime_n, "rep": rep,
                                       "G": r["G"], "L": r["L"], "F": r["F"], "ESS": r["ESS"]})
            log("E3", key, local_n, lime_n)
        # Bootstrap null: ESS with G measured against an independent reference window, on resampled
        # no-drift batches. This is the distribution a monitor should compare against.
        ref = compute_ess(ex, X_ref, local_sample=LOCAL_N, lime_sample=LIME_N, seed=7)
        pool = pd.concat([X_ref, X_mon]).reset_index(drop=True)
        for b in range(30):
            batch = pool.sample(n=len(X_mon), replace=True, random_state=b)
            r = compute_ess(ex, batch, baseline_rank=ref["rank"], local_sample=LOCAL_N, lime_sample=LIME_N, seed=2000 + b)
            append("e3_null", {"dataset": key, "b": b, "G": r["G"], "L": r["L"], "F": r["F"], "ESS": r["ESS"]})
        log("E3 null", key)


# ------------------------------------------------------------------ E4

def e4():
    if (RESULTS / "e4_control.csv").exists():
        return
    for key in ALL:
        ds = ALL[key]()
        for seed in SEEDS[:3]:
            X_tr, X_te, y_tr, y_te = train_test_split(ds.X, ds.y, test_size=0.3, random_state=seed, stratify=ds.y)
            audit = X_te.iloc[:500].reset_index(drop=True)
            y_audit = y_te.iloc[:500].reset_index(drop=True)
            rng = np.random.default_rng(seed)
            configs = [("regularised (default)", 0.0, dict()),
                       ("label noise 20%", 0.2, dict()),
                       ("label noise 40%", 0.4, dict()),
                       ("overfit (depth 10, l2 0.01, 1500 it)", 0.0, dict(depth=10, l2=0.01, iterations=1500, lr=0.1)),
                       ("overfit + label noise 40%", 0.4, dict(depth=10, l2=0.01, iterations=1500, lr=0.1))]
            ref_rank = None
            for name, noise, kw in configs:
                y_noisy = y_tr.copy()
                flip = rng.random(len(y_noisy)) < noise
                y_noisy[flip] = 1 - y_noisy[flip]
                m = train_catboost(ds, X_tr, y_noisy, seed=seed, **kw)
                ex = Explainer(m, ds, X_tr, seed=seed)
                r = compute_ess(ex, audit, local_sample=LOCAL_N, lime_sample=LIME_N, seed=seed)
                # Retrain stability: same config, different seed; how much does the global ranking move?
                m2 = train_catboost(ds, X_tr, y_noisy, seed=seed + 1, **kw)
                ex2 = Explainer(m2, ds, X_tr, seed=seed)
                r2 = compute_ess(ex2, audit, baseline_rank=r["rank"], local_sample=10, lime_sample=1, seed=seed)
                if ref_rank is None:
                    ref_rank = r["rank"]
                g_vs_clean = float(np.mean([1 - abs(ref_rank.index(f) - r["rank"].index(f)) / max(1, len(ds.features) - 1)
                                            for f in ds.features]))
                append("e4_control", {"dataset": key, "seed": seed, "config": name, "G": r["G"], "L": r["L"],
                                      "F": r["F"], "ESS": r["ESS"], "G_retrain": r2["G"], "G_vs_clean_model": g_vs_clean,
                                      **perf(m, ds, audit, y_audit)})
            log("E4", key, seed)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["e1", "e2", "e3", "e4"]
    for s in steps:
        log("start", s)
        globals()[s]()
        log("finished", s)
