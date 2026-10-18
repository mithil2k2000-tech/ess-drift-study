"""Explanation Stability Score (ESS), generalised from the MSc dissertation code.

ESS = wG * G + wL * L + wF * F, with the dissertation's weights (0.4, 0.4, 0.2):

* G (global rank stability): 1 minus the mean normalised displacement of each
  feature's mean-|SHAP| rank relative to a reference ranking.
* L (local perturbation stability): 1 minus the relative L1 change in an
  instance's SHAP vector after adding Gaussian noise (5% of each numeric
  feature's standard deviation), averaged over sampled instances.
* F (cross-method fidelity): overlap of the top-k features from SHAP and LIME
  for the same instance, averaged over sampled instances.

The definitions follow the dissertation implementation exactly; only the
dataset- and model-specific parts are parameterised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lime.lime_tabular import LimeTabularExplainer

WEIGHTS = (0.4, 0.4, 0.2)
GATES = ((0.75, "full_explanations"), (0.60, "limited_explanations_with_warning"))


def gate(ess: float) -> str:
    for threshold, name in GATES:
        if ess >= threshold:
            return name
    return "disclaimer_only"


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    categorical: list[str]
    excluded: list[str] = field(default_factory=list)

    @property
    def numeric(self) -> list[str]:
        return [c for c in self.X.columns if c not in self.categorical]

    @property
    def features(self) -> list[str]:
        return list(self.X.columns)


def prepare(X: pd.DataFrame, ds: Dataset) -> pd.DataFrame:
    """Same imputation as the dissertation: 'MISSING' for categories, median for numbers."""
    Xs = X[ds.features].copy()
    for c in ds.categorical:
        Xs[c] = Xs[c].fillna("MISSING").astype(str)
    for c in ds.numeric:
        num = pd.to_numeric(Xs[c], errors="coerce").astype(float)
        Xs[c] = num.fillna(num.median())
    return Xs


def train_catboost(ds: Dataset, X_tr, y_tr, seed=42, iterations=500, depth=6, l2=3.0, lr=0.05) -> CatBoostClassifier:
    counts = y_tr.value_counts().sort_index()
    weights = [len(y_tr) / (2 * counts[0]), len(y_tr) / (2 * counts[1])]
    model = CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=lr, l2_leaf_reg=l2,
                               loss_function="Logloss", class_weights=weights, random_seed=seed,
                               verbose=False, thread_count=2, allow_writing_files=False)
    model.fit(Pool(prepare(X_tr, ds), y_tr, cat_features=ds.categorical))
    return model


class Explainer:
    """SHAP (CatBoost TreeSHAP) and LIME for one model and dataset."""

    def __init__(self, model: CatBoostClassifier, ds: Dataset, background: pd.DataFrame, seed: int = 42):
        self.model, self.ds = model, ds
        bg = prepare(background, ds)
        self.cat_idx = [ds.features.index(c) for c in ds.categorical]
        self.cat_names = {}
        bg_codes = bg.copy()
        for i in self.cat_idx:
            col = ds.features[i]
            cats = sorted(set(bg[col].astype(str)) | {"MISSING"})
            self.cat_names[i] = cats
            bg_codes[col] = pd.Categorical(bg[col].astype(str), categories=cats).codes
        self.lime = LimeTabularExplainer(
            bg_codes.values.astype(float), feature_names=ds.features, class_names=["0", "1"],
            categorical_features=self.cat_idx, categorical_names=self.cat_names,
            discretize_continuous=True, mode="classification", random_state=seed)

    def shap(self, X: pd.DataFrame) -> np.ndarray:
        Xs = prepare(X, self.ds)
        return self.model.get_feature_importance(Pool(Xs, cat_features=self.ds.categorical), type="ShapValues")[:, :-1]

    def to_codes(self, Xs: pd.DataFrame) -> np.ndarray:
        out = Xs.copy()
        for i in self.cat_idx:
            col = self.ds.features[i]
            out[col] = pd.Categorical(out[col].astype(str), categories=self.cat_names[i]).codes
            out[col] = out[col].replace(-1, self.cat_names[i].index("MISSING"))
        return out.values.astype(float)

    def _predict_codes(self, arr: np.ndarray) -> np.ndarray:
        temp = pd.DataFrame(arr, columns=self.ds.features)
        for i in self.cat_idx:
            col, cats = self.ds.features[i], self.cat_names[i]
            temp[col] = temp[col].round().astype(int).clip(0, len(cats) - 1).map(lambda z, cats=cats: cats[z])
        for c in self.ds.numeric:
            temp[c] = pd.to_numeric(temp[c], errors="coerce")
        return self.model.predict_proba(Pool(temp[self.ds.features], cat_features=self.ds.categorical))

    def lime_top(self, row_codes: np.ndarray, k: int = 5, num_samples: int = 2000) -> set[str]:
        prob = self._predict_codes(row_codes.reshape(1, -1))[0, 1]
        label = int(prob >= 0.5)
        exp = self.lime.explain_instance(row_codes, self._predict_codes, num_features=k, top_labels=1,
                                         num_samples=num_samples)
        found = set()
        for text, _ in exp.as_list(label=label):
            # Longest name first so "education-num" is not matched as "education".
            for f in sorted(self.ds.features, key=len, reverse=True):
                if f in text:
                    found.add(f)
                    break
        return found


def global_rank(shap_vals: np.ndarray, features: list[str]) -> list[str]:
    order = np.argsort(-np.abs(shap_vals).mean(axis=0), kind="stable")
    return [features[i] for i in order]


def compute_ess(ex: Explainer, X: pd.DataFrame, baseline_rank: list[str] | None = None, local_sample: int = 80,
                lime_sample: int = 60, k: int = 5, noise_frac: float = 0.05, seed: int = 42,
                weights=WEIGHTS, lime_num_samples: int = 2000) -> dict:
    ds = ex.ds
    rng = np.random.default_rng(seed)
    Xs = prepare(X, ds).reset_index(drop=True)
    shap_vals = ex.shap(Xs)
    rank = global_rank(shap_vals, ds.features)
    ref = baseline_rank or rank
    d = len(ds.features)
    G = float(np.mean([1 - abs(ref.index(f) - rank.index(f)) / max(1, d - 1) for f in ds.features]))

    # Local: perturb numeric features of sampled rows and compare SHAP vectors.
    idx = rng.choice(len(Xs), min(local_sample, len(Xs)), replace=False)
    x0 = Xs.iloc[idx].copy()
    x1 = x0.copy()
    for c in ds.numeric:
        std = Xs[c].std()
        if pd.notna(std) and std > 0:
            x1[c] = x1[c].astype(float) + rng.normal(0, noise_frac * std, len(x1))
    s0, s1 = ex.shap(x0), ex.shap(x1)
    local = np.maximum(0, 1 - np.abs(s0 - s1).sum(axis=1) / (np.abs(s0).sum(axis=1) + 1e-8))
    L = float(local.mean())

    # Fidelity: SHAP top-k vs LIME top-k for sampled rows.
    codes = ex.to_codes(Xs)
    agree = []
    for i in rng.choice(len(Xs), min(lime_sample, len(Xs)), replace=False):
        shap_top = {ds.features[j] for j in np.argsort(-np.abs(shap_vals[i]), kind="stable")[:k]}
        agree.append(len(shap_top & ex.lime_top(codes[i], k, lime_num_samples)) / k)
    F = float(np.mean(agree))

    wG, wL, wF = weights
    ess = wG * G + wL * L + wF * F
    return {"G": G, "L": L, "F": F, "ESS": float(ess), "gate": gate(ess), "rank": rank,
            "local_scores": local.tolist(), "agreements": agree}


# ---------------------------------------------------------------- drift operators

def covariate_shift(X: pd.DataFrame, cols: list[str], magnitude: float, seed: int = 42) -> pd.DataFrame:
    """Shift each column by magnitude * std (with 30% jitter), as in the dissertation (0.5 sd)."""
    rng = np.random.default_rng(seed)
    out = X.copy()
    for c in cols:
        std = pd.to_numeric(out[c], errors="coerce").std()
        if pd.notna(std) and std > 0 and magnitude > 0:
            out[c] = out[c].astype(float) + rng.normal(magnitude * std, 0.3 * magnitude * std, len(out))
    return out


def missingness(X: pd.DataFrame, cols: list[str], rate: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = X.copy()
    for c in cols:
        out[c] = out[c].astype(object) if c not in X.select_dtypes("number").columns else out[c]
        out.loc[rng.random(len(out)) < rate, c] = np.nan
    return out


def prior_shift(X: pd.DataFrame, y: pd.Series, positive_share: float, n: int, seed: int = 42):
    pos, neg = X[y == 1], X[y == 0]
    n_pos = int(round(n * positive_share))
    n_neg = n - n_pos
    pos_s = pos.sample(n=n_pos, replace=n_pos > len(pos), random_state=seed)
    neg_s = neg.sample(n=n_neg, replace=n_neg > len(neg), random_state=seed)
    Xs = pd.concat([pos_s, neg_s], ignore_index=True)
    ys = pd.Series([1] * n_pos + [0] * n_neg)
    order = np.random.default_rng(seed).permutation(len(Xs))  # positional shuffle: safe with resampled rows
    return Xs.iloc[order].reset_index(drop=True), ys.iloc[order].reset_index(drop=True)


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population stability index, the standard input-drift statistic in model monitoring."""
    ref, cur = ref.astype(object), cur.astype(object)
    numeric = pd.api.types.is_numeric_dtype(pd.to_numeric(ref, errors="coerce").dropna())
    if numeric and pd.to_numeric(ref, errors="coerce").nunique() > bins:
        r = pd.to_numeric(ref, errors="coerce")
        c = pd.to_numeric(cur, errors="coerce")
        edges = np.unique(np.quantile(r.dropna(), np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        rb = pd.cut(r, edges).astype(str).where(r.notna(), "NA")
        cb = pd.cut(c, edges).astype(str).where(c.notna(), "NA")
    else:
        rb, cb = ref.astype(str).fillna("NA"), cur.astype(str).fillna("NA")
    cats = sorted(set(rb) | set(cb))
    rp = rb.value_counts(normalize=True).reindex(cats, fill_value=0).values + 1e-6
    cp = cb.value_counts(normalize=True).reindex(cats, fill_value=0).values + 1e-6
    return float(np.sum((cp - rp) * np.log(cp / rp)))
