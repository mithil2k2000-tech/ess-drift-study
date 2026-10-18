"""Turn results/*.csv into the paper's tables (LaTeX) and figures (PDF)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R, FIG, TAB = ROOT / "results", ROOT / "paper" / "figures", ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
NAMES = {"recruitment": "Recruitment", "adult": "Adult", "german": "German credit"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"ESS": "#1f4e79", "G": "#7f7f7f", "L": "#2e8b57", "F": "#c0504d", "psi": "#b8860b", "acc": "#6a3d9a"}

facts: dict = {}


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float(np.mean(x)), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def table(df: pd.DataFrame, name: str, caption: str, label: str, colfmt: str | None = None):
    colfmt = colfmt or "l" + "r" * (df.shape[1] - 1)
    body = df.to_latex(index=False, escape=False, column_format=colfmt)
    body = body.replace("\\toprule", "\\toprule").replace("\\begin{tabular}", "\\begin{tabular}")
    (TAB / f"{name}.tex").write_text(
        f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\small\n\\setlength{{\\tabcolsep}}{{4pt}}\n{body}\\end{{table}}\n")


def e1():
    cv = pd.read_csv(R / "e1_cv.csv")
    base = json.loads((R / "e1_baseline_ess.json").read_text())
    rows = []
    for model, g in cv.groupby("model", sort=False):
        row = {"Model": model}
        for m, lab in [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced acc."), ("macro_f1", "Macro F1"),
                       ("minority_f1", "Not-joined F1")]:
            mu, h = ci(g[m])
            row[lab] = f"{mu:.3f} $\\pm$ {h:.3f}"
        rows.append(row)
    table(pd.DataFrame(rows), "e1_cv", "Predictive performance on the recruitment data, 25 runs (5 seeds $\\times$ 5 "
          "folds), mean $\\pm$ 95\\% CI. Both models use class weighting.", "tab:cv")
    facts["e1"] = {"baseline": base, "cv": {m: g[["accuracy", "balanced_accuracy"]].mean().round(4).to_dict()
                                            for m, g in cv.groupby("model")}}


def e2():
    d = pd.read_csv(R / "e2_dose.csv")
    kinds = [("covariate", "Covariate shift (SD units)"), ("missingness", "Missing rate on top-5 features"),
             ("prior", "Share of positive class")]
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.0), sharey="row")
    out = []
    for i, key in enumerate(["recruitment", "adult", "german"]):
        dk = d[d.dataset == key]
        none = dk[dk.kind == "none"]
        out.append({"dataset": key, "kind": "none", "level": 0.0, "ESS": none.ESS.mean(), "ESS_sd": none.ESS.std(),
                     "G": none.G.mean(), "L": none.L.mean(), "F": none.F.mean(), "psi": none.psi_mean.mean(),
                     "dbacc": 0.0, "dauc": 0.0, "gate_full_share": (none.gate == "full_explanations").mean()})
        for j, (kind, xlabel) in enumerate(kinds):
            ax = axes[i, j]
            sub = pd.concat([none.assign(level=none.base_positive_share if kind == "prior" else 0.0), dk[dk.kind == kind]])
            g = sub.groupby("level")
            lv = np.array(sorted(sub.level.unique()))
            for col, lab, c in [("ESS", "ESS", COLORS["ESS"]), ("G", "G", COLORS["G"]), ("L", "L", COLORS["L"]),
                                ("F", "F", COLORS["F"])]:
                m = g[col].mean().reindex(lv)
                s = g[col].std().reindex(lv)
                ax.plot(lv, m, marker="o", ms=2.5, lw=1.2 if col == "ESS" else 0.8, color=c, label=lab)
                ax.fill_between(lv, m - s, m + s, color=c, alpha=0.12, lw=0)
            ax2 = ax.twinx()
            ax2.plot(lv, g["delta_balanced_accuracy"].mean().reindex(lv), ls="--", color=COLORS["acc"], lw=0.9,
                     label="$\\Delta$ bal. acc.")
            ax2.set_ylim(-0.5, 0.1)
            ax2.tick_params(labelsize=6)
            if j < 2:
                ax2.set_yticklabels([])
            ax.axhline(0.75, color="k", lw=0.4, ls=":")
            ax.set_ylim(0.3, 1.02)
            if i == 2:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(NAMES[key])
            if i == 0 and j == 0:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower left", frameon=False, ncol=2)
            # Statistics for the text and table.
            for lvl, gg in g:
                out.append({"dataset": key, "kind": kind, "level": lvl, "ESS": gg.ESS.mean(), "ESS_sd": gg.ESS.std(),
                            "G": gg.G.mean(), "L": gg.L.mean(), "F": gg.F.mean(), "psi": gg.psi_mean.mean(),
                            "dbacc": gg.delta_balanced_accuracy.mean(), "dauc": gg.delta_auc.mean(),
                            "gate_full_share": (gg.gate == "full_explanations").mean()})
    fig.tight_layout()
    fig.savefig(FIG / "dose_response.pdf")
    plt.close(fig)
    s = pd.DataFrame(out)
    s.to_csv(R / "e2_summary.csv", index=False)

    # Correlations between ESS and harm, pooled per dataset over all drift conditions.
    corr_rows = []
    for key in ["recruitment", "adult", "german"]:
        dk = d[(d.dataset == key) & (d.kind != "none")]
        r_ess, p_ess = stats.spearmanr(dk.ESS, dk.delta_balanced_accuracy)
        r_psi, p_psi = stats.spearmanr(dk.psi_mean, -dk.delta_balanced_accuracy)
        corr_rows.append({"Dataset": NAMES[key],
                          "$\\rho$(ESS, $\\Delta$bal.acc.)": f"{r_ess:.2f}",
                          "$\\rho$(PSI, $-\\Delta$bal.acc.)": f"{r_psi:.2f}",
                          "Batches gated below 0.75": f"{(dk.gate != 'full_explanations').mean() * 100:.0f}\\%",
                          "Min.~ESS": f"{dk.ESS.min():.3f}"})
    table(pd.DataFrame(corr_rows), "e2_corr", "Does ESS track harm? Spearman correlation between each signal and the "
          "change in balanced accuracy over all drifted batches (75 per dataset), the share of drifted batches the "
          "dissertation's 0.75 gate would have restricted, and the lowest ESS observed.", "tab:corr")
    facts["e2"] = {"corr": corr_rows,
                   "worst": s.sort_values("dbacc").groupby("dataset").head(1)[["dataset", "kind", "level", "ESS", "dbacc", "psi"]]
                   .round(3).to_dict("records"),
                   "baseline": s[s.kind == "none"][["dataset", "ESS", "G", "L", "F"]].round(3).to_dict("records")}

    # Compact table: baseline vs strongest level of each drift type.
    rows = []
    for key in ["recruitment", "adult", "german"]:
        sk = s[s.dataset == key]
        b = sk[sk.kind == "none"].iloc[0]
        rows.append({"Dataset": NAMES[key], "Condition": "No drift", "ESS": f"{b.ESS:.3f}", "G": f"{b.G:.3f}",
                     "L": f"{b.L:.3f}", "F": f"{b.F:.3f}", "PSI": f"{b.psi:.2f}", "$\\Delta$bal.acc.": "--"})
        for kind, lvl, lab in [("covariate", 2.0, "Covariate +2 SD"), ("missingness", 0.8, "80\\% missing"),
                               ("prior", 0.97, "Prior 97\\% positive")]:
            r = sk[(sk.kind == kind) & np.isclose(sk.level, lvl)].iloc[0]
            rows.append({"Dataset": "", "Condition": lab, "ESS": f"{r.ESS:.3f}", "G": f"{r.G:.3f}", "L": f"{r.L:.3f}",
                         "F": f"{r.F:.3f}", "PSI": f"{r.psi:.2f}", "$\\Delta$bal.acc.": f"{r.dbacc:+.3f}"})
    table(pd.DataFrame(rows), "e2_extremes", "ESS and its components at baseline and at the strongest level of each "
          "drift type (mean of 5 seeds). PSI is the mean population stability index across features.", "tab:extremes",
          "llrrrrrr")


def e3():
    v = pd.read_csv(R / "e3_variance.csv")
    rows = []
    for (key, ln, mn), g in v.groupby(["dataset", "local_n", "lime_n"], sort=False):
        rows.append({"Dataset": NAMES[key], "$n_L$ / $n_F$": f"{ln} / {mn}", "ESS mean": f"{g.ESS.mean():.3f}",
                     "ESS SD": f"{g.ESS.std():.4f}", "L SD": f"{g.L.std():.4f}", "F SD": f"{g.F.std():.4f}",
                     "Range": f"{g.ESS.max() - g.ESS.min():.3f}"})
    table(pd.DataFrame(rows), "e3_variance", "Sampling variability of ESS on an unchanged batch: 10 repetitions with "
          "different sampling seeds for each local ($n_L$) and LIME ($n_F$) sample size.", "tab:variance", "llrrrrr")
    null = pd.read_csv(R / "e3_null.csv")
    facts["e3"] = {"variance": rows,
                   "null": {k: {"ESS_mean": g.ESS.mean(), "ESS_p05": g.ESS.quantile(0.05), "G_mean": g.G.mean(),
                                "G_min": g.G.min()} for k, g in null.groupby("dataset")}}


def e4():
    c = pd.read_csv(R / "e4_control.csv")
    order = list(dict.fromkeys(c.config))
    rows = []
    for key in ["recruitment", "adult", "german"]:
        for i, conf in enumerate(order):
            g = c[(c.dataset == key) & (c.config == conf)]
            rows.append({"Dataset": NAMES[key] if i == 0 else "", "Model": conf.replace("%", "\\%"), "ESS": f"{g.ESS.mean():.3f}",
                         "L": f"{g.L.mean():.3f}", "F": f"{g.F.mean():.3f}", "G (retrain)": f"{g.G_retrain.mean():.3f}",
                         "Bal. acc.": f"{g.balanced_accuracy.mean():.3f}"})
    table(pd.DataFrame(rows), "e4_control", "Positive control: models made less reliable by label noise or overfitting, "
          "evaluated on the same clean audit batch (mean of 3 seeds). G (retrain) compares global rankings between two "
          "training runs of the same configuration.", "tab:control", "llrrrrr")
    facts["e4"] = rows


def e5():
    """Calibrated monitor: G against a held-out reference, alarm below the 5th percentile of a bootstrap null."""
    d = pd.read_csv(R / "e2_dose.csv")
    null = pd.read_csv(R / "e3_null.csv")
    rows, recs = [], []
    for key in ["recruitment", "adult", "german"]:
        n = null[null.dataset == key]
        dk = d[d.dataset == key].copy()
        dk["ESS_cal"] = 0.4 * dk.G_heldout_ref + 0.4 * dk.L + 0.2 * dk.F
        thr = n.ESS.quantile(0.05)
        # PSI threshold of 0.1 is the conventional "moderate shift" rule of thumb.
        drifted = dk[dk.kind != "none"]
        harmful = drifted[drifted.delta_balanced_accuracy <= -0.02]
        rows.append({"Dataset": NAMES[key], "Threshold": f"{thr:.3f}",
                     "False alarms": f"{(dk[dk.kind == 'none'].ESS_cal < thr).mean() * 100:.0f}\\%",
                     "ESS$^{\\ast}$": f"{(harmful.ESS_cal < thr).mean() * 100:.0f}\\%",
                     "Gate 0.75": f"{(harmful.gate != 'full_explanations').mean() * 100:.0f}\\%",
                     "PSI$>$0.1": f"{(harmful.psi_mean > 0.1).mean() * 100:.0f}\\%",
                     "Harmful": f"{len(harmful)}"})
        recs.append({"dataset": key, "thr": thr, "n_harmful": len(harmful)})
    table(pd.DataFrame(rows), "e5_calibrated", "Detection of harmful drift (balanced accuracy down by at least 2 points) "
          "by the original fixed gate, the calibrated ESS$^{\\ast}$ (held-out reference, bootstrap 5th-percentile "
          "threshold) and a PSI rule of thumb. False alarms: share of clean batches flagged by ESS$^{\\ast}$. Harmful: number of "
          "harmful batches.", "tab:calibrated", "lrrrrrr")
    facts["e5"] = rows


if __name__ == "__main__":
    for step in (e1, e2, e3, e4, e5):
        try:
            step()
            print("ok", step.__name__)
        except FileNotFoundError as e:
            print("skip", step.__name__, e)
    (R / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
