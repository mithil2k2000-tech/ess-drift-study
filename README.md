# Does an Explanation Stability Score notice data drift?

📄 **Paper (PDF): [ess-drift-study.pdf](ess-drift-study.pdf)** · [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23592998.svg)](https://doi.org/10.5281/zenodo.23592998)

A reproduction and stress test of the Explanation Stability Score (ESS) from my MSc dissertation,
on a recruitment dataset, Adult income and German credit.

`ESS = 0.4·G + 0.4·L + 0.2·F` combines global SHAP rank stability, local perturbation stability and SHAP–LIME
top-5 agreement, and gates what explanations a system may show (full at ≥ 0.75).

## Findings

* **The reproduction holds:** ESS = 0.881 on the original recruitment split.
* **G is constant as specified.** It compares a batch's ranking with itself, so it is exactly 1 on a single batch.
* **The gate never fired under drift.** Across 225 drifted batches (covariate shift up to 2 SD, up to 80% missing,
  prior shift to 97%), the lowest ESS was 0.763, even when balanced accuracy fell by 0.33. Under covariate shift
  L *rose* while G fell, so the components cancelled.
* **ESS does respond to a bad model.** Label noise plus overfitting took it to 0.684 and 0.715 on clean data.
* **A calibrated variant helps but is not enough.** With a held-out reference and a bootstrap threshold it caught 56–95%
  of harmful batches, with some false alarms. A plain PSI > 0.1 rule caught 78–95%.

Conclusion: use explanation stability to vet models, and monitor drift with input statistics.

## Layout

```
src/esslib.py       ESS (faithful to the dissertation), drift operators, PSI
src/datasets.py     the three datasets (downloaded on first run)
src/experiments.py  E1 reproduction, E2 dose-response, E3 variance and bootstrap null, E4 positive control
src/analyse.py      tables, figure and facts.json (also E5, the calibrated variant)
results/            every result file used in the paper
paper/              LaTeX source and compiled PDF
```

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cd src && python experiments.py e1 e2 e3 e4 && python analyse.py
cd ../paper && latexmk -pdf main.tex
```

The full run takes a few hours on two CPU cores, mostly LIME.

## AI assistance

An AI assistant (Claude, Anthropic) helped write experiment code and draft text. The author designed the
study questions, reviewed the code and results, and is responsible for the content.

## Licence

MIT
