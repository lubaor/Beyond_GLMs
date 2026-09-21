# Beyond GLMs

A reproducible benchmark of modern ML and causal methods for motor pricing, on
openly licensed data.

The question this repository answers is not "does gradient boosting beat a
GLM". It is "on which books does it beat a GLM, by how much, and when is the
honest answer that you cannot tell". Two of the three risk portfolios here are
cases where the boosted model does **not** win, and one diagnostic breaks down
badly enough that the only defensible output is a non-finding.

## The one rule

Every model shares one harness: identical split, seed, metric, exposure
weighting and diagnostics. Only the columns change when a new book enters. Any
difference is then attributable to the data or the model, never to the analyst
moving the goalposts between runs.

## The decision framework

[DECISION_FRAMEWORK.md](DECISION_FRAMEWORK.md) is the one-page takeaway: which
model class fits which use case, and when to stop and not build the model at
all. Three sequenced tests for the risk model, two gates for the causal
estimate, and the thresholds are derived from the runs in this repository
rather than asserted.

Read it first if you want the conclusions. Run the pipeline below if you want
to check them.

## Quick start

```bash
pip install -r requirements.txt pyreadr
python fetch_data.py
python 01_load_and_profile.py
```

Then run `02` through `07` in order for the frequency benchmark, and `08`
through `09` for the causal layer.

`fetch_data.py` was last run end to end on 2026-08-28. All five datasets
downloaded cleanly and reproduced content identical to the files these results
were computed from.

## Datasets

**No dataset is redistributed here.** `fetch_data.py` downloads each one from
its upstream source and verifies the row and column counts. Licences differ per
dataset and are recorded in [DATA_LICENCES.md](DATA_LICENCES.md).

| File | Source | Rows | Licence | Role |
|---|---|---|---|---|
| `01_freMTPL2_freq.csv` | OpenML id 41214 | 678,013 | CC0 1.0 | French motor, claim frequency |
| `02_freMTPL2_sev.csv` | CASdatasets `freMTPL2sev` | 26,444 | GPL (>= 2) | French motor, claim severities |
| `03_wasa_motorcycle.csv` | CASdatasets `swmotorcycle` | 64,548 | GPL (>= 2) | Swedish Wasa motorcycle, sparse claims |
| `04_ausprivauto_dejong.csv` | CASdatasets `ausprivauto0405` | 67,856 | GPL (>= 2) | Australian (De Jong and Heller), low signal |
| `05_eudirectlapse_demand.csv` | CASdatasets `eudirectlapse` | 23,060 | GPL (>= 2) | European renewal, lapse demand |

Row counts are the raw files. The pipeline applies its own exclusions, so the
modelled samples can be smaller: the motorcycle book drops 2,074 non-positive
exposure rows to 62,474.

### One step is not included here

The talk this repository accompanies also covers a new-business conversion
example, which fails the second causal gate. That step depends on a dataset
whose licence terms are unresolved, so it is kept out of this repository rather
than published with a source we cannot license cleanly. See
[DATA_LICENCES.md](DATA_LICENCES.md).

Nothing else depends on it. Steps `01` to `09` are complete and self-contained,
and the renewal book in step `09` already demonstrates the first gate.

## Pipeline

Run from the repository root, in order. Each step reads the previous step's
output from the working directory, so there is nothing to configure.

**Frequency: does a boosted model beat the GLM, and on which books?**

| Step | What it does |
|---|---|
| `01_load_and_profile.py` | Load and profile every downloaded portfolio |
| `02_glm_frequency.py` | Poisson GLM baseline |
| `03_gbm_frequency.py` | Untuned gradient boosting challenger |
| `04_calibration_and_lift.py` | Single-split calibration and lift |
| `05_crossfold_stability.py` | 5-fold cross-validation, untuned |
| `06_crossportfolio.py` | Same harness on the Australian and Swedish books |
| `07_tuned_gbm_frequency.py` | Nested CV with tuning, all three portfolios |

**Demand: can a price effect be identified at all?**

| Step | What it does |
|---|---|
| `08_dml_synthetic_validation.py` | DML against a planted ground truth |
| `08b_dml_synthetic_sweep.py` | Confounding-strength sweep |
| `09_dml_real_elasticity.py` | Renewal elasticity, fails gate 1 |

Steps `08` onward need the second environment (see below).

## Repository map

```
README.md                 start here
DECISION_FRAMEWORK.md     the one-page takeaway
DATA_LICENCES.md          one entry per source, per-dataset terms
LICENSE / NOTICE          Apache-2.0, code only

fetch_data.py             downloads all five datasets
requirements.txt          causal-layer environment

01_ .. 07_*.py            frequency benchmark, in order
08_ .. 09_*.py            causal layer, in order

results/                  committed outputs
  04_calibration_*.csv      single-split decile calibration tables
  07_tuned_gbm_summary.txt  the tuned 5-fold run behind the headline numbers
```

Downloaded data lands in the repository root and is gitignored. Fitted models
and prediction dumps are intermediate and also gitignored: only `results/` is
kept, because those are the numbers quoted in the talk.

Step `04` writes its two tables to `results/` directly. The other steps print
to stdout; `results/07_tuned_gbm_summary.txt` is a captured redirect of step
`07`, which is worth keeping because that run takes the longest.

## Environment

Two pinned environments, deliberately. Steps `02` to `07` were produced under
scikit-learn 1.9.0. Installing `econml` for the causal layer pins scikit-learn
down to 1.6.1, so steps `08` onward run there. `requirements.txt` pins the
causal environment; the split is intentional and is disclosed rather than
papered over.

`fetch_data.py` additionally needs `pyreadr` to read the CASdatasets `.rda`
binaries. See its docstring for an R fallback if `pyreadr` will not build.

## Headline results

From `results/07_tuned_gbm_summary.txt`, 5-fold, tuned GBM against an untuned GLM:

| Portfolio | GLM Gini | GBM Gini | Verdict |
|---|---|---|---|
| freMTPL2 | 0.2943 | 0.3525 | GBM wins, +20%, positive in all 5 folds |
| Australian | 0.0995 | 0.0858 | GLM keeps a small edge in every fold |
| Swedish | 0.5528 | 0.5564 | Statistical wash, per-fold gaps change sign |

The return scales with the estimable signal in the book, not with the model's
sophistication.

## Licence

Code in this repository is Apache-2.0. See [LICENSE](LICENSE). The datasets are
**not** covered by that grant, are not redistributed here, and each remains
under its own upstream terms. Four of the five are GPL (>= 2): see
[DATA_LICENCES.md](DATA_LICENCES.md).
