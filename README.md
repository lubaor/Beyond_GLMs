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

Then run `02` through `07f` in order for the frequency benchmark, and `08`
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

**Is it a fair fight? Tuning the GLM as hard as the challenger**

Steps `02` to `07` tune the boosted model and leave the GLM at a fixed
specification, which is the obvious objection to the headline. These four
steps answer it on freMTPL2 by tuning the incumbent too, one lever at a time.

| Step | What it does |
|---|---|
| `07b_tuned_glm_binning.py` | 32 quantile bins per numeric, step encoded, L2 on the jumps |
| `07c_tuned_glm_enet.py` | Adds elastic net and a monotone `BonusMalus` constraint |
| `07d_tuned_glm_interactions.py` | Forward search over all 36 pairwise interactions |
| `07e_tuned_glm_final.py` | Everything combined: the tuned GLM behind the headline |
| `07f_tuned_glm_crossportfolio.py` | The same treatment on the Australian and Swedish books |
| `07g_swedish_exposure_cap.py` | What the frozen exposure cap cost the Swedish book |

Numerics are **step encoded** as `1(bin > k)`, so each coefficient is the jump
between adjacent bins. L1 drives a jump to exactly zero and merges those bins,
so the surviving jumps are the binning the data chose rather than one an
analyst picked. Monotonicity in `BonusMalus` is then the box constraint
"every jump >= 0", enforced exactly rather than checked after the fact.

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
07b .. 07f_*.py           tuning the GLM, so the comparison is fair
08_ .. 09_*.py            causal layer, in order

results/                  committed outputs
  04_calibration_*.csv      single-split decile calibration tables
  07_tuned_gbm_summary.txt  the tuned GBM 5-fold run
  07e_tuned_glm_final.csv   the tuned GLM 5-fold run, freMTPL2
  07f_tuned_glm_crossportfolio.csv  the same, Australian and Swedish
  07g_swedish_exposure_cap.csv      capped vs uncapped, Swedish book
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

**freMTPL2, tuned against tuned.** Five-fold, identical folds throughout. This
is the comparison to quote: both sides had their hyperparameters and their
structure searched.

| Model | Normalised Gini | Poisson deviance | Worst-decile cal. err |
|---|---|---|---|
| GLM, fixed specification | 0.2943 ±0.0112 | 0.593092 | 0.1162 ±0.0588 |
| **GLM, tuned** (`07e`) | **0.3127 ±0.0107** | **0.585702** | **0.1151 ±0.0115** |
| **GBM, tuned** (`07`) | **0.3525 ±0.0098** | **0.571598** | **0.0977** |

Tuning the GLM closes **a third of the ranking gap** and 36% of the deviance
gap. The boosted model still wins by **+13%**, and it is still the better
calibrated of the two. Both halves of that sentence matter: the gap is smaller
than an untuned comparison suggests, and it does not go away.

What the GLM's tuning bought, one lever at a time:

| Variant | Gini | Worst-decile cal. err |
|---|---|---|
| Fixed specification | 0.2943 | 0.1162 ±0.0588 |
| + fine bins, L2 (`07b`) | 0.3097 | 0.1172 ±0.0271 |
| + elastic net, monotone (`07c`) | 0.3082 | 0.1454 ±0.0702 |
| + interactions, L2 (`07d`) | 0.3157 | 0.1502 ±0.0630 |
| all combined (`07e`) | 0.3127 | 0.1151 ±0.0115 |

Interactions carry most of the gain, and the constraints pay for themselves in
the tails: the combined model is the best calibrated GLM variant and by far the
most stable across folds (SD 0.0115 against 0.06 to 0.07 for the unconstrained
ones). Six crosses were selected in at least four of the five independent
per-fold searches, led by `VehAge x VehBrand` and `VehAge x VehPower` in 5/5.
That is structure a boosted model finds for free and a GLM has to be told.

**Across portfolios**, every model tuned (`07f` does the Australian and Swedish
GLMs, with ordered categoricals step encoded so the penalty can merge adjacent
levels the way an actuary groups bonus classes by hand):

| Portfolio | GLM fixed | GLM tuned | GBM tuned | Verdict |
|---|---|---|---|---|
| freMTPL2 | 0.2943 | **0.3127** | **0.3525** | GBM wins by +13% |
| Australian | 0.0995 | 0.0930 | 0.0858 | GLM keeps the edge; tuning does not help it |
| Swedish | 0.5528 | 0.5560 | 0.5564 | A wash, tuned or not: 0.0004 apart on a fold SD of 0.058 |

The tuning that bought +0.018 Gini on freMTPL2 buys **nothing** on the other
two. On the Australian book it is if anything slightly negative, and the
selection is visibly fitting noise: across five folds of the same data the
inner split chose alphas spanning two orders of magnitude, `l1_ratio` across
its whole range, and between 14 and 120 non-zero coefficients. On the Swedish
book the elastic net chose pure ridge in four folds of five and merged nothing
at all, because there was no grouping structure to find.

That is the same finding from the other direction: the return to sophistication
scales with the estimable signal in the book, and it does so whether the
sophistication is a boosted model or a carefully tuned GLM.

For the Australian book the honest figure to quote is the **fixed** GLM's
0.0995, because it is the better model and the tuned variant overfit its own
selection. "We tried, and it did not help" is the result.

### What the frozen harness cost the Swedish book

The harness caps `Exposure` at 1 for every portfolio. On freMTPL2 that is the
correct fix for a documented quirk: 0.2% of rows, 0.0% of exposure. The
Swedish data is aggregated over 1994 to 1998, so a row is a risk cell that can
hold up to 31 bike-years, and the same rule hits 23% of rows and discards
**32% of all policy-years**. Claim counts are untouched, so the implied
frequency on those rows inflates 2.4x while their weight in the fit collapses.

`07g` runs the book both ways. Deviance is not comparable across the two,
since capping changes the target and the weights, so only Gini and the decile
ratio are reported.

| | Gini capped | Gini uncapped | worst decile capped | uncapped |
|---|---|---|---|---|
| GLM, fixed | 0.5528 ±0.0636 | 0.5679 ±0.0692 | 1.4049 ±1.4668 | 1.5473 ±0.8046 |
| GBM, untuned | 0.5336 ±0.0392 | 0.5455 ±0.0478 | 1.8734 ±1.7575 | 1.6371 ±0.5623 |
| GBM, tuned | 0.5526 ±0.0611 | 0.5535 ±0.0644 | 1.3453 ±1.2494 | 1.3526 ±1.3343 |

**The wash survives.** Uncapped the GLM is nominally ahead by 0.014 instead of
level, which is 0.2 of a fold SD. An ordering that flips under a preprocessing
change is not an ordering.

**The cap inflated the diagnostic's variance, not its level.** Miscalibration
stays at 1.35 to 1.65 either way, so the worst decile is still out by 135 to
165% with every policy-year restored. What changes is the spread: removing the
cap cuts the fold-to-fold SD by 45% for the GLM and 68% for the untuned GBM.
The broken diagnostic is therefore a sparsity problem, and the cap was making
it noisier on top rather than causing it.

## Licence

Code in this repository is Apache-2.0. See [LICENSE](LICENSE). The datasets are
**not** covered by that grant, are not redistributed here, and each remains
under its own upstream terms. Four of the five are GPL (>= 2): see
[DATA_LICENCES.md](DATA_LICENCES.md).
