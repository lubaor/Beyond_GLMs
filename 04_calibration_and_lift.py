"""
Step 4 - Calibration and lift diagnostics on the freMTPL2 frequency test set.

Fits NOTHING. Reuses the saved step-2 (GLM) and step-3 (GBM) test-set
predictions and evaluates them on the identical test set with the same exposure
weighting as steps 2-3.

Two diagnostics per model:
  1) Calibration by exposure-weighted predicted-risk decile.
  2) Discrimination via the normalised (exposure-weighted, ordered) Gini.
"""

import numpy as np
import pandas as pd

GLM_PRED = "02_glm_frequency_test_predictions.csv"
GBM_PRED = "03_gbm_frequency_test_predictions.csv"
CAL_GLM_OUT = "04_calibration_glm.csv"
CAL_GBM_OUT = "04_calibration_gbm.csv"

# ----------------------------------------------------------------------------
# 1. Load the two saved prediction sets and confirm they line up 1-for-1
# ----------------------------------------------------------------------------
glm = pd.read_csv(GLM_PRED)
gbm = pd.read_csv(GBM_PRED)

assert np.array_equal(glm["IDpol"].values, gbm["IDpol"].values), (
    "GLM and GBM prediction files are not aligned on the same test policies."
)

exposure = glm["Exposure"].values
claims = glm["ClaimNb"].values                 # observed claim counts (capped at 4)
observed_freq = glm["Frequency_actual"].values # = ClaimNb / Exposure
pred = {
    "GLM": glm["Frequency_pred_glm"].values,
    "GBM": gbm["Frequency_pred_gbm"].values,
}

# ----------------------------------------------------------------------------
# 2. Calibration by exposure-weighted decile of predicted frequency
# ----------------------------------------------------------------------------
def calibration_deciles(pred_freq, w, claim_cnt, n_bins=10):
    """Sort by predicted freq, split into n_bins of equal total exposure."""
    order = np.argsort(pred_freq, kind="mergesort")   # ascending, stable
    p, ww, cc = pred_freq[order], w[order], claim_cnt[order]

    cumw = np.cumsum(ww)
    total = cumw[-1]
    # midpoint cumulative-exposure fraction -> reduces boundary bias
    frac = (cumw - 0.5 * ww) / total
    dec = np.clip((frac * n_bins).astype(int), 0, n_bins - 1)

    rows = []
    for d in range(n_bins):
        m = dec == d
        sw = ww[m].sum()
        mean_pred = np.sum(ww[m] * p[m]) / sw
        claim_sum = cc[m].sum()
        mean_obs = claim_sum / sw
        rows.append(
            {
                "bin": d + 1,
                "exposure": sw,
                "mean_pred_freq": mean_pred,
                "mean_obs_freq": mean_obs,
                "pred_over_obs": mean_pred / mean_obs,
            }
        )
    return pd.DataFrame(rows)

cal_tables = {name: calibration_deciles(pred[name], exposure, claims) for name in pred}

def print_cal_table(name, tbl):
    print()
    print("=" * 70)
    print(f"  CALIBRATION BY EXPOSURE-WEIGHTED DECILE  -  {name}")
    print("  (bin 1 = lowest predicted risk ... bin 10 = highest)")
    print("=" * 70)
    print("  {:>3}{:>12}{:>16}{:>16}{:>12}".format(
        "bin", "exposure", "mean predicted", "mean observed", "pred/obs"))
    print("-" * 70)
    for _, r in tbl.iterrows():
        print("  {:>3}{:>12.1f}{:>16.4f}{:>16.4f}{:>12.3f}".format(
            int(r["bin"]), r["exposure"], r["mean_pred_freq"],
            r["mean_obs_freq"], r["pred_over_obs"]))
    print("=" * 70)

for name in ("GLM", "GBM"):
    print_cal_table(name, cal_tables[name])

cal_tables["GLM"].to_csv(CAL_GLM_OUT, index=False)
cal_tables["GBM"].to_csv(CAL_GBM_OUT, index=False)

# ----------------------------------------------------------------------------
# 3. Discrimination: normalised exposure-weighted ordered Gini
# ----------------------------------------------------------------------------
def ordered_gini(pred_freq, w, claim_cnt):
    """Gini from the exposure-weighted ordered Lorenz curve.

    Policies are ordered from lowest to highest predicted frequency; the Lorenz
    curve plots cumulative share of exposure (x) vs cumulative share of observed
    claims (y). Gini = 1 - 2 * area-under-curve. Higher = better risk ranking.
    """
    order = np.argsort(pred_freq, kind="mergesort")
    w_o = w[order]
    c_o = claim_cnt[order]

    x = np.concatenate([[0.0], np.cumsum(w_o) / w_o.sum()])
    y = np.concatenate([[0.0], np.cumsum(c_o) / c_o.sum()])
    area = np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0)  # trapezoidal AUC
    return 1.0 - 2.0 * area

# Oracle = ordering by the observed frequency itself (best achievable ranking).
gini_oracle = ordered_gini(observed_freq, exposure, claims)
raw_gini = {name: ordered_gini(pred[name], exposure, claims) for name in pred}
norm_gini = {name: raw_gini[name] / gini_oracle for name in pred}

print()
print("=" * 70)
print("  DISCRIMINATION  -  normalised Gini (model ranking / perfect ranking)")
print("=" * 70)
print("  Normalised Gini   ->   GLM: {:.4f}     GBM: {:.4f}".format(
    norm_gini["GLM"], norm_gini["GBM"]))
print("-" * 70)
print("  Context (raw ordered Gini):  GLM {:.4f}   GBM {:.4f}   oracle {:.4f}".format(
    raw_gini["GLM"], raw_gini["GBM"], gini_oracle))
print("=" * 70)
print()

print(f"Saved calibration table -> {CAL_GLM_OUT}")
print(f"Saved calibration table -> {CAL_GBM_OUT}")
