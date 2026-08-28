"""
Step 5 - Cross-fold stability of the GLM vs the GBM on freMTPL2 frequency.

Same setup as steps 2-4 (ClaimNb capped at 4, Frequency = ClaimNb/Exposure,
Exposure as sample weight, same per-model preprocessing, same mean-Poisson-
deviance scoring, same exposure-weighted decile calibration and normalised Gini),
but evaluated with 5-fold cross-validation instead of a single 90/10 split.

In every fold BOTH models are refit on the training folds and scored on the
held-out fold. Only touches 01_freMTPL2_freq.csv.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

INPUT_CSV = "01_freMTPL2_freq.csv"
RANDOM_SEED = 0
N_SPLITS = 5
AREA_ORDER = ["A", "B", "C", "D", "E", "F"]

# ----------------------------------------------------------------------------
# Model builders — IDENTICAL to steps 2 (GLM) and 3 (GBM)
# ----------------------------------------------------------------------------
def build_glm():
    log_scale_transformer = make_pipeline(
        FunctionTransformer(np.log, validate=False), StandardScaler()
    )
    preprocessor = ColumnTransformer(
        [
            ("passthrough_numeric", "passthrough", ["BonusMalus"]),
            ("binned_numeric",
             KBinsDiscretizer(n_bins=10, random_state=RANDOM_SEED),
             ["VehAge", "DrivAge"]),
            ("log_scaled_numeric", log_scale_transformer, ["Density"]),
            ("ordinal_area", OrdinalEncoder(categories=[AREA_ORDER]), ["Area"]),
            ("onehot_categorical", OneHotEncoder(handle_unknown="ignore"),
             ["VehBrand", "VehPower", "VehGas", "Region"]),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor",
             PoissonRegressor(alpha=1e-12, solver="newton-cholesky", max_iter=300)),
        ]
    )


def build_gbm():
    preprocessor = ColumnTransformer(
        [
            ("categorical",
             OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             ["VehBrand", "VehPower", "VehGas", "Region", "Area"]),
            ("numeric", "passthrough", ["VehAge", "DrivAge", "BonusMalus", "Density"]),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor",
             HistGradientBoostingRegressor(
                 loss="poisson",
                 categorical_features=[0, 1, 2, 3, 4],
                 random_state=RANDOM_SEED)),
        ]
    )

# ----------------------------------------------------------------------------
# Scoring helpers — IDENTICAL definitions to step 4
# ----------------------------------------------------------------------------
def ordered_gini(pred_freq, w, claim_cnt):
    order = np.argsort(pred_freq, kind="mergesort")
    w_o, c_o = w[order], claim_cnt[order]
    x = np.concatenate([[0.0], np.cumsum(w_o) / w_o.sum()])
    y = np.concatenate([[0.0], np.cumsum(c_o) / c_o.sum()])
    area = np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0)
    return 1.0 - 2.0 * area


def normalised_gini(pred_freq, observed_freq, w, claim_cnt):
    return ordered_gini(pred_freq, w, claim_cnt) / ordered_gini(observed_freq, w, claim_cnt)


def decile_ratios(pred_freq, w, claim_cnt, n_bins=10):
    """Return predicted/observed ratio for each exposure-weighted decile."""
    order = np.argsort(pred_freq, kind="mergesort")
    p, ww, cc = pred_freq[order], w[order], claim_cnt[order]
    frac = (np.cumsum(ww) - 0.5 * ww) / ww.sum()
    dec = np.clip((frac * n_bins).astype(int), 0, n_bins - 1)
    ratios = []
    for d in range(n_bins):
        m = dec == d
        sw = ww[m].sum()
        mean_pred = np.sum(ww[m] * p[m]) / sw
        mean_obs = cc[m].sum() / sw
        ratios.append(mean_pred / mean_obs)
    return np.array(ratios)   # index 0 = bottom (lowest-risk) decile

# ----------------------------------------------------------------------------
# Load + identical preprocessing / target as steps 2-4
# ----------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV).reset_index(drop=True)
df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
df["Exposure"] = df["Exposure"].clip(upper=1)
df["Frequency"] = df["ClaimNb"] / df["Exposure"]

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

# Per-fold metric stores
metrics = {
    "GLM": {"dev": [], "gini": [], "worstcal": []},
    "GBM": {"dev": [], "gini": [], "worstcal": []},
}
gbm_bottom_ratio = []   # GBM lowest-risk-decile pred/obs, one per fold

for fold, (tr_idx, te_idx) in enumerate(kf.split(df), start=1):
    df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]
    y_te = df_te["Frequency"].values
    w_te = df_te["Exposure"].values
    c_te = df_te["ClaimNb"].values
    obs_freq_te = y_te   # observed frequency = ClaimNb / Exposure

    for name, builder in (("GLM", build_glm), ("GBM", build_gbm)):
        model = builder()
        model.fit(df_tr, df_tr["Frequency"],
                  regressor__sample_weight=df_tr["Exposure"])
        pred = model.predict(df_te)

        dev = mean_poisson_deviance(y_te, pred, sample_weight=w_te)
        gini = normalised_gini(pred, obs_freq_te, w_te, c_te)
        ratios = decile_ratios(pred, w_te, c_te)
        worstcal = np.max(np.abs(ratios - 1.0))

        metrics[name]["dev"].append(dev)
        metrics[name]["gini"].append(gini)
        metrics[name]["worstcal"].append(worstcal)

        if name == "GBM":
            gbm_bottom_ratio.append(ratios[0])

    print(f"  fold {fold}/{N_SPLITS} done")

# ----------------------------------------------------------------------------
# 1) Summary table: mean +/- std (ddof=1) across the 5 folds
# ----------------------------------------------------------------------------
def ms(vals):
    a = np.array(vals)
    return a.mean(), a.std(ddof=1)

print()
print("=" * 78)
print("  5-FOLD CROSS-VALIDATION SUMMARY  -  freMTPL2 frequency (mean +/- std)")
print("=" * 78)
print("  {:<8}{:>22}{:>22}{:>22}".format(
    "Model", "Poisson deviance", "Normalised Gini", "Worst-decile cal.err"))
print("-" * 78)
for name in ("GLM", "GBM"):
    dm, ds = ms(metrics[name]["dev"])
    gm, gs = ms(metrics[name]["gini"])
    cm, cs = ms(metrics[name]["worstcal"])
    print("  {:<8}{:>13.6f} +/-{:>6.6f}{:>13.4f} +/-{:>6.4f}{:>13.4f} +/-{:>6.4f}".format(
        name, dm, ds, gm, gs, cm, cs))
print("=" * 78)

# ----------------------------------------------------------------------------
# 2) GBM bottom (lowest-risk) decile predicted/observed ratio, per fold
# ----------------------------------------------------------------------------
print()
print("  GBM lowest-risk-decile predicted/observed ratio, by fold:")
for i, r in enumerate(gbm_bottom_ratio, start=1):
    print("      fold {}:  {:.3f}".format(i, r))
print("      (values > 1.00 = GBM over-predicts risk for the safest policies)")
print()

# ----------------------------------------------------------------------------
# Extra context for the write-up: per-fold Gini gap and calibration comparison
# ----------------------------------------------------------------------------
gini_gap = np.array(metrics["GBM"]["gini"]) - np.array(metrics["GLM"]["gini"])
cal_gbm = np.array(metrics["GBM"]["worstcal"])
cal_glm = np.array(metrics["GLM"]["worstcal"])
print("  Per-fold GBM-minus-GLM Gini gap:", np.round(gini_gap, 4).tolist())
print("  Folds where GBM worst-cal > GLM worst-cal: {}/{}".format(
    int(np.sum(cal_gbm > cal_glm)), N_SPLITS))
print()
