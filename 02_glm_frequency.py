"""
Step 2 — Classical model on the French freMTPL2 frequency data.

Builds a Poisson frequency GLM (scikit-learn PoissonRegressor) end to end,
following the preprocessing in scikit-learn's "Poisson regression and non-normal
loss" example, and validates it against an intercept-only null model.

Only touches 01_freMTPL2_freq.csv.
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

INPUT_CSV = "01_freMTPL2_freq.csv"
MODEL_OUT = "02_glm_frequency_model.joblib"
PRED_OUT = "02_glm_frequency_test_predictions.csv"
RANDOM_SEED = 0

# ----------------------------------------------------------------------------
# 1. Load and apply the standard freMTPL2 preprocessing
# ----------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)

# Standard literature caps: at most 4 claims, at most 1 year of exposure.
df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
df["Exposure"] = df["Exposure"].clip(upper=1)

# Target: claim frequency = number of claims per year of exposure.
df["Frequency"] = df["ClaimNb"] / df["Exposure"]

# Area is a genuinely ordered category (A < B < C < D < E < F, by density).
AREA_ORDER = ["A", "B", "C", "D", "E", "F"]

# ----------------------------------------------------------------------------
# 2. Feature preprocessing (mirrors the scikit-learn example)
# ----------------------------------------------------------------------------
log_scale_transformer = make_pipeline(
    FunctionTransformer(np.log, validate=False, feature_names_out="one-to-one"),
    StandardScaler(),
)

linear_model_preprocessor = ColumnTransformer(
    [
        # BonusMalus kept as a single raw numeric column -> one interpretable coef.
        ("passthrough_numeric", "passthrough", ["BonusMalus"]),
        # Numeric driver/vehicle ages -> quantile bins.
        (
            "binned_numeric",
            KBinsDiscretizer(n_bins=10, random_state=RANDOM_SEED),
            ["VehAge", "DrivAge"],
        ),
        # Density -> log then standard-scaled.
        ("log_scaled_numeric", log_scale_transformer, ["Density"]),
        # Area -> ordered category (single ordinal column).
        (
            "ordinal_area",
            OrdinalEncoder(categories=[AREA_ORDER]),
            ["Area"],
        ),
        # Remaining categoricals -> one-hot.
        (
            "onehot_categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["VehBrand", "VehPower", "VehGas", "Region"],
        ),
    ],
    remainder="drop",
)

# ----------------------------------------------------------------------------
# 3. Train / test split (same 90/10 split and seed as the example)
# ----------------------------------------------------------------------------
df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)

# ----------------------------------------------------------------------------
# 4. Fit the Poisson GLM, weighting each row by its Exposure
# ----------------------------------------------------------------------------
poisson_glm = Pipeline(
    [
        ("preprocessor", linear_model_preprocessor),
        (
            "regressor",
            PoissonRegressor(alpha=1e-12, solver="newton-cholesky", max_iter=300),
        ),
    ]
)

poisson_glm.fit(
    df_train, df_train["Frequency"], regressor__sample_weight=df_train["Exposure"]
)

# ----------------------------------------------------------------------------
# 5. Intercept-only null model (exposure-weighted mean frequency)
# ----------------------------------------------------------------------------
null_model = DummyRegressor(strategy="mean")
null_model.fit(df_train, df_train["Frequency"], sample_weight=df_train["Exposure"])

# ----------------------------------------------------------------------------
# 6. Evaluate out-of-sample (test set), weighting by Exposure
# ----------------------------------------------------------------------------
y_test = df_test["Frequency"].values
w_test = df_test["Exposure"].values

pred_glm = poisson_glm.predict(df_test)
pred_null = null_model.predict(df_test)

dev_glm = mean_poisson_deviance(y_test, pred_glm, sample_weight=w_test)
dev_null = mean_poisson_deviance(y_test, pred_null, sample_weight=w_test)
pct_reduction = 100.0 * (1.0 - dev_glm / dev_null)

# ----------------------------------------------------------------------------
# 7. Coefficient signs for BonusMalus and DrivAge
# ----------------------------------------------------------------------------
feature_names = poisson_glm.named_steps["preprocessor"].get_feature_names_out()
coefs = poisson_glm.named_steps["regressor"].coef_

# BonusMalus is a standalone raw numeric column, so its coefficient is uniquely
# identified: we can read its sign directly.
bm_idx = list(feature_names).index("passthrough_numeric__BonusMalus")
bm_coef = coefs[bm_idx]
bm_sign = "positive (+)" if bm_coef > 0 else "negative (-)"

# DrivAge is binned into 10 one-hot bins that (with the intercept and the other
# full dummy groups) are collinear, so an individual bin coefficient is not
# uniquely interpretable. We therefore read DrivAge's effect direction from the
# model's PREDICTIONS, which ARE identified: the correlation between driver age
# and predicted frequency on the test set.
da_corr = np.corrcoef(df_test["DrivAge"].values, pred_glm)[0, 1]
da_sign = "positive (+)" if da_corr > 0 else "negative (-)"

# ----------------------------------------------------------------------------
# 8. Print one clearly labelled results table
# ----------------------------------------------------------------------------
print()
print("=" * 74)
print("  POISSON FREQUENCY GLM  -  out-of-sample validation (freMTPL2freq)")
print("  Test set: {:,} policies (10% hold-out, seed={})".format(len(df_test), RANDOM_SEED))
print("=" * 74)
print("  {:<52}{:>18}".format("Metric", "Value"))
print("-" * 74)
print("  {:<52}{:>18.6f}".format("(a) Mean Poisson deviance - GLM", dev_glm))
print("  {:<52}{:>18.6f}".format("(b) Mean Poisson deviance - null (intercept only)", dev_null))
print("  {:<52}{:>17.2f}%".format("(c) Deviance reduction (null -> GLM)", pct_reduction))
print("-" * 74)
print("  (d) Fitted effect direction (expected: BonusMalus +, DrivAge -)")
print(
    "      {:<44}{:>22}".format(
        "BonusMalus  (raw coef = {:+.4f})".format(bm_coef), bm_sign
    )
)
print(
    "      {:<44}{:>22}".format(
        "DrivAge     (corr w/ pred = {:+.3f})".format(da_corr), da_sign
    )
)
print("=" * 74)
print()

# ----------------------------------------------------------------------------
# 9. Persist model and test-set predictions for later steps
# ----------------------------------------------------------------------------
joblib.dump(poisson_glm, MODEL_OUT)

pd.DataFrame(
    {
        "IDpol": df_test["IDpol"].values,
        "Exposure": w_test,
        "ClaimNb": df_test["ClaimNb"].values,
        "Frequency_actual": y_test,
        "Frequency_pred_glm": pred_glm,
        "Frequency_pred_null": pred_null,
    }
).to_csv(PRED_OUT, index=False)

print(f"Saved fitted model      -> {MODEL_OUT}")
print(f"Saved test predictions  -> {PRED_OUT}")
