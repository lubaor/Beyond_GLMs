"""
Step 3 - Machine-learning challenger on the French freMTPL2 frequency data.

Keeps the EXACT step-2 setup (same 90/10 split, seed=0, ClaimNb capped at 4,
Frequency = ClaimNb/Exposure, Exposure as sample weight, same mean-Poisson-
deviance scoring) and adds a gradient-boosted-trees model as a challenger to the
step-2 Poisson GLM.

Only touches 01_freMTPL2_freq.csv (+ the GLM predictions saved in step 2).
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

INPUT_CSV = "01_freMTPL2_freq.csv"
GLM_PRED = "02_glm_frequency_test_predictions.csv"   # reuse step-2 predictions
GBM_MODEL_OUT = "03_gbm_frequency_model.joblib"
GBM_PRED_OUT = "03_gbm_frequency_test_predictions.csv"
RANDOM_SEED = 0

# ----------------------------------------------------------------------------
# 1. Load + identical preprocessing / target as step 2
# ----------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
df["Exposure"] = df["Exposure"].clip(upper=1)
df["Frequency"] = df["ClaimNb"] / df["Exposure"]

# ----------------------------------------------------------------------------
# 2. Identical train / test split (same 90/10, same seed)
# ----------------------------------------------------------------------------
df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)

y_test = df_test["Frequency"].values
w_test = df_test["Exposure"].values
observed_claims = df_test["ClaimNb"].sum()   # total observed claims on test

# ----------------------------------------------------------------------------
# 3. Reuse step-2 GLM (and null) predictions on the identical test set
# ----------------------------------------------------------------------------
glm_df = pd.read_csv(GLM_PRED)
assert np.array_equal(df_test["IDpol"].values, glm_df["IDpol"].values), (
    "Test-set mismatch: step-2 predictions are not aligned with this split."
)
pred_null = glm_df["Frequency_pred_null"].values
pred_glm = glm_df["Frequency_pred_glm"].values

# ----------------------------------------------------------------------------
# 4. Gradient-boosted trees challenger (Poisson loss, native categoricals)
#    Trees need no one-hot / scaling: ordinal-encode categoricals, pass numerics
#    through as-is, and tell the booster which columns are categorical.
# ----------------------------------------------------------------------------
categorical_cols = ["VehBrand", "VehPower", "VehGas", "Region", "Area"]
numeric_cols = ["VehAge", "DrivAge", "BonusMalus", "Density"]

tree_preprocessor = ColumnTransformer(
    [
        (
            "categorical",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            categorical_cols,
        ),
        ("numeric", "passthrough", numeric_cols),
    ],
    remainder="drop",
)

# After the transformer the first 5 columns are the categoricals.
gbm = Pipeline(
    [
        ("preprocessor", tree_preprocessor),
        (
            "regressor",
            HistGradientBoostingRegressor(
                loss="poisson",
                categorical_features=[0, 1, 2, 3, 4],
                random_state=RANDOM_SEED,
            ),
        ),
    ]
)

gbm.fit(df_train, df_train["Frequency"], regressor__sample_weight=df_train["Exposure"])
pred_gbm = gbm.predict(df_test)

# ----------------------------------------------------------------------------
# 5. Score all three models with the SAME metric as step 2
# ----------------------------------------------------------------------------
def scorecard(pred):
    dev = mean_poisson_deviance(y_test, pred, sample_weight=w_test)
    predicted_claims = np.sum(pred * w_test)          # freq * exposure = claims
    calib = predicted_claims / observed_claims
    return dev, calib

dev_null, calib_null = scorecard(pred_null)
dev_glm, calib_glm = scorecard(pred_glm)
dev_gbm, calib_gbm = scorecard(pred_gbm)

def pct_vs_null(dev):
    return 100.0 * (1.0 - dev / dev_null)

# ----------------------------------------------------------------------------
# 6. One labelled comparison table
# ----------------------------------------------------------------------------
print()
print("=" * 74)
print("  FREQUENCY MODEL COMPARISON  -  out-of-sample (freMTPL2freq test set)")
print("  Test set: {:,} policies (10% hold-out, seed={})".format(len(df_test), RANDOM_SEED))
print("=" * 74)
print("  {:<22}{:>16}{:>18}{:>14}".format(
    "Model", "Poisson dev.", "% red. vs null", "Calib. ratio"))
print("-" * 74)
for name, dev, calib in [
    ("Null (intercept only)", dev_null, calib_null),
    ("GLM (step 2)", dev_glm, calib_glm),
    ("GBM (HistGBR, step 3)", dev_gbm, calib_gbm),
]:
    print("  {:<22}{:>16.6f}{:>17.2f}%{:>14.4f}".format(
        name, dev, pct_vs_null(dev), calib))
print("-" * 74)
print("  (a) Poisson dev. = out-of-sample mean Poisson deviance (lower is better)")
print("  (b) % red. vs null = deviance reduction relative to the null model")
print("  (c) Calib. ratio = total predicted claims / total observed claims (~1.00 ideal)")
print("=" * 74)
print()

# ----------------------------------------------------------------------------
# 7. Persist GBM model + test-set predictions (like the GLM)
# ----------------------------------------------------------------------------
joblib.dump(gbm, GBM_MODEL_OUT)
pd.DataFrame(
    {
        "IDpol": df_test["IDpol"].values,
        "Exposure": w_test,
        "ClaimNb": df_test["ClaimNb"].values,
        "Frequency_actual": y_test,
        "Frequency_pred_gbm": pred_gbm,
    }
).to_csv(GBM_PRED_OUT, index=False)

print(f"Saved fitted model      -> {GBM_MODEL_OUT}")
print(f"Saved test predictions  -> {GBM_PRED_OUT}")
