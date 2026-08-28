"""
Step 8 - Causal layer, METHOD VALIDATION on a semi-synthetic outcome.

Works only on 05_eudirectlapse_demand.csv. We do NOT touch the real 'lapse'
column. Instead we plant a KNOWN price effect (THETA_TRUE = 0.15) in a synthetic
lapse outcome, deliberately confounded through prem_pure, then check that:
  - a NAIVE price-only regression is biased by the confounding, and
  - econml's LinearDML (double machine learning) recovers the true effect.

Treatment  T = log(prem_final), centred.
Controls   W = risk factors + prem_pure + prem_last + prem_market (NOT prem_final).
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from econml.dml import LinearDML

warnings.filterwarnings("ignore")

INPUT_CSV = "05_eudirectlapse_demand.csv"
THETA_TRUE = 0.15
SEED = 0

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)

# Column roles ---------------------------------------------------------------
# Numeric controls = numeric risk factors + the three premium references.
numeric_controls = [
    "polholder_age", "policy_age", "policy_nbcontract",
    "vehicl_age", "vehicl_agepurchase",
    "prem_pure", "prem_last", "prem_market",
]
# Categorical controls = categorical risk factors.
categorical_controls = [
    "polholder_BMCevol", "polholder_diffdriver", "polholder_gender",
    "polholder_job", "policy_caruse", "prem_freqperyear",
    "vehicl_garage", "vehicl_powerkw", "vehicl_region",
]
# prem_final is the TREATMENT source and is deliberately excluded from controls.
# The real 'lapse' column is not used at all in this step.

# ---------------------------------------------------------------------------
# 2. Treatment: centred log price
# ---------------------------------------------------------------------------
T_raw = np.log(df["prem_final"].values)
T = T_raw - T_raw.mean()                      # centred log price

# ---------------------------------------------------------------------------
# 3. Controls W: standardise numerics, one-hot categoricals
# ---------------------------------------------------------------------------
control_preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_controls),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_controls),
    ],
    remainder="drop",
)
W = control_preprocessor.fit_transform(df)
W = np.asarray(W.todense()) if hasattr(W, "todense") else np.asarray(W)

# ---------------------------------------------------------------------------
# 4. Semi-synthetic outcome with a planted, confounded price effect
# ---------------------------------------------------------------------------
prem_pure = df["prem_pure"].values
z_prem_pure = (prem_pure - prem_pure.mean()) / prem_pure.std()   # standardised

p = 0.18 + THETA_TRUE * T + 0.12 * z_prem_pure
p_clipped = np.clip(p, 0.02, 0.60)
clip_frac = np.mean((p < 0.02) | (p > 0.60))

rng = np.random.default_rng(SEED)
lapse_synth = rng.binomial(1, p_clipped)

# Confounding check: T and prem_pure must be clearly correlated.
corr_T_prempure = np.corrcoef(T, prem_pure)[0, 1]

# ---------------------------------------------------------------------------
# 5a. NAIVE estimate: outcome on treatment alone, no controls
# ---------------------------------------------------------------------------
naive = LinearRegression().fit(T.reshape(-1, 1), lapse_synth)
theta_naive = naive.coef_[0]

# ---------------------------------------------------------------------------
# 5b. DML estimate: LinearDML with GB nuisances, cross-fitting
# ---------------------------------------------------------------------------
dml = LinearDML(
    model_y=GradientBoostingRegressor(random_state=SEED),
    model_t=GradientBoostingRegressor(random_state=SEED),
    discrete_treatment=False,
    cv=5,
    random_state=SEED,
)
dml.fit(lapse_synth, T, X=None, W=W)
theta_dml = float(np.ravel(dml.const_marginal_effect(X=None))[0])
lb, ub = dml.const_marginal_effect_interval(X=None, alpha=0.05)
lb, ub = float(np.ravel(lb)[0]), float(np.ravel(ub)[0])
ci_covers = (lb <= THETA_TRUE <= ub)

# ---------------------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("  STEP 8 - DML METHOD VALIDATION on semi-synthetic lapse outcome")
print("=" * 78)
print(f"  Sample size                         : {len(df):,}")
print(f"  Treatment T                         : centred log(prem_final)  (SD={T.std():.4f})")
print(f"  Controls W                          : {W.shape[1]} columns "
      f"({len(numeric_controls)} numeric + {len(categorical_controls)} categorical one-hot)")
print(f"  Synthetic p clipped at [0.02,0.60]  : {clip_frac*100:.1f}% of rows hit a bound")
print(f"  Base lapse rate (mean lapse_synth)  : {lapse_synth.mean():.4f}")
print("-" * 78)
print(f"  CONFOUNDING CHECK  corr(T, prem_pure) = {corr_T_prempure:+.4f}", end="")
print("   (clearly non-zero -> confounding present)"
      if abs(corr_T_prempure) > 0.05 else "   (WARNING: weak)")
print("=" * 78)

print()
print("  ESTIMATES vs TRUE price effect THETA_TRUE = {:.4f}".format(THETA_TRUE))
print("  " + "-" * 74)
print("  {:<8}{:>12}{:>24}{:>16}{:>12}".format(
    "Method", "Estimate", "95% CI", "Error vs TRUE", "CI covers?"))
print("  " + "-" * 74)
print("  {:<8}{:>12.4f}{:>24}{:>16}{:>12}".format("TRUE", THETA_TRUE, "-", "-", "-"))
print("  {:<8}{:>12.4f}{:>24}{:>+16.4f}{:>12}".format(
    "NAIVE", theta_naive, "(no CI)", theta_naive - THETA_TRUE, "n/a"))
print("  {:<8}{:>12.4f}{:>24}{:>+16.4f}{:>12}".format(
    "DML", theta_dml, "[{:.4f}, {:.4f}]".format(lb, ub),
    theta_dml - THETA_TRUE, "YES" if ci_covers else "NO"))
print("  " + "-" * 74)
print()
