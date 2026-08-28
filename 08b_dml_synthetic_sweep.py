"""
Step 8b - DML method validation with a CONFOUNDING-STRENGTH SWEEP.

Same dataset (05_eudirectlapse_demand.csv), same purpose: recover a KNOWN planted
price effect (THETA_TRUE = 0.15) on a semi-synthetic Bernoulli lapse outcome. The
real 'lapse' column is still not used.

Fix vs step 8: in step 8 the treatment log(prem_final) had corr 0.93 with prem_pure
(near-collinear), leaving almost no independent price variation, so DML could only
return a very wide interval. Here we CONSTRUCT the treatment as a confounded part
(loading RHO on standardised prem_pure) plus independent Gaussian noise (SD 0.25)
representing loadings/competition/rounding, and sweep RHO across weak/medium/strong
confounding regimes.

Construction note: the base term log(prem_pure) named in the brief is a deterministic
function of prem_pure and by itself forces corr(T,prem_pure) ~ 0.85+ regardless of
RHO, which would defeat the sweep. It is therefore omitted; T is centred anyway, and
because the outcome is generated from this same T with THETA_TRUE=0.15, the known
truth is exactly 0.15 in every regime. The guaranteed 0.25-SD independent noise is
what lets DML identify the effect even under strong confounding.
"""

import warnings
import numpy as np
import pandas as pd
import sklearn
import econml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from econml.dml import LinearDML

warnings.filterwarnings("ignore")

INPUT_CSV = "05_eudirectlapse_demand.csv"
REQ_OUT = "requirements.txt"
THETA_TRUE = 0.15
NOISE_SD = 0.25
SEED = 0

# ---------------------------------------------------------------------------
# 0. Reproducibility housekeeping
# ---------------------------------------------------------------------------
VERS = {
    "scikit-learn": sklearn.__version__,
    "econml": econml.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
}
print("=" * 78)
print("  STEP 8b - installed versions (causal-layer environment)")
print("=" * 78)
for k, v in VERS.items():
    print(f"  {k:<14} {v}")
print("-" * 78)
print("  REPRODUCIBILITY SPLIT (documented): benchmark steps 2-7 were produced under")
print("  scikit-learn 1.9.0; installing econml pinned scikit-learn down to 1.6.1, so")
print("  the causal steps (8+) run under 1.6.1. This split is intentional and pinned")
print(f"  in {REQ_OUT}.")
print("=" * 78)

with open(REQ_OUT, "w") as f:
    f.write("# Causal-layer environment (benchmark steps 8+).\n")
    f.write("# NOTE: steps 2-7 were produced under scikit-learn==1.9.0; installing\n")
    f.write("# econml pinned scikit-learn down to 1.6.1 for the causal steps. This\n")
    f.write("# version split is intentional and documented here.\n")
    for k, v in VERS.items():
        f.write(f"{k}=={v}\n")

# ---------------------------------------------------------------------------
# 1. Load + control columns (same roles as step 8)
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
n = len(df)

numeric_controls = [
    "polholder_age", "policy_age", "policy_nbcontract",
    "vehicl_age", "vehicl_agepurchase",
    "prem_pure", "prem_last", "prem_market",
]
categorical_controls = [
    "polholder_BMCevol", "polholder_diffdriver", "polholder_gender",
    "polholder_job", "policy_caruse", "prem_freqperyear",
    "vehicl_garage", "vehicl_powerkw", "vehicl_region",
]

control_preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_controls),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_controls),
    ],
    remainder="drop",
)
W = control_preprocessor.fit_transform(df)
W = np.asarray(W.todense()) if hasattr(W, "todense") else np.asarray(W)

# Standardised pure premium: the confounder (appears in the outcome and in W).
prem_pure = df["prem_pure"].values
z_pp = (prem_pure - prem_pure.mean()) / prem_pure.std()

# ---------------------------------------------------------------------------
# 2. Confounding sweep
#    T = RHO * z_pp + noise ;  corr(T,prem_pure) = RHO / sqrt(RHO^2 + NOISE_SD^2).
#    Solve RHO for each target correlation.
# ---------------------------------------------------------------------------
def rho_for_corr(target):
    return NOISE_SD * target / np.sqrt(1.0 - target ** 2)

REGIMES = [
    ("WEAK",   0.30),
    ("MEDIUM", 0.60),
    ("STRONG", 0.90),
]

rng = np.random.default_rng(SEED)
rows = []

for label, target_corr in REGIMES:
    rho = rho_for_corr(target_corr)

    noise = rng.normal(0.0, NOISE_SD, n)
    logprice = rho * z_pp + noise          # confounded part + independent variation
    T = logprice - logprice.mean()         # centred treatment
    corr_T_pp = np.corrcoef(T, prem_pure)[0, 1]

    p = np.clip(0.18 + THETA_TRUE * T + 0.12 * z_pp, 0.02, 0.60)
    y = rng.binomial(1, p)

    # NAIVE: outcome on treatment only, no controls.
    theta_naive = LinearRegression().fit(T.reshape(-1, 1), y).coef_[0]

    # DML: LinearDML, GB nuisances, cross-fitting.
    dml = LinearDML(
        model_y=GradientBoostingRegressor(random_state=SEED),
        model_t=GradientBoostingRegressor(random_state=SEED),
        discrete_treatment=False, cv=5, random_state=SEED,
    )
    dml.fit(y, T, X=None, W=W)
    theta_dml = float(np.ravel(dml.const_marginal_effect(X=None))[0])
    lb, ub = dml.const_marginal_effect_interval(X=None, alpha=0.05)
    lb, ub = float(np.ravel(lb)[0]), float(np.ravel(ub)[0])

    rows.append({
        "regime": label, "rho": rho, "corr": corr_T_pp,
        "naive": theta_naive, "dml": theta_dml, "lb": lb, "ub": ub,
        "covers": lb <= THETA_TRUE <= ub,
    })

# ---------------------------------------------------------------------------
# 3. Report table
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print("  CONFOUNDING SWEEP  -  price-effect recovery vs TRUE THETA = {:.2f}".format(THETA_TRUE))
print("  T = RHO*std(prem_pure) + N(0,{:.2f}) ; controls W = risk factors + prem_pure/last/market"
      .format(NOISE_SD))
print("=" * 90)
print("  {:<8}{:>8}{:>10}{:>12}{:>26}{:>12}".format(
    "regime", "RHO", "corr", "NAIVE est", "DML est [95% CI]", "covers 0.15?"))
print("-" * 90)
for r in rows:
    print("  {:<8}{:>8.3f}{:>10.3f}{:>12.4f}{:>26}{:>12}".format(
        r["regime"], r["rho"], r["corr"], r["naive"],
        "{:.4f} [{:.4f}, {:.4f}]".format(r["dml"], r["lb"], r["ub"]),
        "YES" if r["covers"] else "NO"))
print("-" * 90)
print("  NAIVE = outcome on price alone (no controls); DML = LinearDML w/ GB nuisances, 5-fold CF.")
print("  CI width shrinks / stays usable because the treatment always carries independent")
print("  variation (the SD={:.2f} noise), unlike step 8 where price was near-collinear.".format(NOISE_SD))
print("=" * 90)
print()
print(f"  Wrote pinned versions to {REQ_OUT}")
