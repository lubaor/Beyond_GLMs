"""
Step 9 - REAL renewal price elasticity via Double Machine Learning.

Dataset 05_eudirectlapse_demand.csv, causal-layer environment (see requirements.txt).
Now the REAL 'lapse' column is the outcome.

  Outcome   Y = lapse (0/1)
  Treatment T = centred log(prem_final)           (the offered renewal price)
  Controls  W = risk factors + prem_pure + prem_last + prem_market
              (categoricals categorical, numerics standardised; prem_final NOT in W)

Produces: (1) DML point estimate + business translation + naive comparison,
(2) overlap/positivity check on the residualised treatment, (3) sensitivity to
unobserved confounding (robustness value), (4) light heterogeneity via causal forest.
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from econml.dml import LinearDML, CausalForestDML

warnings.filterwarnings("ignore")

INPUT_CSV = "05_eudirectlapse_demand.csv"
SEED = 0

def gbr():
    return GradientBoostingRegressor(random_state=SEED)

def dlogp(pct):                       # log-price change for a +pct% price move
    return np.log(1.0 + pct / 100.0)

# ---------------------------------------------------------------------------
# Load + build Y, T, W
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
    [("num", StandardScaler(), numeric_controls),
     ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_controls)],
    remainder="drop",
)
W = control_preprocessor.fit_transform(df)
W = np.asarray(W.todense()) if hasattr(W, "todense") else np.asarray(W)

Y = df["lapse"].values.astype(float)
logprice = np.log(df["prem_final"].values)
T = logprice - logprice.mean()
base_lapse = Y.mean()

print("=" * 82)
print("  STEP 9 - REAL renewal price elasticity (DML)")
print("=" * 82)
print(f"  Sample size            : {n:,}")
print(f"  Observed lapse rate    : {base_lapse:.4f}")
print(f"  Controls W             : {W.shape[1]} columns")
print(f"  Raw treatment SD (logprice, centred): {T.std():.4f}")

# ===========================================================================
# 1) POINT ESTIMATE
# ===========================================================================
dml = LinearDML(model_y=gbr(), model_t=gbr(),
                discrete_treatment=False, cv=5, random_state=SEED)
dml.fit(Y, T, X=None, W=W, cache_values=True)
theta = float(np.ravel(dml.const_marginal_effect(X=None))[0])
lb, ub = dml.const_marginal_effect_interval(X=None, alpha=0.05)
lb, ub = float(np.ravel(lb)[0]), float(np.ravel(ub)[0])

theta_naive = LinearRegression().fit(T.reshape(-1, 1), Y).coef_[0]

print()
print("-" * 82)
print("  1) POINT ESTIMATE  -  price effect on P(lapse), per unit of log-price")
print("-" * 82)
print(f"     DML  theta = {theta:+.4f}   95% CI [{lb:+.4f}, {ub:+.4f}]")
print(f"     NAIVE theta = {theta_naive:+.4f}   (price only, no controls)")
print(f"     Confounding correction moved the estimate by "
      f"{theta - theta_naive:+.4f} (naive -> DML)")
print()
print("     Business translation (change in lapse probability, percentage points):")
print("     {:<26}{:>16}{:>16}{:>16}".format("price move", "DML", "DML 95% CI", "NAIVE"))
for pct in (1, 10):
    d = dlogp(pct)
    print("     {:<26}{:>15.3f}%{:>7.2f}..{:<6.2f}{:>15.3f}%".format(
        f"+{pct}% renewal price",
        theta * d * 100, lb * d * 100, ub * d * 100, theta_naive * d * 100))

# ===========================================================================
# 2) OVERLAP / POSITIVITY CHECK
# ===========================================================================
# Residualised treatment (T after partialling out W) from DML's first stage.
T_res = np.ravel(dml.residuals_[1])
sd_res = T_res.std()
p5, p95 = np.percentile(T_res, [5, 95])
var_frac = np.var(T_res) / np.var(T)      # fraction of price variance that is independent
enough = sd_res > 0.05                     # heuristic: some meaningful independent variation

print()
print("-" * 82)
print("  2) OVERLAP / POSITIVITY  -  independent price variation after controls")
print("-" * 82)
print(f"     Residualised treatment SD          : {sd_res:.4f}  (raw {T.std():.4f})")
print(f"     Residualised 5th-95th pct range    : [{p5:+.4f}, {p95:+.4f}]")
print(f"     Fraction of price variance that is independent of W: {var_frac*100:.1f}%")
print("     Verdict: {}".format(
    "enough independent price variation to identify the effect."
    if enough else
    "WARNING - very little independent price variation; effect weakly identified."))

# ===========================================================================
# 3) SENSITIVITY TO UNOBSERVED CONFOUNDING
# ===========================================================================
print()
print("-" * 82)
print("  3) SENSITIVITY TO UNOBSERVED CONFOUNDING")
print("-" * 82)
rv_point = float(dml.robustness_value(alpha=1.0))      # alpha=1 -> point estimate to 0
rv_ci = float(dml.robustness_value(alpha=0.05))        # to bring 95% CI to include 0
print(f"     Robustness value (point estimate -> 0) : {rv_point:.4f}")
print(f"     Robustness value (95% CI touches 0)    : {rv_ci:.4f}")
print("     Reading: an unobserved confounder would have to explain about "
      f"{rv_point*100:.1f}% of the")
print("     residual variation in BOTH price and lapse to nullify the point estimate.")
try:
    print()
    print(dml.sensitivity_summary(alpha=0.05))
except Exception as e:
    print("     (sensitivity_summary unavailable:", repr(e)[:120], ")")

# ===========================================================================
# 4) HETEROGENEITY, LIGHT  (causal forest)
# ===========================================================================
# Let the forest use the controls as effect modifiers (X) as well as confounders.
cf = CausalForestDML(model_y=gbr(), model_t=gbr(),
                     discrete_treatment=False, cv=5, random_state=SEED)
cf.fit(Y, T, X=W, W=None)
cate = np.ravel(cf.const_marginal_effect(W))     # per-policy price effect

q = np.quantile(cate, [0.25, 0.75])
least = cate[cate <= q[0]].mean()                # least-elastic quartile
most = cate[cate >= q[1]].mean()                 # most-elastic quartile

print()
print("-" * 82)
print("  4) HETEROGENEITY (CausalForestDML) - does price sensitivity vary?")
print("-" * 82)
print(f"     Per-policy price effect: mean {cate.mean():+.4f}, SD {cate.std():.4f}, "
      f"range [{cate.min():+.4f}, {cate.max():+.4f}]")
print(f"     Least-elastic quartile mean effect : {least:+.4f}   "
      f"(+10% price -> {least*dlogp(10)*100:+.2f} pp lapse)")
print(f"     Most-elastic  quartile mean effect : {most:+.4f}   "
      f"(+10% price -> {most*dlogp(10)*100:+.2f} pp lapse)")
print(f"     Spread most vs least               : {most - least:+.4f}")
print("=" * 82)
