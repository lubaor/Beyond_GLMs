"""
What did the exposure cap cost us on the Swedish book?

The frozen harness clips Exposure at 1 for every portfolio. On freMTPL2 that
is the correct fix for a known quirk (0.2% of rows, 0.0% of exposure). The
Swedish data is aggregated over 1994 to 1998, so a row can hold up to 31
bike-years, and the same rule discards 32% of all policy-years while leaving
claim counts untouched.

This runs the Swedish book both ways, same folds, same seeds, same models.

Deviance is NOT comparable between the two regimes: capping changes the target
(Frequency = ClaimNb / Exposure) and the weights, so the two runs are scoring
different quantities. Gini and the decile calibration ratio are comparable in
the sense that matters here: they say how well each model ranks and calibrates
against whichever version of the truth it was given.

What it found, on the run committed alongside this file:

  Gini            capped            uncapped
  GLM             0.5528 +/-0.0636  0.5679 +/-0.0692
  GBM untuned     0.5336 +/-0.0392  0.5455 +/-0.0478
  GBM tuned       0.5526 +/-0.0611  0.5535 +/-0.0644

  worst decile    capped            uncapped
  GLM             1.4049 +/-1.4668  1.5473 +/-0.8046
  GBM untuned     1.8734 +/-1.7575  1.6371 +/-0.5623
  GBM tuned       1.3453 +/-1.2494  1.3526 +/-1.3343

The wash survives: uncapped the GLM is nominally ahead by 0.014 rather than
level, which is 0.2 of a fold SD, and an ordering that flips under a
preprocessing change is not an ordering. Slide 6's verdict is unaffected.

Calibration is the interesting one. The LEVEL of miscalibration barely moves,
1.35 to 1.65 either way, so the worst decile is still out by 135 to 165% with
every policy-year restored. What the cap was inflating is the VARIANCE: remove
it and the fold-to-fold SD falls 45% for the GLM and 68% for the untuned GBM.
So the broken diagnostic is a sparsity problem, as step 06 claims, and the cap
was making it noisier on top rather than causing it.

Validation: the capped GLM reproduces 06/07 exactly at 0.5528 +/-0.0636. The
capped tuned GBM gives 0.5526 against 07's 0.5564, because 07 routes exposure
weights into the inner scoring objective and this script does not. Treat the
GLM comparison as exact and the GBM one as close but not identical.
"""

import os, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance, make_scorer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder

warnings.filterwarnings("ignore")

CSV = "03_wasa_motorcycle.csv"
OUT = "results"
SEED, N_OUTER, N_INNER, N_ITER = 0, 5, 3, 40
NUM = ["OwnerAge", "VehAge"]
CAT = ["Gender", "Area", "RiskClass", "BonusClass"]


def ordered_gini(pred, w, cnt):
    o = np.argsort(pred, kind="mergesort"); wo, co = w[o], cnt[o]
    x = np.concatenate([[0.0], np.cumsum(wo) / wo.sum()])
    y = np.concatenate([[0.0], np.cumsum(co) / co.sum()])
    return 1.0 - 2.0 * np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0)


def norm_gini(pred, obs, w, cnt):
    return ordered_gini(pred, w, cnt) / ordered_gini(obs, w, cnt)


def decile_ratios(pred, w, cnt, n_bins=10):
    o = np.argsort(pred, kind="mergesort"); p, ww, cc = pred[o], w[o], cnt[o]
    frac = (np.cumsum(ww) - 0.5 * ww) / ww.sum()
    dec = np.clip((frac * n_bins).astype(int), 0, n_bins - 1)
    out = np.full(n_bins, np.nan)
    for d in range(n_bins):
        m = dec == d; sw, oc = ww[m].sum(), cc[m].sum()
        if oc > 0:
            out[d] = (np.sum(ww[m] * p[m]) / sw) / (oc / sw)
    return out


def build_glm():
    pre = ColumnTransformer([
        ("binned", KBinsDiscretizer(n_bins=10, random_state=SEED), NUM),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), CAT),
    ], remainder="drop")
    return Pipeline([("pre", pre),
                     ("reg", PoissonRegressor(alpha=1e-12, solver="newton-cholesky",
                                              max_iter=300))])


def build_gbm():
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
        ("num", "passthrough", NUM),
    ], remainder="drop")
    return Pipeline([("pre", pre),
                     ("reg", HistGradientBoostingRegressor(
                         loss="poisson", categorical_features=[0, 1, 2, 3],
                         random_state=SEED))])


scorer = make_scorer(mean_poisson_deviance, greater_is_better=False)
PARAM_DIST = {
    "reg__learning_rate": uniform(0.01, 0.29),
    "reg__max_leaf_nodes": randint(8, 64),
    "reg__min_samples_leaf": randint(20, 2001),
    "reg__l2_regularization": uniform(0.0, 10.0),
}

os.makedirs(OUT, exist_ok=True)
raw = pd.read_csv(CSV)
raw = raw[raw["Exposure"] > 0].reset_index(drop=True)
print(f"rows {len(raw):,}   total policy-years {raw.Exposure.sum():,.0f}", flush=True)

results = {}
for regime in ("capped", "uncapped"):
    df = raw.copy()
    if regime == "capped":
        df["Exposure"] = df["Exposure"].clip(upper=1)
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    y, w, cnt = df["Frequency"].values, df["Exposure"].values, df["ClaimNb"].values
    print(f"\n=== {regime.upper()}   policy-years {w.sum():,.0f}   "
          f"mean freq {cnt.sum()/w.sum():.4f} ===", flush=True)

    kf = KFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    acc = {m: {"gini": [], "cal": []} for m in ("GLM", "GBM_untuned", "GBM_tuned")}
    t0 = time.time()
    for fold, (tr, te) in enumerate(kf.split(df), start=1):
        d_tr, d_te = df.iloc[tr], df.iloc[te]
        w_te, c_te, y_te = w[te], cnt[te], y[te]

        for name in ("GLM", "GBM_untuned", "GBM_tuned"):
            if name == "GLM":
                m = build_glm()
                m.fit(d_tr, y[tr], reg__sample_weight=w[tr])
                pred = m.predict(d_te)
            elif name == "GBM_untuned":
                m = build_gbm()
                m.fit(d_tr, y[tr], reg__sample_weight=w[tr])
                pred = m.predict(d_te)
            else:
                base = build_gbm()
                base.named_steps["reg"].set_params(early_stopping=True)
                srch = RandomizedSearchCV(base, PARAM_DIST, n_iter=N_ITER, scoring=scorer,
                                          cv=KFold(N_INNER, shuffle=True, random_state=SEED),
                                          random_state=SEED, refit=True, n_jobs=4)
                srch.fit(d_tr, y[tr], reg__sample_weight=w[tr])
                pred = srch.best_estimator_.predict(d_te)
            acc[name]["gini"].append(norm_gini(pred, y_te, w_te, c_te))
            acc[name]["cal"].append(np.nanmax(np.abs(decile_ratios(pred, w_te, c_te) - 1.0)))
        print(f"  fold {fold} done [{time.time()-t0:.0f}s]", flush=True)
    results[regime] = acc

print("\n" + "=" * 74, flush=True)
print("  Swedish Wasa motorcycle: exposure capped at 1 vs left alone", flush=True)
print("=" * 74, flush=True)
print(f"  {'model':14} {'Gini capped':>20} {'Gini uncapped':>20}", flush=True)
for m in ("GLM", "GBM_untuned", "GBM_tuned"):
    a = np.array(results["capped"][m]["gini"]); b = np.array(results["uncapped"][m]["gini"])
    print(f"  {m:14} {a.mean():>10.4f} +/-{a.std(ddof=1):.4f} {b.mean():>10.4f} +/-{b.std(ddof=1):.4f}", flush=True)
print(f"\n  {'model':14} {'worst-decile capped':>24} {'uncapped':>20}", flush=True)
for m in ("GLM", "GBM_untuned", "GBM_tuned"):
    a = np.array(results["capped"][m]["cal"]); b = np.array(results["uncapped"][m]["cal"])
    print(f"  {m:14} {a.mean():>14.4f} +/-{a.std(ddof=1):.4f} {b.mean():>10.4f} +/-{b.std(ddof=1):.4f}", flush=True)

rows = []
for regime in results:
    for m in results[regime]:
        g = np.array(results[regime][m]["gini"]); c = np.array(results[regime][m]["cal"])
        rows.append({"regime": regime, "model": m, "gini": g.mean(), "gini_sd": g.std(ddof=1),
                     "cal": c.mean(), "cal_sd": c.std(ddof=1)})
pd.DataFrame(rows).to_csv(f"{OUT}/07g_swedish_exposure_cap.csv", index=False)
print("\nwrote results/07g_swedish_exposure_cap.csv", flush=True)
