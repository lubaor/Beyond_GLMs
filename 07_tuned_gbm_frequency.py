"""
Step 7 - Nested-CV tuned GBM vs untuned GLM, across three portfolios.

Same frozen outer protocol as steps 5-6 (5 outer folds, seed=0, exposure
weighting, Exposure capped at 1, mean-Poisson-deviance scoring, exposure-weighted
10-decile calibration, normalised ordered Gini). The ONLY change vs step 6: the
GBM is tuned by an inner 3-fold randomised search inside each outer training fold
(no leakage). The GLM is unchanged from steps 2/5 (freMTPL2) and step 6
(Australian, Swedish).

categorical_features are passed to the GBM BY COLUMN NAME (the true rating
factors of each book), using HistGradientBoostingRegressor's native categorical
support. Exposure-weighted Poisson deviance is used for the inner scoring too,
routed via scikit-learn metadata routing so the weights reach both fit and score.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from scipy.stats import uniform, randint

from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import make_scorer, mean_poisson_deviance
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer, KBinsDiscretizer, OneHotEncoder, OrdinalEncoder, StandardScaler,
)

set_config(enable_metadata_routing=True)          # route sample_weight to fit + scorer
warnings.filterwarnings("ignore", category=LinAlgWarning)

SMOKE = os.environ.get("SMOKE") == "1"
RANDOM_SEED = 0
N_OUTER = 5
N_INNER = 3
N_ITER = 5 if SMOKE else 40
N_JOBS = 1 if SMOKE else min(8, (os.cpu_count() or 2))
AREA_ORDER = ["A", "B", "C", "D", "E", "F"]

PORTFOLIOS = [
    {
        "name": "freMTPL2freq (French motor)",
        "file": "01_freMTPL2_freq.csv",
        "cap_claimnb": 4,
        "glm": {"passthrough": ["BonusMalus"], "binned": ["VehAge", "DrivAge"],
                "density": ["Density"], "ordinal_area": ["Area"],
                "onehot": ["VehBrand", "VehPower", "VehGas", "Region"]},
        "gbm": {"numeric": ["VehAge", "DrivAge", "BonusMalus", "Density"],
                "categorical": ["VehBrand", "VehPower", "VehGas", "Region", "Area"]},
    },
    {
        "name": "Australian (ausprivauto)",
        "file": "04_ausprivauto_dejong.csv",
        "cap_claimnb": None,
        "glm": {"passthrough": [], "binned": ["VehValue"], "density": [],
                "ordinal_area": [], "onehot": ["VehAge", "VehBody", "Gender", "DrivAge"]},
        "gbm": {"numeric": ["VehValue"],
                "categorical": ["VehAge", "VehBody", "Gender", "DrivAge"]},
    },
    {
        "name": "Swedish Wasa motorcycle",
        "file": "03_wasa_motorcycle.csv",
        "cap_claimnb": None,
        "glm": {"passthrough": [], "binned": ["OwnerAge", "VehAge"], "density": [],
                "ordinal_area": [], "onehot": ["Gender", "Area", "RiskClass", "BonusClass"]},
        "gbm": {"numeric": ["OwnerAge", "VehAge"],
                "categorical": ["Gender", "Area", "RiskClass", "BonusClass"]},
    },
]

# ---------------------------------------------------------------------------
# GLM (unchanged baseline) - portfolio-specific, as in steps 2/5 and 6
# ---------------------------------------------------------------------------
def build_glm(g):
    ts = []
    if g["passthrough"]:
        ts.append(("passthrough_numeric", "passthrough", g["passthrough"]))
    if g["binned"]:
        # bins computed unweighted (as in steps 2/5/6): opt out of sample_weight
        kbd = KBinsDiscretizer(n_bins=10,
                               random_state=RANDOM_SEED).set_fit_request(sample_weight=False)
        ts.append(("binned_numeric", kbd, g["binned"]))
    if g["density"]:
        scaler = StandardScaler().set_fit_request(sample_weight=False)
        ts.append(("log_scaled",
                   make_pipeline(FunctionTransformer(np.log, validate=False), scaler),
                   g["density"]))
    if g["ordinal_area"]:
        ts.append(("ordinal_area",
                   OrdinalEncoder(categories=[AREA_ORDER]), g["ordinal_area"]))
    if g["onehot"]:
        ts.append(("onehot_categorical",
                   OneHotEncoder(handle_unknown="ignore"), g["onehot"]))
    pre = ColumnTransformer(ts, remainder="drop")
    reg = PoissonRegressor(alpha=1e-12, solver="newton-cholesky",
                           max_iter=300).set_fit_request(sample_weight=True)
    return Pipeline([("preprocessor", pre), ("regressor", reg)])

# ---------------------------------------------------------------------------
# Scoring helpers - IDENTICAL to steps 4-6
# ---------------------------------------------------------------------------
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
    order = np.argsort(pred_freq, kind="mergesort")
    p, ww, cc = pred_freq[order], w[order], claim_cnt[order]
    frac = (np.cumsum(ww) - 0.5 * ww) / ww.sum()
    dec = np.clip((frac * n_bins).astype(int), 0, n_bins - 1)
    ratios = np.full(n_bins, np.nan)
    for d in range(n_bins):
        m = dec == d
        sw = ww[m].sum()
        obs_claims = cc[m].sum()
        if obs_claims > 0:
            ratios[d] = (np.sum(ww[m] * p[m]) / sw) / (obs_claims / sw)
    return ratios

# ---------------------------------------------------------------------------
# Tuning search: exposure-weighted Poisson deviance, routed to fit AND score
# ---------------------------------------------------------------------------
poisson_scorer = make_scorer(
    mean_poisson_deviance, greater_is_better=False).set_score_request(sample_weight=True)

PARAM_DIST = {
    "learning_rate": uniform(0.01, 0.29),      # -> [0.01, 0.30]
    "max_leaf_nodes": randint(8, 64),          # -> [8, 63]
    "min_samples_leaf": randint(20, 2001),     # -> [20, 2000]
    "l2_regularization": uniform(0.0, 10.0),   # -> [0, 10]
}


def build_search(cat_names):
    base = HistGradientBoostingRegressor(
        loss="poisson", categorical_features=cat_names,
        early_stopping=True, random_state=RANDOM_SEED,
    ).set_fit_request(sample_weight=True)
    inner_cv = KFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_SEED)
    return RandomizedSearchCV(
        base, PARAM_DIST, n_iter=N_ITER, scoring=poisson_scorer,
        cv=inner_cv, random_state=RANDOM_SEED, refit=True, n_jobs=N_JOBS,
    )

# ---------------------------------------------------------------------------
# Run one portfolio
# ---------------------------------------------------------------------------
def run_portfolio(cfg):
    df = pd.read_csv(cfg["file"])
    n0 = len(df)
    df = df[df["Exposure"] > 0].copy()
    dropped = n0 - len(df)
    if cfg["cap_claimnb"] is not None:
        df["ClaimNb"] = df["ClaimNb"].clip(upper=cfg["cap_claimnb"])
    df["Exposure"] = df["Exposure"].clip(upper=1)
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    df = df.reset_index(drop=True)
    if SMOKE:
        df = df.sample(n=min(4000, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)

    g, gb = cfg["glm"], cfg["gbm"]
    cat_names = gb["categorical"]
    feat_cols = gb["numeric"] + cat_names

    # GBM feature frame: categoricals -> category dtype so native handling works.
    X = df[feat_cols].copy()
    for c in cat_names:
        X[c] = X[c].astype("category")
    y = df["Frequency"].values
    w = df["Exposure"].values
    claims = df["ClaimNb"].values

    kf = KFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_SEED)
    metrics = {m: {"dev": [], "gini": [], "worstcal": []} for m in ("GLM", "GBM")}
    best_params_per_fold = []
    empty_deciles = 0

    for tr, te in kf.split(df):
        y_te, w_te, c_te = y[te], w[te], claims[te]

        # --- GLM (unchanged) ---
        glm = build_glm(g)
        glm.fit(df.iloc[tr], y[tr], sample_weight=w[tr])
        pred_glm = glm.predict(df.iloc[te])

        # --- Tuned GBM (inner randomised search on the outer-train fold only) ---
        search = build_search(cat_names)
        search.fit(X.iloc[tr], y[tr], sample_weight=w[tr])
        pred_gbm = search.best_estimator_.predict(X.iloc[te])
        best_params_per_fold.append(search.best_params_)

        for name, pred in (("GLM", pred_glm), ("GBM", pred_gbm)):
            metrics[name]["dev"].append(
                mean_poisson_deviance(y_te, pred, sample_weight=w_te))
            metrics[name]["gini"].append(normalised_gini(pred, y_te, w_te, c_te))
            ratios = decile_ratios(pred, w_te, c_te)
            empty_deciles += int(np.isnan(ratios).sum())
            metrics[name]["worstcal"].append(np.nanmax(np.abs(ratios - 1.0)))

    return {"metrics": metrics, "best_params": best_params_per_fold,
            "dropped": dropped, "empty_deciles": empty_deciles,
            "cat_names": cat_names}


def ms(vals):
    a = np.array(vals, dtype=float)
    return np.nanmean(a), np.nanstd(a, ddof=1)


def report(cfg, res):
    m = res["metrics"]
    print()
    print("#" * 84)
    print(f"#  PORTFOLIO: {cfg['name']}   (file={cfg['file']})")
    print(f"#  rows dropped (Exposure<=0)={res['dropped']}")
    print(f"#  categorical_features (by name): {res['cat_names']}")
    print(f"#  numeric features:               {cfg['gbm']['numeric']}")
    print("#" * 84)

    print("\n  1) 5-FOLD SUMMARY  (mean +/- std, ddof=1)   GLM = untuned, GBM = tuned")
    print("  " + "-" * 76)
    print("  {:<6}{:>22}{:>22}{:>22}".format(
        "Model", "Poisson deviance", "Normalised Gini", "Worst-decile cal.err"))
    print("  " + "-" * 76)
    for name in ("GLM", "GBM"):
        dm, ds = ms(m[name]["dev"]); gm, gs = ms(m[name]["gini"]); cm, cs = ms(m[name]["worstcal"])
        print("  {:<6}{:>13.6f} +/-{:>6.6f}{:>13.4f} +/-{:>6.4f}{:>13.4f} +/-{:>6.4f}".format(
            name, dm, ds, gm, gs, cm, cs))
    print("  " + "-" * 76)
    if res["empty_deciles"]:
        print(f"  (note: {res['empty_deciles']} decile(s) had zero observed claims, skipped)")

    gap = np.array(m["GBM"]["gini"]) - np.array(m["GLM"]["gini"])
    cg, cl = np.array(m["GBM"]["worstcal"]), np.array(m["GLM"]["worstcal"])
    print("\n  2) Per-fold TUNED-GBM-minus-GLM normalised-Gini gap:")
    print("       " + ", ".join("{:+.4f}".format(x) for x in gap))
    print("       folds where tuned-GBM worst-decile cal.err > GLM's: {}/{}".format(
        int(np.sum(cg > cl)), N_OUTER))

    print("\n  3) Tuned GBM hyperparameters chosen per outer fold:")
    print("     {:>4}{:>16}{:>16}{:>18}{:>18}".format(
        "fold", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"))
    for i, bp in enumerate(res["best_params"], start=1):
        print("     {:>4}{:>16.4f}{:>16d}{:>18d}{:>18.3f}".format(
            i, bp["learning_rate"], int(bp["max_leaf_nodes"]),
            int(bp["min_samples_leaf"]), bp["l2_regularization"]))


if __name__ == "__main__":
    print(f"[config] SMOKE={SMOKE}  n_iter={N_ITER}  n_jobs={N_JOBS}  "
          f"outer={N_OUTER}  inner={N_INNER}")
    for cfg in PORTFOLIOS:
        report(cfg, run_portfolio(cfg))
    print()
