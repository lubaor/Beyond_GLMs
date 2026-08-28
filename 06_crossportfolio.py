"""
Step 6 - Cross-portfolio generalisation test.

Applies the EXACT frozen step-2..5 method (Poisson frequency GLM vs Poisson
HistGradientBoostingRegressor GBM, 5-fold CV, exposure weighting, mean-Poisson-
deviance scoring, exposure-weighted 10-decile calibration, normalised ordered
Gini) to two new motor portfolios, changing nothing about the method.

Only per-portfolio column mapping is adapted (each book has its own rating
factors; categoricals treated as categorical, numerics as numeric, Exposure
capped at 1, non-positive-exposure rows dropped because frequency is otherwise
undefined). Neither book has a numeric density-like variable, so no log-scaling
applies. ClaimNb is not capped: its maximum is already <= 4 in both books.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder

# GLM's near-unregularised full one-hot design is collinear; the fast solver
# auto-falls back to lbfgs (predictions unaffected). Silence that known notice.
warnings.filterwarnings("ignore", category=LinAlgWarning)

RANDOM_SEED = 0
N_SPLITS = 5

PORTFOLIOS = [
    {
        "name": "Australian (ausprivauto, De Jong & Heller)",
        "file": "04_ausprivauto_dejong.csv",
        "numeric": ["VehValue"],
        "categorical": ["VehAge", "VehBody", "Gender", "DrivAge"],
    },
    {
        "name": "Swedish Wasa motorcycle (Ohlsson & Johansson)",
        "file": "03_wasa_motorcycle.csv",
        "numeric": ["OwnerAge", "VehAge"],
        "categorical": ["Gender", "Area", "RiskClass", "BonusClass"],
    },
]

# ----------------------------------------------------------------------------
# Model builders - same structure as steps 2 (GLM) and 3 (GBM)
# ----------------------------------------------------------------------------
def build_glm(numeric, categorical):
    transformers = []
    if numeric:
        transformers.append(
            ("binned_numeric",
             KBinsDiscretizer(n_bins=10, random_state=RANDOM_SEED), numeric)
        )
    if categorical:
        transformers.append(
            ("onehot_categorical", OneHotEncoder(handle_unknown="ignore"), categorical)
        )
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline(
        [("preprocessor", pre),
         ("regressor",
          PoissonRegressor(alpha=1e-12, solver="newton-cholesky", max_iter=300))]
    )


def build_gbm(numeric, categorical):
    pre = ColumnTransformer(
        [("categorical",
          OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
          categorical),
         ("numeric", "passthrough", numeric)],
        remainder="drop",
    )
    cat_idx = list(range(len(categorical)))
    return Pipeline(
        [("preprocessor", pre),
         ("regressor",
          HistGradientBoostingRegressor(
              loss="poisson", categorical_features=cat_idx, random_state=RANDOM_SEED))]
    )

# ----------------------------------------------------------------------------
# Scoring helpers - IDENTICAL to steps 4-5
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
    """Exposure-weighted deciles; pred/obs per decile (NaN if a decile has no
    observed claims - happens in very-low-frequency books)."""
    order = np.argsort(pred_freq, kind="mergesort")
    p, ww, cc = pred_freq[order], w[order], claim_cnt[order]
    frac = (np.cumsum(ww) - 0.5 * ww) / ww.sum()
    dec = np.clip((frac * n_bins).astype(int), 0, n_bins - 1)
    ratios = np.full(n_bins, np.nan)
    for d in range(n_bins):
        m = dec == d
        sw = ww[m].sum()
        mean_pred = np.sum(ww[m] * p[m]) / sw
        obs_claims = cc[m].sum()
        if obs_claims > 0:
            ratios[d] = mean_pred / (obs_claims / sw)
    return ratios   # index 0 = bottom (lowest-risk) decile

# ----------------------------------------------------------------------------
# Run one portfolio
# ----------------------------------------------------------------------------
def run_portfolio(cfg):
    df = pd.read_csv(cfg["file"])
    n0 = len(df)
    df = df[df["Exposure"] > 0].copy()          # frequency undefined at 0 exposure
    dropped = n0 - len(df)
    df["Exposure"] = df["Exposure"].clip(upper=1)
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    df = df.reset_index(drop=True)

    numeric, categorical = cfg["numeric"], cfg["categorical"]
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    metrics = {m: {"dev": [], "gini": [], "worstcal": []} for m in ("GLM", "GBM")}
    gbm_bottom = []
    empty_deciles = 0

    for tr, te in kf.split(df):
        df_tr, df_te = df.iloc[tr], df.iloc[te]
        y_te = df_te["Frequency"].values
        w_te = df_te["Exposure"].values
        c_te = df_te["ClaimNb"].values

        for name, builder in (("GLM", build_glm), ("GBM", build_gbm)):
            model = builder(numeric, categorical)
            model.fit(df_tr, df_tr["Frequency"],
                      regressor__sample_weight=df_tr["Exposure"])
            pred = model.predict(df_te)

            metrics[name]["dev"].append(
                mean_poisson_deviance(y_te, pred, sample_weight=w_te))
            metrics[name]["gini"].append(
                normalised_gini(pred, y_te, w_te, c_te))
            ratios = decile_ratios(pred, w_te, c_te)
            empty_deciles += int(np.isnan(ratios).sum())
            metrics[name]["worstcal"].append(np.nanmax(np.abs(ratios - 1.0)))
            if name == "GBM":
                gbm_bottom.append(ratios[0])

    return metrics, gbm_bottom, dropped, empty_deciles


def ms(vals):
    a = np.array(vals, dtype=float)
    return np.nanmean(a), np.nanstd(a, ddof=1)


def report(cfg, metrics, gbm_bottom, dropped, empty_deciles):
    print()
    print("#" * 80)
    print(f"#  PORTFOLIO: {cfg['name']}")
    print(f"#  file={cfg['file']}   rows dropped (Exposure<=0)={dropped}")
    print(f"#  numeric={cfg['numeric']}   categorical={cfg['categorical']}")
    print("#" * 80)

    print()
    print("  1) 5-FOLD SUMMARY  (mean +/- std, ddof=1)")
    print("  " + "-" * 74)
    print("  {:<6}{:>22}{:>22}{:>22}".format(
        "Model", "Poisson deviance", "Normalised Gini", "Worst-decile cal.err"))
    print("  " + "-" * 74)
    for name in ("GLM", "GBM"):
        dm, ds = ms(metrics[name]["dev"])
        gm, gs = ms(metrics[name]["gini"])
        cm, cs = ms(metrics[name]["worstcal"])
        print("  {:<6}{:>13.6f} +/-{:>6.6f}{:>13.4f} +/-{:>6.4f}{:>13.4f} +/-{:>6.4f}".format(
            name, dm, ds, gm, gs, cm, cs))
    print("  " + "-" * 74)
    if empty_deciles:
        print(f"  (note: {empty_deciles} decile(s) across all folds/models had zero observed")
        print("   claims and were skipped in the worst-decile calc - rare-claim book)")

    print()
    print("  2) GBM bottom (lowest-risk) decile predicted/observed ratio, by fold:")
    for i, r in enumerate(gbm_bottom, start=1):
        txt = "n/a (no observed claims in decile)" if np.isnan(r) else "{:.3f}".format(r)
        print(f"       fold {i}:  {txt}")

    gap = np.array(metrics["GBM"]["gini"]) - np.array(metrics["GLM"]["gini"])
    cal_gbm = np.array(metrics["GBM"]["worstcal"])
    cal_glm = np.array(metrics["GLM"]["worstcal"])
    print()
    print("  3) Per-fold GBM-minus-GLM normalised-Gini gap:")
    print("       " + ", ".join("{:+.4f}".format(g) for g in gap))
    print("       folds where GBM worst-decile cal.err > GLM's: {}/{}".format(
        int(np.sum(cal_gbm > cal_glm)), N_SPLITS))


for cfg in PORTFOLIOS:
    out = run_portfolio(cfg)
    report(cfg, *out)
print()
