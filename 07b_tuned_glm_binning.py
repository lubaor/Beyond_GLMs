"""
Tuned GLM, phase 1: fine binning + regularisation.

Numerics are cut into many quantile bins and STEP encoded: feature k is
1(bin > k), so a coefficient is the jump between adjacent bins. An L2 penalty
on those jumps is a first-difference (roughness) penalty, which lets the data
choose how much structure each variable gets; an L1 penalty sets jumps to
exactly zero, merging neighbouring bins. That is what makes "regularisation
identifies the binning" literal rather than a figure of speech.

alpha is chosen by an inner 3-fold on each outer training fold only, so the
outer 5-fold estimate stays honest. Outer folds, caps and scoring are copied
from 05_crossfold_stability.py so the numbers are directly comparable.
"""

import os, time, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

warnings.filterwarnings("ignore")

CSV = "01_freMTPL2_freq.csv"
OUT = "results"

SEED, N_OUTER, N_INNER = 0, 5, 3
N_BINS = 32
NUM = ["VehAge", "DrivAge", "BonusMalus", "Density"]
CAT = ["VehBrand", "VehPower", "VehGas", "Region", "Area"]
ALPHAS = np.logspace(-7, 0, 8)


def ordered_gini(pred, w, cnt):
    o = np.argsort(pred, kind="mergesort")
    wo, co = w[o], cnt[o]
    x = np.concatenate([[0.0], np.cumsum(wo) / wo.sum()])
    y = np.concatenate([[0.0], np.cumsum(co) / co.sum()])
    return 1.0 - 2.0 * np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0)


def norm_gini(pred, obs, w, cnt):
    return ordered_gini(pred, w, cnt) / ordered_gini(obs, w, cnt)


def decile_ratios(pred, w, cnt, n_bins=10):
    o = np.argsort(pred, kind="mergesort")
    p, ww, cc = pred[o], w[o], cnt[o]
    frac = (np.cumsum(ww) - 0.5 * ww) / ww.sum()
    dec = np.clip((frac * n_bins).astype(int), 0, n_bins - 1)
    out = np.full(n_bins, np.nan)
    for d in range(n_bins):
        m = dec == d
        sw = ww[m].sum()
        oc = cc[m].sum()
        if oc > 0:
            out[d] = (np.sum(ww[m] * p[m]) / sw) / (oc / sw)
    return out


def step_encode(b, n_bins):
    """bin index -> 1(bin > k) for k = 0 .. n_bins-2, as a sparse matrix."""
    n, p = b.shape
    blocks = []
    for j in range(p):
        col = b[:, j].astype(int)
        M = (col[:, None] > np.arange(n_bins - 1)[None, :]).astype(np.float64)
        blocks.append(sp.csr_matrix(M))
    return sp.hstack(blocks).tocsr()


os.makedirs("results", exist_ok=True)
print("loading", flush=True)
df = pd.read_csv(CSV).reset_index(drop=True)
df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
df["Exposure"] = df["Exposure"].clip(upper=1)
df["Frequency"] = df["ClaimNb"] / df["Exposure"]
y = df["Frequency"].values
w = df["Exposure"].values
c = df["ClaimNb"].values
print(f"  {len(df):,} rows", flush=True)

kf = KFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
res = {"dev": [], "gini": [], "worstcal": [], "alpha": [], "ncol": []}
t0 = time.time()

for fold, (tr, te) in enumerate(kf.split(df), start=1):
    # fit the encoders on the training fold only
    kb = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="quantile", subsample=None)
    b_tr = kb.fit_transform(df.iloc[tr][NUM])
    b_te = kb.transform(df.iloc[te][NUM])
    nb = int(max(len(e) - 1 for e in kb.bin_edges_))

    oh = OneHotEncoder(handle_unknown="ignore")
    c_tr = oh.fit_transform(df.iloc[tr][CAT])
    c_te = oh.transform(df.iloc[te][CAT])

    X_tr = sp.hstack([step_encode(b_tr, nb), c_tr]).tocsr()
    X_te = sp.hstack([step_encode(b_te, nb), c_te]).tocsr()

    # inner 3-fold to pick alpha, on the training fold only
    inner = KFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
    scores = np.zeros(len(ALPHAS))
    for itr, iva in inner.split(np.arange(X_tr.shape[0])):
        for ai, a in enumerate(ALPHAS):
            m = PoissonRegressor(alpha=a, solver="newton-cholesky", max_iter=300)
            m.fit(X_tr[itr], y[tr][itr], sample_weight=w[tr][itr])
            pv = m.predict(X_tr[iva])
            scores[ai] += mean_poisson_deviance(y[tr][iva], pv, sample_weight=w[tr][iva])
    best = ALPHAS[int(np.argmin(scores))]

    m = PoissonRegressor(alpha=best, solver="newton-cholesky", max_iter=300)
    m.fit(X_tr, y[tr], sample_weight=w[tr])
    pred = m.predict(X_te)

    res["dev"].append(mean_poisson_deviance(y[te], pred, sample_weight=w[te]))
    res["gini"].append(norm_gini(pred, y[te], w[te], c[te]))
    res["worstcal"].append(np.nanmax(np.abs(decile_ratios(pred, w[te], c[te]) - 1.0)))
    res["alpha"].append(best)
    res["ncol"].append(X_tr.shape[1])
    print(f"  fold {fold}: alpha={best:.2e}  cols={X_tr.shape[1]}  "
          f"dev={res['dev'][-1]:.6f}  gini={res['gini'][-1]:.4f}  "
          f"worstcal={res['worstcal'][-1]:.4f}   [{time.time()-t0:.0f}s]", flush=True)

def ms(v):
    v = np.asarray(v)
    return v.mean(), v.std(ddof=1)

print("\n" + "=" * 78, flush=True)
print("  TUNED GLM phase 1: fine bins (step encoded) + L2 on the jumps", flush=True)
print("=" * 78, flush=True)
for k, lab in (("dev", "Poisson deviance"), ("gini", "Normalised Gini"), ("worstcal", "Worst-decile cal.err")):
    mu, sd = ms(res[k])
    print(f"  {lab:22} {mu:.6f} +/-{sd:.6f}", flush=True)
print(f"  alpha chosen per fold: {[f'{a:.1e}' for a in res['alpha']]}", flush=True)
print(f"  design columns:        {res['ncol'][0]}", flush=True)
print("\n  benchmark, same folds:", flush=True)
print("    GLM  (baseline, untuned)  dev 0.593092   gini 0.2943   worstcal 0.1162", flush=True)
print("    GBM  (untuned)            dev 0.572703   gini 0.3505   worstcal 0.0825", flush=True)
print("    GBM  (tuned)              dev 0.571598   gini 0.3525   worstcal 0.0977", flush=True)

pd.DataFrame(res).to_csv(f"{OUT}/\tuned_glm_p1.csv", index=False)
print(f"\nwrote {OUT}\\tuned_glm_p1.csv", flush=True)
