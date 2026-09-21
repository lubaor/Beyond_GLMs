"""
Tuned GLM: fine bins + ELASTIC NET + monotone BonusMalus.

Encoding. Each numeric is cut into 32 quantile bins and step encoded as
1(bin > k), so coefficient k is the JUMP from bin k to bin k+1.
  - L1 on a jump drives it to exactly zero, merging those two bins. The
    surviving jumps ARE the binning the data chose.
  - L2 on the jumps is a first-difference roughness penalty.
  - Monotonicity in BonusMalus holds iff every one of its jumps is >= 0,
    which is a box constraint, not an approximation.

sklearn's PoissonRegressor is L2-only and has no coefficient bounds, so the
fit is done directly: weighted Poisson negative log-likelihood plus an elastic
net penalty, minimised with L-BFGS-B under the split beta = u - v, u,v >= 0.
On the non-negative orthant the L1 term is linear, so the whole objective is
smooth and L-BFGS-B is well behaved. Forcing v = 0 on a block pins that block
to beta >= 0, which is the monotonicity constraint.

(alpha, l1_ratio) are chosen on a 75/25 split inside each outer training fold.
Outer folds, caps and scoring match 05_crossfold_stability.py.
"""

import os, time, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

warnings.filterwarnings("ignore")

CSV = "01_freMTPL2_freq.csv"
OUT = "results"

SEED, N_OUTER, N_BINS = 0, 5, 32
NUM = ["VehAge", "DrivAge", "BonusMalus", "Density"]
CAT = ["VehBrand", "VehPower", "VehGas", "Region", "Area"]
MONO = "BonusMalus"                    # constrained non-decreasing
ALPHAS = np.logspace(-6, -2, 5)
L1_RATIOS = [0.0, 0.5, 0.9, 1.0]


# ------------------------------------------------------------------ scoring
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


def step_encode(b, n_bins):
    blocks = []
    for j in range(b.shape[1]):
        col = b[:, j].astype(int)
        blocks.append(sp.csr_matrix((col[:, None] > np.arange(n_bins - 1)[None, :]).astype(np.float64)))
    return sp.hstack(blocks).tocsr()


# ------------------------------------------------- elastic net Poisson fit
def fit_enet(X, y, w, alpha, l1_ratio, mono_idx=None, maxiter=500):
    """beta = u - v with u,v >= 0; v pinned to 0 on mono_idx => beta >= 0 there."""
    n, p = X.shape
    W = w.sum()
    wy = w * y

    def obj(theta):
        c = theta[0]
        u, v = theta[1:1 + p], theta[1 + p:]
        b = u - v
        eta = np.clip(X @ b + c, -30.0, 30.0)
        mu = np.exp(eta)
        nll = (w @ mu - wy @ eta) / W
        pen = alpha * (l1_ratio * (u.sum() + v.sum()) + 0.5 * (1 - l1_ratio) * (b @ b))
        r = (w * mu - wy) / W
        gb = X.T @ r
        gc = r.sum()
        l2g = alpha * (1 - l1_ratio) * b
        gu = gb + alpha * l1_ratio + l2g
        gv = -gb + alpha * l1_ratio - l2g
        return nll + pen, np.concatenate([[gc], gu, gv])

    bounds = [(None, None)] + [(0.0, None)] * (2 * p)
    if mono_idx is not None:
        for j in mono_idx:
            bounds[1 + p + j] = (0.0, 0.0)          # v_j = 0  ->  beta_j >= 0
    th0 = np.zeros(1 + 2 * p)
    th0[0] = np.log(max((wy.sum() / W), 1e-6))
    r = minimize(obj, th0, jac=True, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": maxiter, "maxfun": maxiter * 2})
    c = r.x[0]; b = r.x[1:1 + p] - r.x[1 + p:]
    return c, b

def predict(X, c, b):
    return np.exp(np.clip(X @ b + c, -30.0, 30.0))


os.makedirs("results", exist_ok=True)
print("loading", flush=True)
df = pd.read_csv(CSV).reset_index(drop=True)
df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
df["Exposure"] = df["Exposure"].clip(upper=1)
df["Frequency"] = df["ClaimNb"] / df["Exposure"]
y, w, c_cnt = df["Frequency"].values, df["Exposure"].values, df["ClaimNb"].values
print(f"  {len(df):,} rows", flush=True)

kf = KFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
res = {"dev": [], "gini": [], "worstcal": [], "alpha": [], "l1": [], "nnz": []}
sparsity_report = None
t0 = time.time()

for fold, (tr, te) in enumerate(kf.split(df), start=1):
    d_tr, d_te = df.iloc[tr], df.iloc[te]
    kb = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="quantile", subsample=None)
    b_tr, b_te = kb.fit_transform(d_tr[NUM]), kb.transform(d_te[NUM])
    nb = int(max(len(e) - 1 for e in kb.bin_edges_))
    S = nb - 1                                        # jumps per numeric
    oh = OneHotEncoder(handle_unknown="ignore")
    X_tr = sp.hstack([step_encode(b_tr, nb), oh.fit_transform(d_tr[CAT])]).tocsr()
    X_te = sp.hstack([step_encode(b_te, nb), oh.transform(d_te[CAT])]).tocsr()
    mono_idx = np.arange(NUM.index(MONO) * S, (NUM.index(MONO) + 1) * S)

    if fold == 1:   # one-off check that the custom optimiser matches sklearn
        sk = PoissonRegressor(alpha=1e-4, solver="newton-cholesky", max_iter=300)
        sk.fit(X_tr, y[tr], sample_weight=w[tr])
        d_sk = mean_poisson_deviance(y[te], sk.predict(X_te), sample_weight=w[te])
        cc, bb = fit_enet(X_tr, y[tr], w[tr], 1e-4, 0.0)      # pure L2, unconstrained
        d_my = mean_poisson_deviance(y[te], predict(X_te, cc, bb), sample_weight=w[te])
        print(f"  [check] sklearn L2 dev={d_sk:.6f}   custom L2 dev={d_my:.6f}   "
              f"diff={abs(d_sk-d_my):.2e}", flush=True)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(X_tr.shape[0]); cut = int(0.75 * len(perm))
    ifit, ival = perm[:cut], perm[cut:]

    best = (np.inf, None, None)
    for a in ALPHAS:
        for l1 in L1_RATIOS:
            cc, bb = fit_enet(X_tr[ifit], y[tr][ifit], w[tr][ifit], a, l1, mono_idx)
            d = mean_poisson_deviance(y[tr][ival], predict(X_tr[ival], cc, bb),
                                      sample_weight=w[tr][ival])
            if d < best[0]:
                best = (d, a, l1)
    _, a_best, l1_best = best

    cc, bb = fit_enet(X_tr, y[tr], w[tr], a_best, l1_best, mono_idx)
    pred = predict(X_te, cc, bb)
    nnz = int(np.sum(np.abs(bb) > 1e-8))

    res["dev"].append(mean_poisson_deviance(y[te], pred, sample_weight=w[te]))
    res["gini"].append(norm_gini(pred, y[te], w[te], c_cnt[te]))
    res["worstcal"].append(np.nanmax(np.abs(decile_ratios(pred, w[te], c_cnt[te]) - 1.0)))
    res["alpha"].append(a_best); res["l1"].append(l1_best); res["nnz"].append(nnz)

    if fold == 1:
        sparsity_report = []
        for j, v in enumerate(NUM):
            blk = bb[j * S:(j + 1) * S]
            sparsity_report.append((v, int(np.sum(np.abs(blk) > 1e-8)), S,
                                    bool(np.all(blk >= -1e-9))))
    print(f"  fold {fold}: alpha={a_best:.1e} l1={l1_best} nnz={nnz}/{X_tr.shape[1]} "
          f"dev={res['dev'][-1]:.6f} gini={res['gini'][-1]:.4f} "
          f"worstcal={res['worstcal'][-1]:.4f}  [{time.time()-t0:.0f}s]", flush=True)


def ms(v):
    v = np.asarray(v, float); return v.mean(), v.std(ddof=1)

print("\n" + "=" * 78, flush=True)
print("  TUNED GLM: fine bins + elastic net + monotone BonusMalus", flush=True)
print("=" * 78, flush=True)
for k, lab in (("dev","Poisson deviance"),("gini","Normalised Gini"),("worstcal","Worst-decile cal.err")):
    mu, sd = ms(res[k]); print(f"  {lab:22} {mu:.6f} +/-{sd:.6f}", flush=True)
print(f"  alpha per fold:    {[f'{a:.0e}' for a in res['alpha']]}", flush=True)
print(f"  l1_ratio per fold: {res['l1']}", flush=True)
print(f"  non-zero coefs:    {res['nnz']}", flush=True)
print("\n  binning chosen by the penalty (fold 1): surviving jumps of 31", flush=True)
for v, k, tot, mono_ok in sparsity_report:
    tag = "  monotone OK" if v == MONO and mono_ok else ("  MONOTONE VIOLATED" if v == MONO else "")
    print(f"    {v:12} {k:3}/{tot} jumps{tag}", flush=True)
print("\n  benchmark, same folds:", flush=True)
print("    GLM baseline (untuned)   dev 0.593092  gini 0.2943  worstcal 0.1162", flush=True)
print("    GBM untuned              dev 0.572703  gini 0.3505  worstcal 0.0825", flush=True)
print("    GBM tuned                dev 0.571598  gini 0.3525  worstcal 0.0977", flush=True)
pd.DataFrame(res).to_csv(f"{OUT}/\tuned_glm_enet.csv", index=False)
print(f"\nwrote {OUT}\\tuned_glm_enet.csv", flush=True)
