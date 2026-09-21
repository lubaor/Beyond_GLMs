"""
Tuned GLM, final: fine bins + monotone BonusMalus + elastic net + the
consensus interactions from the forward search.

The forward search (tuned_glm_p2.py) was run independently inside each outer
fold. Six crosses were selected in at least four of the five folds, so that
consensus set is treated as the model structure here rather than re-searching.
alpha is retuned per fold; l1_ratio is fixed at 0.5, the median choice of the
main-effects run, to keep the grid affordable at ~600 columns.
"""

import os, time, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

warnings.filterwarnings("ignore")

SC = "results"
CSV = "01_freMTPL2_freq.csv"

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



SEED, N_OUTER, N_BINS, N_COARSE = 0, 5, 32, 5
NUM = ["VehAge", "DrivAge", "BonusMalus", "Density"]
CAT = ["VehBrand", "VehPower", "VehGas", "Region", "Area"]
MONO = "BonusMalus"
L1_FIXED = 0.5
ALPHAS = [3e-5, 1e-4, 3e-4]

# selected in >= 4 of 5 folds by the independent per-fold forward search
INTERACTIONS = [
    ("VehAge", "VehBrand"), ("VehAge", "VehPower"), ("VehAge", "BonusMalus"),
    ("DrivAge", "Region"), ("BonusMalus", "VehBrand"), ("VehAge", "VehGas"),
]

os.makedirs("results", exist_ok=True)
print("loading", flush=True)
df = pd.read_csv(CSV).reset_index(drop=True)
df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
df["Exposure"] = df["Exposure"].clip(upper=1)
df["Frequency"] = df["ClaimNb"] / df["Exposure"]
y, w, cnt = df["Frequency"].values, df["Exposure"].values, df["ClaimNb"].values
print(f"  {len(df):,} rows, {len(INTERACTIONS)} fixed interactions", flush=True)

kf = KFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
res = {"dev": [], "gini": [], "worstcal": [], "alpha": [], "nnz": [], "ncol": []}
t0 = time.time()

for fold, (tr, te) in enumerate(kf.split(df), start=1):
    d_tr, d_te = df.iloc[tr], df.iloc[te]
    kb = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="quantile", subsample=None)
    b_tr, b_te = kb.fit_transform(d_tr[NUM]), kb.transform(d_te[NUM])
    nb = int(max(len(e) - 1 for e in kb.bin_edges_)); S = nb - 1
    oh = OneHotEncoder(handle_unknown="ignore")
    X_tr = sp.hstack([step_encode(b_tr, nb), oh.fit_transform(d_tr[CAT])]).tocsr()
    X_te = sp.hstack([step_encode(b_te, nb), oh.transform(d_te[CAT])]).tocsr()
    mono_idx = np.arange(NUM.index(MONO) * S, (NUM.index(MONO) + 1) * S)

    kc = KBinsDiscretizer(n_bins=N_COARSE, encode="ordinal", strategy="quantile", subsample=None)
    cn_tr, cn_te = kc.fit_transform(d_tr[NUM]).astype(int), kc.transform(d_te[NUM]).astype(int)
    code_tr, code_te, nlev = {}, {}, {}
    for j, v in enumerate(NUM):
        code_tr[v], code_te[v], nlev[v] = cn_tr[:, j], cn_te[:, j], N_COARSE
    for v in CAT:
        cats = pd.Index(sorted(d_tr[v].astype(str).unique()))
        code_tr[v] = cats.get_indexer(d_tr[v].astype(str)).clip(min=0)
        code_te[v] = cats.get_indexer(d_te[v].astype(str)).clip(min=0)
        nlev[v] = len(cats)

    def cross(a, b_, src_codes):
        comb = src_codes[a] * nlev[b_] + src_codes[b_]
        n = nlev[a] * nlev[b_]
        return sp.csr_matrix((np.ones(len(comb)), (np.arange(len(comb)), comb)),
                             shape=(len(comb), n))

    for (a, b_) in INTERACTIONS:
        X_tr = sp.hstack([X_tr, cross(a, b_, code_tr)]).tocsr()
        X_te = sp.hstack([X_te, cross(a, b_, code_te)]).tocsr()

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(X_tr.shape[0]); cut = int(0.75 * len(perm))
    ifit, ival = perm[:cut], perm[cut:]

    best = (np.inf, None)
    for a in ALPHAS:
        cc, bb = fit_enet(X_tr[ifit], y[tr][ifit], w[tr][ifit], a, L1_FIXED, mono_idx)
        d = mean_poisson_deviance(y[tr][ival], predict(X_tr[ival], cc, bb),
                                  sample_weight=w[tr][ival])
        if d < best[0]:
            best = (d, a)
    a_best = best[1]

    cc, bb = fit_enet(X_tr, y[tr], w[tr], a_best, L1_FIXED, mono_idx)
    pred = predict(X_te, cc, bb)
    res["dev"].append(mean_poisson_deviance(y[te], pred, sample_weight=w[te]))
    res["gini"].append(norm_gini(pred, y[te], w[te], cnt[te]))
    res["worstcal"].append(np.nanmax(np.abs(decile_ratios(pred, w[te], cnt[te]) - 1.0)))
    res["alpha"].append(a_best)
    res["nnz"].append(int(np.sum(np.abs(bb) > 1e-8)))
    res["ncol"].append(X_tr.shape[1])
    blk = bb[mono_idx]
    print(f"  fold {fold}: alpha={a_best:.0e} nnz={res['nnz'][-1]}/{X_tr.shape[1]} "
          f"dev={res['dev'][-1]:.6f} gini={res['gini'][-1]:.4f} "
          f"worstcal={res['worstcal'][-1]:.4f} monoOK={bool(np.all(blk>=-1e-9))} "
          f"[{time.time()-t0:.0f}s]", flush=True)

def ms(v):
    v = np.asarray(v, float); return v.mean(), v.std(ddof=1)

print("\n" + "=" * 78, flush=True)
print("  FULLY TUNED GLM: bins + monotone BonusMalus + enet + interactions", flush=True)
print("=" * 78, flush=True)
for k, lab in (("dev","Poisson deviance"),("gini","Normalised Gini"),("worstcal","Worst-decile cal.err")):
    mu, sd = ms(res[k]); print(f"  {lab:22} {mu:.6f} +/-{sd:.6f}", flush=True)
print(f"  alpha per fold: {[f'{a:.0e}' for a in res['alpha']]}", flush=True)
print(f"  non-zero / total columns: {res['nnz']} / {res['ncol']}", flush=True)
pd.DataFrame(res).to_csv(f"{SC}/\tuned_glm_final.csv", index=False)
print(f"\nwrote {SC}\\tuned_glm_final.csv", flush=True)
