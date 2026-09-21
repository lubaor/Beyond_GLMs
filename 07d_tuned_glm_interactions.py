"""
Tuned GLM, phase 2: forward interaction search on top of phase 1.

Main effects are the phase-1 design (fine quantile bins, step encoded, plus
one-hot categoricals). Candidate interactions are pairwise crosses built on a
COARSE recode of each variable, so a cross costs tens of columns rather than
hundreds.

Selection is nested: inside each outer training fold the data is split 75/25,
interactions are added greedily by validation Poisson deviance, and only then
is the chosen set refitted on the whole outer training fold. Nothing from the
outer test fold touches the search.

Round 1 scores all 36 candidate pairs; later rounds only reconsider the 12 that
looked most promising, which keeps the cost near 6 minutes per outer fold.
"""

import os, time, warnings, itertools, json
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

SEED, N_OUTER = 0, 5
N_BINS, N_COARSE = 32, 5
MAX_INT, KEEP, TOL = 8, 12, 1e-5
NUM = ["VehAge", "DrivAge", "BonusMalus", "Density"]
CAT = ["VehBrand", "VehPower", "VehGas", "Region", "Area"]
ALL_VARS = NUM + CAT
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
        sw, oc = ww[m].sum(), cc[m].sum()
        if oc > 0:
            out[d] = (np.sum(ww[m] * p[m]) / sw) / (oc / sw)
    return out


def step_encode(b, n_bins):
    blocks = []
    for j in range(b.shape[1]):
        col = b[:, j].astype(int)
        blocks.append(sp.csr_matrix((col[:, None] > np.arange(n_bins - 1)[None, :]).astype(np.float64)))
    return sp.hstack(blocks).tocsr()


def fit_dev(Xf, yf, wf, Xv, yv, wv, alpha):
    m = PoissonRegressor(alpha=alpha, solver="newton-cholesky", max_iter=300)
    m.fit(Xf, yf, sample_weight=wf)
    return mean_poisson_deviance(yv, m.predict(Xv), sample_weight=wv)


os.makedirs("results", exist_ok=True)
print("loading", flush=True)
df = pd.read_csv(CSV).reset_index(drop=True)
df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
df["Exposure"] = df["Exposure"].clip(upper=1)
df["Frequency"] = df["ClaimNb"] / df["Exposure"]
y, w, c = df["Frequency"].values, df["Exposure"].values, df["ClaimNb"].values

PAIRS = list(itertools.combinations(ALL_VARS, 2))
print(f"  {len(df):,} rows, {len(PAIRS)} candidate pairs", flush=True)

kf = KFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
res = {"dev": [], "gini": [], "worstcal": [], "alpha": [], "ncol": [], "chosen": []}
t0 = time.time()

for fold, (tr, te) in enumerate(kf.split(df), start=1):
    d_tr, d_te = df.iloc[tr], df.iloc[te]

    # ---- main effects, encoders fit on the training fold only -------------
    kb = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="quantile", subsample=None)
    b_tr, b_te = kb.fit_transform(d_tr[NUM]), kb.transform(d_te[NUM])
    nb = int(max(len(e) - 1 for e in kb.bin_edges_))
    oh = OneHotEncoder(handle_unknown="ignore")
    M_tr = sp.hstack([step_encode(b_tr, nb), oh.fit_transform(d_tr[CAT])]).tocsr()
    M_te = sp.hstack([step_encode(b_te, nb), oh.transform(d_te[CAT])]).tocsr()

    # ---- coarse integer codes used to build the crosses -------------------
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

    def cross(a, b_, which):
        src = code_tr if which == "tr" else code_te
        comb = src[a] * nlev[b_] + src[b_]
        n = nlev[a] * nlev[b_]
        rows = np.arange(len(comb))
        return sp.csr_matrix((np.ones(len(comb)), (rows, comb)), shape=(len(comb), n))

    # ---- inner 75/25 split for the search ---------------------------------
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(M_tr.shape[0])
    cut = int(0.75 * len(perm))
    ifit, ival = perm[:cut], perm[cut:]
    yf, wf, yv, wv = y[tr][ifit], w[tr][ifit], y[tr][ival], w[tr][ival]

    alpha_s = 1e-4
    base_fit, base_val = M_tr[ifit], M_tr[ival]
    best_dev = fit_dev(base_fit, yf, wf, base_val, yv, wv, alpha_s)
    chosen, pool = [], list(PAIRS)
    cur_fit, cur_val = base_fit, base_val

    for rnd in range(MAX_INT):
        scored = []
        for (a, b_) in pool:
            if (a, b_) in chosen:
                continue
            Xf = sp.hstack([cur_fit, cross(a, b_, "tr")[ifit]]).tocsr()
            Xv = sp.hstack([cur_val, cross(a, b_, "tr")[ival]]).tocsr()
            scored.append((fit_dev(Xf, yf, wf, Xv, yv, wv, alpha_s), (a, b_)))
        scored.sort()
        gain = best_dev - scored[0][0]
        if gain < TOL:
            print(f"    fold {fold} round {rnd+1}: best gain {gain:.2e} < tol, stop", flush=True)
            break
        best_dev, pick = scored[0]
        chosen.append(pick)
        cur_fit = sp.hstack([cur_fit, cross(*pick, "tr")[ifit]]).tocsr()
        cur_val = sp.hstack([cur_val, cross(*pick, "tr")[ival]]).tocsr()
        if rnd == 0:
            pool = [p for _, p in scored[:KEEP]]
        print(f"    fold {fold} round {rnd+1}: + {pick[0]} x {pick[1]}  "
              f"dev={best_dev:.6f}  gain={gain:.2e}  [{time.time()-t0:.0f}s]", flush=True)

    # ---- refit chosen structure on the full training fold, retune alpha ---
    X_tr, X_te = M_tr, M_te
    for (a, b_) in chosen:
        X_tr = sp.hstack([X_tr, cross(a, b_, "tr")]).tocsr()
        X_te = sp.hstack([X_te, cross(a, b_, "te")]).tocsr()

    sub = np.random.RandomState(SEED).permutation(X_tr.shape[0])
    s_fit, s_val = sub[:int(0.75 * len(sub))], sub[int(0.75 * len(sub)):]
    devs = [fit_dev(X_tr[s_fit], y[tr][s_fit], w[tr][s_fit],
                    X_tr[s_val], y[tr][s_val], w[tr][s_val], a) for a in ALPHAS]
    best_alpha = ALPHAS[int(np.argmin(devs))]

    m = PoissonRegressor(alpha=best_alpha, solver="newton-cholesky", max_iter=300)
    m.fit(X_tr, y[tr], sample_weight=w[tr])
    pred = m.predict(X_te)

    res["dev"].append(mean_poisson_deviance(y[te], pred, sample_weight=w[te]))
    res["gini"].append(norm_gini(pred, y[te], w[te], c[te]))
    res["worstcal"].append(np.nanmax(np.abs(decile_ratios(pred, w[te], c[te]) - 1.0)))
    res["alpha"].append(best_alpha)
    res["ncol"].append(X_tr.shape[1])
    res["chosen"].append([f"{a}*{b_}" for a, b_ in chosen])
    print(f"  fold {fold}: alpha={best_alpha:.1e} cols={X_tr.shape[1]} "
          f"dev={res['dev'][-1]:.6f} gini={res['gini'][-1]:.4f} "
          f"worstcal={res['worstcal'][-1]:.4f}  [{time.time()-t0:.0f}s]", flush=True)


def ms(v):
    v = np.asarray(v, dtype=float)
    return v.mean(), v.std(ddof=1)


print("\n" + "=" * 78, flush=True)
print("  TUNED GLM phase 2: fine bins + L2 + forward-selected interactions", flush=True)
print("=" * 78, flush=True)
for k, lab in (("dev", "Poisson deviance"), ("gini", "Normalised Gini"), ("worstcal", "Worst-decile cal.err")):
    mu, sd = ms(res[k])
    print(f"  {lab:22} {mu:.6f} +/-{sd:.6f}", flush=True)
print(f"  design columns per fold: {res['ncol']}", flush=True)
print("\n  interactions chosen, by fold:", flush=True)
for i, ch in enumerate(res["chosen"], start=1):
    print(f"    fold {i}: {', '.join(ch) if ch else '(none)'}", flush=True)
from collections import Counter
cnt = Counter(x for ch in res["chosen"] for x in ch)
print("\n  selected in how many of the 5 folds:", flush=True)
for k, v in cnt.most_common():
    print(f"    {v}/5  {k}", flush=True)

with open(f"{OUT}/\tuned_glm_p2.json", "w") as fh:
    json.dump({k: (v if k != "chosen" else v) for k, v in res.items()}, fh, indent=2, default=float)
print(f"\nwrote {OUT}\\tuned_glm_p2.json", flush=True)
