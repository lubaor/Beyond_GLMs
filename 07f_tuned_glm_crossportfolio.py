"""
Step 7f - the tuned GLM applied to the Australian and Swedish books.

Steps 07b to 07e tuned the GLM on freMTPL2 only, which left the cross-portfolio
conclusions on slide 6 resting on a fixed-specification incumbent. This closes
that gap with the same machinery, adapted to what each book actually contains.

Encoding, in three kinds:

  numeric            -> quantile bins, then STEP encoded as 1(bin > k)
  ordered category   -> STEP encoded directly on the known level order
  nominal category   -> one-hot

Step encoding is what makes the penalty do the grouping: each coefficient is
the jump between adjacent levels, so L1 sets a jump to exactly zero and merges
those levels. For BonusClass that is the machine doing what an actuary does by
hand when grouping bonus classes; for the EV-ratio risk classes and the age
bands it is the same idea.

No monotonicity is imposed here. On freMTPL2, BonusMalus had an unambiguous
direction. Nothing in these two books does: driver age is U-shaped, vehicle
value is not obviously ordered in risk, and the sign convention on BonusClass
is not certain from the data alone. The fitted jumps are reported so the
direction can be read off rather than assumed.

Interactions are forward selected from every pair, scored on a 75/25 split
inside each outer training fold. Outer folds, exposure handling and scoring
match 06_crossportfolio.py and 07_tuned_gbm_frequency.py.
"""

import itertools, os, time, warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

warnings.filterwarnings("ignore")

SEED, N_OUTER, N_BINS, N_COARSE = 0, 5, 20, 5
MAX_INT, TOL = 6, 1e-5
ALPHAS = np.logspace(-6, -2, 5)
L1_RATIOS = [0.0, 0.5, 0.9, 1.0]
OUT = "results"

AUS_VEHAGE = ["youngest cars", "young cars", "old cars", "oldest cars"]
AUS_DRIVAGE = ["youngest people", "young people", "working people",
               "older work. people", "old people", "oldest people"]
SWE_BONUS = [f"BM{i}" for i in range(1, 8)]
SWE_RISK = ["EV ratio <5", "EV ratio 6-8", "EV ratio 9-12", "EV ratio 13-15",
            "EV ratio 16-19", "EV ratio 20-24", "EV ratio >25"]

PORTFOLIOS = [
    {"name": "Australian (ausprivauto)", "file": "04_ausprivauto_dejong.csv",
     "num": ["VehValue"],
     "ord": {"VehAge": AUS_VEHAGE, "DrivAge": AUS_DRIVAGE},
     "nom": ["VehBody", "Gender"],
     "bench": {"glm_dev": 0.798375, "glm_gini": 0.0995, "glm_cal": 0.1874,
               "gbm_dev": 0.799749, "gbm_gini": 0.0858, "gbm_cal": 0.2416}},
    {"name": "Swedish Wasa motorcycle", "file": "03_wasa_motorcycle.csv",
     "num": ["OwnerAge", "VehAge"],
     "ord": {"BonusClass": SWE_BONUS, "RiskClass": SWE_RISK},
     "nom": ["Area", "Gender"],
     "bench": {"glm_dev": 0.131172, "glm_gini": 0.5528, "glm_cal": 1.4049,
               "gbm_dev": 0.131341, "gbm_gini": 0.5564, "gbm_cal": 1.0554}},
]


# ----------------------------------------------------------------- scoring
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


def steps_from_codes(codes, n_levels):
    """integer level -> 1(level > k) for k = 0 .. n_levels-2"""
    c = np.asarray(codes).astype(int)
    return sp.csr_matrix((c[:, None] > np.arange(n_levels - 1)[None, :]).astype(np.float64))


# --------------------------------------------------- elastic net Poisson
def fit_enet(X, y, w, alpha, l1_ratio, maxiter=400):
    n, p = X.shape
    W = w.sum(); wy = w * y

    def obj(theta):
        c = theta[0]; u, v = theta[1:1 + p], theta[1 + p:]
        b = u - v
        eta = np.clip(X @ b + c, -30.0, 30.0)
        mu = np.exp(eta)
        nll = (w @ mu - wy @ eta) / W
        pen = alpha * (l1_ratio * (u.sum() + v.sum()) + 0.5 * (1 - l1_ratio) * (b @ b))
        r = (w * mu - wy) / W
        gb = X.T @ r; l2g = alpha * (1 - l1_ratio) * b
        return nll + pen, np.concatenate([[r.sum()],
                                          gb + alpha * l1_ratio + l2g,
                                          -gb + alpha * l1_ratio - l2g])

    th0 = np.zeros(1 + 2 * p); th0[0] = np.log(max(wy.sum() / W, 1e-6))
    r = minimize(obj, th0, jac=True, method="L-BFGS-B",
                 bounds=[(None, None)] + [(0.0, None)] * (2 * p),
                 options={"maxiter": maxiter, "maxfun": maxiter * 2})
    return r.x[0], r.x[1:1 + p] - r.x[1 + p:]


def predict(X, c, b):
    return np.exp(np.clip(X @ b + c, -30.0, 30.0))


def dev_of(Xf, yf, wf, Xv, yv, wv, a, l1):
    c, b = fit_enet(Xf, yf, wf, a, l1)
    return mean_poisson_deviance(yv, predict(Xv, c, b), sample_weight=wv)


os.makedirs(OUT, exist_ok=True)
summary = []

for P in PORTFOLIOS:
    print("=" * 78, flush=True)
    print(f"  {P['name']}", flush=True)
    print("=" * 78, flush=True)
    df = pd.read_csv(P["file"])
    n_raw = len(df)
    df = df[df["Exposure"] > 0].copy()
    df["Exposure"] = df["Exposure"].clip(upper=1)
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    df = df.reset_index(drop=True)
    print(f"  rows {len(df):,} (dropped {n_raw - len(df)} with Exposure <= 0)", flush=True)

    y, w, cnt = df["Frequency"].values, df["Exposure"].values, df["ClaimNb"].values
    NUM, ORD, NOM = P["num"], P["ord"], P["nom"]
    ALLV = NUM + list(ORD) + NOM
    PAIRS = list(itertools.combinations(ALLV, 2))
    print(f"  {len(NUM)} numeric, {len(ORD)} ordered, {len(NOM)} nominal"
          f"  ->  {len(PAIRS)} candidate interactions", flush=True)

    kf = KFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    res = {"dev": [], "gini": [], "cal": [], "alpha": [], "l1": [], "nnz": [], "ncol": []}
    chosen_all, spars = [], None
    t0 = time.time()

    for fold, (tr, te) in enumerate(kf.split(df), start=1):
        d_tr, d_te = df.iloc[tr], df.iloc[te]
        blocks_tr, blocks_te, block_span = [], [], {}
        pos = 0

        kb = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="quantile", subsample=None)
        b_tr = kb.fit_transform(d_tr[NUM]); b_te = kb.transform(d_te[NUM])
        nb = int(max(len(e) - 1 for e in kb.bin_edges_))
        for j, v in enumerate(NUM):
            blocks_tr.append(steps_from_codes(b_tr[:, j], nb))
            blocks_te.append(steps_from_codes(b_te[:, j], nb))
            block_span[v] = (pos, pos + nb - 1); pos += nb - 1

        for v, order in ORD.items():
            idx = pd.Index(order)
            blocks_tr.append(steps_from_codes(idx.get_indexer(d_tr[v].astype(str)).clip(min=0), len(order)))
            blocks_te.append(steps_from_codes(idx.get_indexer(d_te[v].astype(str)).clip(min=0), len(order)))
            block_span[v] = (pos, pos + len(order) - 1); pos += len(order) - 1

        oh = OneHotEncoder(handle_unknown="ignore")
        blocks_tr.append(oh.fit_transform(d_tr[NOM])); blocks_te.append(oh.transform(d_te[NOM]))
        X_tr = sp.hstack(blocks_tr).tocsr(); X_te = sp.hstack(blocks_te).tocsr()

        # coarse codes for the interaction crosses
        kc = KBinsDiscretizer(n_bins=N_COARSE, encode="ordinal", strategy="quantile", subsample=None)
        cn_tr = kc.fit_transform(d_tr[NUM]).astype(int); cn_te = kc.transform(d_te[NUM]).astype(int)
        code_tr, code_te, nlev = {}, {}, {}
        for j, v in enumerate(NUM):
            code_tr[v], code_te[v], nlev[v] = cn_tr[:, j], cn_te[:, j], N_COARSE
        for v in list(ORD) + NOM:
            cats = pd.Index(ORD[v]) if v in ORD else pd.Index(sorted(d_tr[v].astype(str).unique()))
            code_tr[v] = cats.get_indexer(d_tr[v].astype(str)).clip(min=0)
            code_te[v] = cats.get_indexer(d_te[v].astype(str)).clip(min=0)
            nlev[v] = len(cats)

        def cross(a, b_, codes):
            comb = codes[a] * nlev[b_] + codes[b_]
            return sp.csr_matrix((np.ones(len(comb)), (np.arange(len(comb)), comb)),
                                 shape=(len(comb), nlev[a] * nlev[b_]))

        rng = np.random.RandomState(SEED)
        perm = rng.permutation(X_tr.shape[0]); cut = int(0.75 * len(perm))
        ifit, ival = perm[:cut], perm[cut:]
        yf, wf, yv, wv = y[tr][ifit], w[tr][ifit], y[tr][ival], w[tr][ival]

        a_s, l1_s = 1e-4, 0.5
        cur_f, cur_v = X_tr[ifit], X_tr[ival]
        best_dev = dev_of(cur_f, yf, wf, cur_v, yv, wv, a_s, l1_s)
        chosen = []
        for _ in range(MAX_INT):
            scored = []
            for pr in PAIRS:
                if pr in chosen:
                    continue
                C = cross(pr[0], pr[1], code_tr)
                scored.append((dev_of(sp.hstack([cur_f, C[ifit]]).tocsr(), yf, wf,
                                      sp.hstack([cur_v, C[ival]]).tocsr(), yv, wv, a_s, l1_s), pr))
            if not scored:
                break
            scored.sort()
            if best_dev - scored[0][0] < TOL:
                break
            best_dev, pick = scored[0]
            chosen.append(pick)
            C = cross(pick[0], pick[1], code_tr)
            cur_f = sp.hstack([cur_f, C[ifit]]).tocsr(); cur_v = sp.hstack([cur_v, C[ival]]).tocsr()

        for pr in chosen:
            X_tr = sp.hstack([X_tr, cross(pr[0], pr[1], code_tr)]).tocsr()
            X_te = sp.hstack([X_te, cross(pr[0], pr[1], code_te)]).tocsr()

        best = (np.inf, None, None)
        for a in ALPHAS:
            for l1 in L1_RATIOS:
                d = dev_of(X_tr[ifit], yf, wf, X_tr[ival], yv, wv, a, l1)
                if d < best[0]:
                    best = (d, a, l1)
        _, a_b, l1_b = best

        c_, b_ = fit_enet(X_tr, y[tr], w[tr], a_b, l1_b)
        pred = predict(X_te, c_, b_)
        res["dev"].append(mean_poisson_deviance(y[te], pred, sample_weight=w[te]))
        res["gini"].append(norm_gini(pred, y[te], w[te], cnt[te]))
        res["cal"].append(np.nanmax(np.abs(decile_ratios(pred, w[te], cnt[te]) - 1.0)))
        res["alpha"].append(a_b); res["l1"].append(l1_b)
        res["nnz"].append(int((np.abs(b_) > 1e-8).sum())); res["ncol"].append(X_tr.shape[1])
        chosen_all.append([f"{a}*{b}" for a, b in chosen])
        if fold == 1:
            spars = [(v, int((np.abs(b_[s:e]) > 1e-8).sum()), e - s,
                      bool(np.all(b_[s:e] >= -1e-9)), bool(np.all(b_[s:e] <= 1e-9)))
                     for v, (s, e) in block_span.items()]
        print(f"  fold {fold}: a={a_b:.0e} l1={l1_b} cols={X_tr.shape[1]} nnz={res['nnz'][-1]} "
              f"dev={res['dev'][-1]:.6f} gini={res['gini'][-1]:.4f} cal={res['cal'][-1]:.4f} "
              f"int={len(chosen)} [{time.time()-t0:.0f}s]", flush=True)

    def ms(v):
        v = np.asarray(v, float); return v.mean(), v.std(ddof=1)

    g_m, g_s = ms(res["gini"]); d_m, d_s = ms(res["dev"]); c_m, c_s = ms(res["cal"])
    b = P["bench"]
    print(f"\n  {'model':26} {'Gini':>18} {'deviance':>12} {'worst-decile':>14}", flush=True)
    print(f"  {'GLM, fixed spec':26} {b['glm_gini']:>18.4f} {b['glm_dev']:>12.6f} {b['glm_cal']:>14.4f}", flush=True)
    print(f"  {'GLM, tuned (this step)':26} {g_m:>10.4f} +/-{g_s:.4f} {d_m:>12.6f} {c_m:>9.4f} +/-{c_s:.4f}", flush=True)
    print(f"  {'GBM, tuned':26} {b['gbm_gini']:>18.4f} {b['gbm_dev']:>12.6f} {b['gbm_cal']:>14.4f}", flush=True)
    print(f"\n  alpha {[f'{a:.0e}' for a in res['alpha']]}   l1 {res['l1']}", flush=True)
    print("  levels kept by the penalty (fold 1), of the jumps available:", flush=True)
    for v, k, tot, up, down in spars:
        tag = "  all jumps >= 0" if up and k else ("  all jumps <= 0" if down and k else "")
        print(f"    {v:12} {k:3}/{tot}{tag}", flush=True)
    print("  interactions chosen, by fold:", flush=True)
    for i, ch in enumerate(chosen_all, start=1):
        print(f"    fold {i}: {', '.join(ch) if ch else '(none)'}", flush=True)
    from collections import Counter
    cc = Counter(x for ch in chosen_all for x in ch)
    if cc:
        print("  selected in how many folds:", flush=True)
        for k, v in cc.most_common():
            print(f"    {v}/5  {k}", flush=True)
    print("", flush=True)

    summary.append({"portfolio": P["name"], "gini": g_m, "gini_sd": g_s,
                    "dev": d_m, "cal": c_m, "cal_sd": c_s,
                    "glm_fixed_gini": b["glm_gini"], "gbm_tuned_gini": b["gbm_gini"]})

pd.DataFrame(summary).to_csv(f"{OUT}/07f_tuned_glm_crossportfolio.csv", index=False)
print(f"wrote {OUT}/07f_tuned_glm_crossportfolio.csv", flush=True)
