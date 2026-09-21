# Beyond GLMs: a one-page decision framework

Two decisions, in order. **Signal decides the risk model. Two gates decide the
causal estimate.** Everything below is measurable before you commit to building
anything.

Figures in brackets are the benchmark results that produced each rule. Five-fold
cross-validation, tuned GBM against an untuned GLM, one frozen harness.

---

## 1. Which risk model?

Measure the estimable signal first. The return on a boosted model scales with
it, not with the sophistication of the model.

| Your book | Diagnostic | Decision |
|---|---|---|
| **Rich signal**, normalised Gini around 0.25 or above | GBM beats GLM in every fold, gap well above fold-to-fold noise | **Tune a GBM and adopt it.** Expect a real lift, but size it against a *tuned* GLM [freMTPL2, tuned against tuned: 0.3127 to 0.3525, **+13%**; against a fixed-specification GLM the same result reads +20%] |
| **Low signal**, Gini around 0.15 or below | Tuned GBM converges towards the GLM and falls short | **Stay with the GLM.** You will not get a lift, and you pay in interpretability and governance for nothing [Australian: GLM 0.0995 vs GBM 0.0858, GLM ahead in 5/5 folds] |
| **Sparse claims** | Diagnostic SD is the same order as the estimate itself; deciles with zero observed claims | **Default to the GLM** and report a non-finding. You cannot demonstrate a difference either way, and you should not defend a model you cannot validate [Swedish: 0.5528 vs 0.5564, per-fold gaps change sign; worst-decile error 1.40 +/- 1.47] |

A lift is only real if it is **calibration-free**. Check that the challenger's
worst-decile error does not deteriorate [freMTPL2, tuned against tuned: GBM
0.098 vs GLM 0.115, so the challenger was *better* calibrated as well as
better at ranking].

### Tune both sides, and say what each lever bought

Comparing a searched challenger against a fixed incumbent is the most common
way these results get inflated. On freMTPL2, tuning the GLM properly closes a
third of the ranking gap:

| GLM variant | Gini | Worst-decile cal. err |
|---|---|---|
| Fixed specification | 0.2943 | 0.1162 +/-0.0588 |
| Fine bins, L2 on the jumps | 0.3097 | 0.1172 +/-0.0271 |
| Elastic net, monotone BonusMalus | 0.3082 | 0.1454 +/-0.0702 |
| Forward-selected interactions | 0.3157 | 0.1502 +/-0.0630 |
| **All combined** | **0.3127** | **0.1151 +/-0.0115** |
| *GBM, tuned* | *0.3525* | *0.0977* |

Two things worth carrying away. **Interactions did most of the work**, which is
the honest reason a boosted model wins here: it finds interaction structure for
free, and a GLM has to be told. Six crosses were picked in at least four of
five independent per-fold searches, led by `VehAge x VehBrand` and
`VehAge x VehPower` in all five. **The constraints paid for themselves in the
tails**: unconstrained interactions degraded worst-decile calibration, while
the monotone elastic net version is the best calibrated GLM variant and by far
the most stable across folds.

Technique worth stealing: step encode fine bins as `1(bin > k)`, so each
coefficient is the jump between adjacent bins. L1 then merges bins by zeroing
jumps, and monotonicity becomes the exact constraint "every jump >= 0". On
freMTPL2 the penalty reduced 31 candidate jumps to 6 for `BonusMalus`, 13 for
`VehAge`, 17 for `Density` and 18 for `DrivAge`.

---

## 2. Trust a causal estimate? Two gates, in this order

Applies to price elasticity, retention, conversion: anything where you want to
know what *would* happen if you changed price, not what correlates with it.

### Gate 1. Overlap

> After controlling for everything you have, what fraction of treatment
> variance is independent?

**If near zero, stop.** The effect is not identifiable from observational data
and no better estimator will rescue it. The route to an answer is *designed
price variation*, a controlled test.

*Fails loudly* [eudirectlapse renewal: 0.3% independent variation; robustness
value 0.022, so a 2% confounder nullifies the estimate; causal-forest
heterogeneity magnitudes physically implausible]. Renewal price is mechanically
pinned to the prior premium and the technical premium, so nothing is left to
identify from.

### Gate 2. Unconfoundedness

> Does the **sign** match domain knowledge, and what is the **robustness value**
> against the confounder you can actually name?

Overlap is necessary but not sufficient. Every variation-based diagnostic can
pass while the estimate is still wrong, because none of them can see a
confounder you did not measure.

*Fails silently* [new-business quotes: 87% independent variation, robustness
value 0.46, and the sign still backwards, with a higher quote predicting more
conversion]. This is the dangerous case: it looks fine and it reaches
production.

### Both gates pass

**Cautiously usable.** Not proven. Report the point estimate with the overlap
fraction, the robustness value and the named confounder on the same page.

---

## 3. Five guardrails

These are how you satisfy what TAS 100 asks of the modelling and the
communication. SS1/23 puts independent validation more sharply than any
actuarial standard does, but it binds banks, not insurers: worth borrowing,
not a requirement you can claim. The reader is your pricing or model
governance committee, and above it the board risk committee.

1. **Cross-validate.** Never a single split. [A 25% calibration miss on one
   90/10 split, against the GLM's 11%, vanished under 5-fold: 0.082 untuned and
   0.098 tuned, against 0.118. A clean headline that was a false positive.]
2. **Tune the challenger, and tune the incumbent.** Never searched against
   fixed, in either direction. [Doing this honestly moved our own headline from
   +20% to +13%. The finding survived; the number did not.]
3. **Freeze the harness.** One metric, one split, one weighting, all decided
   before you look at results.
4. **Diagnose the tails.** The aggregate loss ratio hides what a regulator cares
   about. Use decile calibration, and say so plainly when the metric is itself
   too noisy to interpret.
5. **Disclose the environment.** Pin versions and state the reason for any split
   between them.

TAS 100 v2.0, *Principles for Technical Actuarial Work* (FRC, effective 1 July
2023). SS1/23, *Model risk management principles for banks* (PRA, May 2023),
Principle 4.

---

## 4. Residual gaps: benchmark to production

This framework is derived from open research data. Be candid about what does not
transfer.

- **Scale and drift.** These books are tens to hundreds of thousands of rows and
  a decade or more old. They are static. A production book drifts, and a model
  that wins on a frozen extract may not hold its edge across a refresh cycle.
- **Frequency only.** No severity modelling, no claims inflation, no trend, no
  large-loss or reinsurance treatment. The signal threshold above is calibrated
  on frequency ranking, not on burning cost.
- **No regulatory envelope.** Nothing here models protected characteristics,
  proxy discrimination, price-walking constraints or rate filing. Those bind
  before accuracy does, and they can rule out the model the diagnostics favour.
- **No experimental variation exists in public data.** Gate 1 will fail on
  essentially any open book, because none contains designed price variation.
  The most valuable causal work cannot be demonstrated on public data at all,
  only the discipline for recognising when you are stuck.
- **Comparability costs realism.** One frozen harness is what makes the
  comparison honest, and it occasionally distorts a book. Capping exposure at
  one year is correct for freMTPL2 and compresses a genuinely multi-year
  motorcycle portfolio.
- **Nothing here is about deployment.** Monitoring, stability, challenger
  refresh cadence and rollback are out of scope and are where most of the
  operational risk actually lives.

---

**Know what your book can support.** The contribution is a discipline, not a
leaderboard.

Repository: `github.com/lubaor/Beyond_GLMs`
