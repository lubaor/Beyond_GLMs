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
| **Rich signal**, normalised Gini around 0.25 or above | GBM beats GLM in every fold, gap well above fold-to-fold noise | **Tune a GBM and adopt it.** Expect a real ranking lift [freMTPL2: 0.29 to 0.35, +20%, positive in 5/5 folds at roughly 25x the paired gap SD] |
| **Low signal**, Gini around 0.15 or below | Tuned GBM converges towards the GLM and falls short | **Stay with the GLM.** You will not get a lift, and you pay in interpretability and governance for nothing [Australian: GLM 0.0995 vs GBM 0.0858, GLM ahead in 5/5 folds] |
| **Sparse claims** | Diagnostic SD is the same order as the estimate itself; deciles with zero observed claims | **Default to the GLM** and report a non-finding. You cannot demonstrate a difference either way, and you should not defend a model you cannot validate [Swedish: 0.5528 vs 0.5564, per-fold gaps change sign; worst-decile error 1.40 +/- 1.47] |

A lift is only real if it is **calibration-free**. Check that the challenger's
worst-decile error does not deteriorate [freMTPL2: GBM 0.098 vs GLM 0.118, so
the challenger was *better* calibrated].

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

These are the substance of an SS1/23 validation story.

1. **Cross-validate.** Never a single split. [A 25% calibration miss on one
   90/10 split, against the GLM's 11%, vanished under 5-fold: 0.082 untuned and
   0.098 tuned, against 0.118. A clean headline that was a false positive.]
2. **Tune the challenger.** Never tuned against untuned, in either direction.
3. **Freeze the harness.** One metric, one split, one weighting, all decided
   before you look at results.
4. **Diagnose the tails.** The aggregate loss ratio hides what a regulator cares
   about. Use decile calibration, and say so plainly when the metric is itself
   too noisy to interpret.
5. **Disclose the environment.** Pin versions and state the reason for any split
   between them.

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
