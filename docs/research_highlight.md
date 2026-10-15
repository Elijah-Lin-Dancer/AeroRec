# Research Highlight — Bias-Propagation Diagnosis & the Debiasing Effective Boundary

*AeroRec · travel-recommendation platform · one-page research summary*
*(full report: [`docs/bias_report.md`](bias_report.md) · interactive: [`data/bias/report.html`](../data/bias/report.html))*

---

## 1. Problem

Recommenders learn from **biased logs** (exposure follows popularity, not the truth), so the CTR they estimate is
distorted. Two questions are hard to answer honestly on real data:

1. **How much** does the distortion matter — is it a level problem (estimates run high) or something worse?
2. **When** does debiasing (IPS / SNIPS / DR) actually help, and when does it backfire?

On real logs you can only *estimate* bias, never *measure* it: the latent true CTR is unobservable and the
exposure propensity is unknown. That makes both questions ill-posed on the data people usually study.

## 2. Approach — an *oracle* setting

AeroRec generates its data from an analytic function (`compute_true_ctr`), which makes the design measurable:

- the **latent true CTR is a known quantity** → bias `= biased observation − known truth` can be measured *exactly*;
- the **exposure policy is self-constructed**, so its **propensity is closed-form** `π ∝ popularity^τ`
  (τ = 0 degrades to uniform) → debiasing error comes only from the estimator, with no propensity-estimation noise.

The pipeline mirrors an industrial one (4-way recall → DeepFM CTR ranking → 5-step mixing), and the diagnosis
runs at three stages (**L1 recall / L2 ranking / L3 mixing**) so that bias can be tracked as it propagates.

## 3. Key findings (multi-seed averaged ±std, one-command reproducible)

**(a) The main harm is *inversion*, not inflation.** As exposure skew τ grows, the ranking-level Spearman
correlation against the true CTR **inverts** (500 items × 5,000 requests, 30-seed mean ± std):

| τ (exposure skew) | 0.0 | 0.5 | 1.0 | 2.0 | 3.0 |
|---|---|---|---|---|---|
| Ranking-level Spearman (L2) | **+0.87** | +0.80 | +0.63 | +0.03 | **−0.51** |
| Min impressions (coldest item) | 30.1 | 15.3 | 5.4 | 0.1 | **0.0** |

The sign flips between **τ ≈ 2 and 3** (τ=3 inversion is statistically significant, bootstrap p < 0.001). A
mean-only view would call the bias "modest" and miss that **the leaderboard is reversed** — cold items, pulled
toward the smoothing prior, are ranked *above* genuinely good ones.

**(b) Two mechanisms, not one.** *Level bias* (Bayesian smoothing α=5) exists **even at τ=0** (sample-size effect,
exposure-independent); *ordering bias* is the exposure-driven inversion above. Conflating them yields wrong
conclusions — an earlier version of this project did exactly that and was corrected.

**(c) Debiasing has an *effective boundary*, set by propensity range — not by bias strength.**
Sweeping the propensity's dynamic range gives a clean threshold:

> **`ESS/N ≈ 0.358`** — below it, IPS/SNIPS/DR gains turn **negative** (the variance cost of importance weights
> exceeds the correction benefit). The threshold is stable across scales (0.3576 → 0.3579 after a 8× scale-up).

Crucially, **`ESS/N` needs no ground truth**, so it is deployable as an online warning signal (exposed as
`ess_warning_from_log`). This is the project's single most actionable engineering takeaway.

**(d) Mixing re-injection is *signed*.** Re-injecting duplicated popularity as a mixing base score **hurts**
(project default `w_base = 0`); injecting *independent* information helps.

**(e) The oracle conclusions are robust to propensity estimation error.** Replacing the closed-form π with a
log-estimated π̂ leaves SNIPS error unchanged at this scale (delta ≈ 0, p = 0.87), and the ESS/N warning stays
**fail-safe** (false alarms possible, zero misses).

**(f) User-side bias must be modeled at the user-item granularity.** With self-selection (softmax affinity,
strength β), the pooled CTR's estimand drifts from 0.281 to 0.702 — a 2.5× inflation of apparent performance.
Debiasing with item-marginal propensities leaves a significant residual (+0.063 → +0.222, p < 0.001), while
correct user-item-level propensity debiasing keeps error at 0.004–0.037.

## 4. Why it matters (engineering relevance)

- Reframes "bias" from *a number that's too high* into **a ranking-integrity failure** — the right thing to monitor.
- Turns "should we debias?" into a **measurable, ground-truth-free decision rule** (`ESS/N`).
- The whole diagnosis is **model-free** (works on the estimator directly) and runs in seconds with a fixed seed.

## 5. Honest limitations

- **Synthetic data**: results show method validity **under controlled conditions**, not online transferability.
  The oracle setting is the *enabling assumption*, and it is a limitation as much as a strength.
- **Propensity estimation**: partially quantified (#4) — negligible at this scale, warning stays fail-safe;
  a full sensitivity sweep of the `ESS/N` boundary under estimation error remains future work.
- **User-side bias**: modeled synthetically at user-item granularity (#5/#6); real multi-user structures
  (sessions, long-term interest, cold start) and real self-selection datasets (Coat / Open Bandit) remain
  future work.
- **No new algorithm is proposed**: all estimators and metrics are standard; the contribution is the
  measurement methodology and the deployable decision rule.

## 6. Reproduce

```bash
python scripts/run_bias_experiments.py   # → 7 JSONs: propagation / injection / boundary /
                                         #    estimand_check / boundary_sweep /
                                         #    propensity_estimation / userbias
python scripts/build_bias_report.py      # → data/bias/report.html  (self-contained)
```

All numbers above are produced by these two scripts (multi-seed, byte-for-byte drift-gated in CI).
Code anchor: tag `v0.1.0`. The same code path powers the
interactive **Bias Diagnosis** page of the live demo (bilingual EN / 中文).
