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

## 3. Key findings (seed-fixed, one-command reproducible)

**(a) The main harm is *inversion*, not inflation.** As exposure skew τ grows, the ranking-level Spearman
correlation against the true CTR **inverts**:

| τ (exposure skew) | 0.0 | 0.5 | 1.0 | 2.0 | 3.0 |
|---|---|---|---|---|---|
| Ranking-level Spearman (L2) | **+0.85** | +0.81 | +0.62 | −0.05 | **−0.50** |
| Min impressions (coldest item) | 32 | 21 | 20 | 0 | **0** |

The sign flips near **τ ≈ 2.0**. A mean-only view would call the bias "modest" and miss that **the leaderboard is
reversed** — cold items, pulled toward the smoothing prior, are ranked *above* genuinely good ones.

**(b) Two mechanisms, not one.** *Level bias* (Bayesian smoothing α=5) exists **even at τ=0** (sample-size effect,
exposure-independent); *ordering bias* is the exposure-driven inversion above. Conflating them yields wrong
conclusions — an earlier version of this project did exactly that and was corrected.

**(c) Debiasing has an *effective boundary*, set by propensity range — not by bias strength.**
Sweeping the propensity's dynamic range gives a clean threshold:

> **`ESS/N ≈ 0.358`** — below it, IPS/SNIPS/DR gains turn **negative** (the variance cost of importance weights
> exceeds the correction benefit).

Crucially, **`ESS/N` needs no ground truth**, so it is deployable as an online warning signal (exposed as
`ess_warning_from_log`). This is the project's single most actionable engineering takeaway.

**(d) Mixing re-injection is *signed*.** Re-injecting duplicated popularity as a mixing base score **hurts**
(project default `w_base = 0`); injecting *independent* information helps.

## 4. Why it matters (engineering relevance)

- Reframes "bias" from *a number that's too high* into **a ranking-integrity failure** — the right thing to monitor.
- Turns "should we debias?" into a **measurable, ground-truth-free decision rule** (`ESS/N`).
- The whole diagnosis is **model-free** (works on the estimator directly) and runs in seconds with a fixed seed.

## 5. Honest limitations

- **Synthetic data**: results show method validity **under controlled conditions**, not online transferability.
  The oracle setting is the *enabling assumption*, and it is a limitation as much as a strength.
- **Scale**: propagation matrix is 60 items × 600 requests × 5 slots; boundary uses 20 000 records.
  Stability of the `ESS/N` boundary at larger scale is **not verified**.
- **No user-side preference bias** is modeled (only exposure-side).

## 6. Reproduce

```bash
python scripts/run_bias_experiments.py   # → data/bias/{propagation,injection,boundary}.json
python scripts/build_bias_report.py      # → data/bias/report.html  (self-contained)
```

All numbers above are produced by these two scripts under a fixed seed. The same code path powers the
interactive **Bias Diagnosis** page of the live demo (bilingual EN / 中文).
