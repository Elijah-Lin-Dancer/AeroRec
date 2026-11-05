# Preprint materials

Everything needed to read, verify or regenerate the preprint:

> **Bias Propagation and Effective Boundaries of Debiasing in Recommender Systems: An Oracle-Based Diagnosis Study**
> Yulin Zhao · *Preprints* **2026**, 202610.2582 · [DOI 10.20944/preprints202610.2582.v1](https://doi.org/10.20944/preprints202610.2582.v1)

| File | What it is |
|---|---|
| `manuscript.md` | **Source text of v2** (corrected). Every number in it is machine-checked against `data/` by `scripts/audit_paper_numbers.py`, which runs in CI on every push. |
| `AeroRec-preprints.org-v1.pdf` | **Version of record (v1)** — the PDF typeset and published by Preprints.org (14 pages, CC BY 4.0). Kept as-is: it is the evidence for the `0.3576` value corrected in v2. |
| `AeroRec-preprints.org-v2-draft.docx` | **v2 upload manuscript** — the corrected version submitted to Preprints.org (boundary in §4.3 and the Fig. 3 caption: `0.3576` → `0.3625`). Regenerate with `python scripts/build_preprint_docx.py`. Replaced by the publisher's PDF once v2 is posted. |
| `AeroRec-preprint.pdf` | Author-typeset version, laid out to the ChinaXiv template (structured abstract, three-line tables, sequential numeric citations). Same results, different house style. |
| `figures/fig1_framework.png` | Fig. 1 — Oracle-based diagnostic framework (schematic: data → policy → estimators → boundary) |
| `figures/fig2_propagation.png` | Fig. 2 — Ranking-level Spearman vs exposure skew τ (30 seeds, mean ± std; sign flip at τ ≈ 2–3) |
| `figures/fig3_boundary.png` | Fig. 3 — SNIPS error and gain vs `ESS/N`; the boundary sits at `ESS/N ≈ 0.358` |
| `figures/fig4_injection.png` | Fig. 4 — Second bias injection vs mixing base weight `w_base` (signed effect) |
| `figures/fig5_estimand.png` | Fig. 5 — Naive vs debiased estimates against `T_exp` and `T_pop` (the estimand, not the estimator) |
| `figures/fig6_userbias.png` | Fig. 6 — User-item granularity: pooled CTR drift under self-selection (β sweep) |

## Figures are generated from the result JSONs, not drawn by hand

Every number in these figures is read from the committed experiment outputs in `data/bias/*.json`, so the figures
cannot silently drift away from the results:

```bash
python scripts/run_bias_experiments.py   # regenerate data/bias/*.json (numpy only)
python scripts/make_paper_figures.py     # regenerate figures/ from those JSONs
```

Both steps are deterministic (fixed seeds). The figures are exported at 300–340 dpi so that the **effective** DPI
at the typeset column width still clears the ≥300 dpi requirement of the ChinaXiv template; re-running the script
reproduces the committed PNGs byte for byte.

## Citing

```text
Zhao, Y. Bias Propagation and Effective Boundaries of Debiasing in Recommender
Systems: An Oracle-Based Diagnosis Study. Preprints 2026, 202610.2582.
https://doi.org/10.20944/preprints202610.2582.v1
```

```bibtex
@article{zhao2026aerorec,
  title   = {Bias Propagation and Effective Boundaries of Debiasing in Recommender Systems: An Oracle-Based Diagnosis Study},
  author  = {Zhao, Yulin},
  journal = {Preprints},
  year    = {2026},
  number  = {202610.2582},
  doi     = {10.20944/preprints202610.2582.v1},
  url     = {https://doi.org/10.20944/preprints202610.2582.v1}
}
```

## Scope reminder

All data behind these figures is **synthetic**, generated from an analytically known CTR function. The oracle
setting is what makes bias *measurable* rather than estimated — and it is simultaneously the enabling assumption
and the main limitation. See `docs/bias_report.md` for the full narrative and the honest-boundary section of the
root README for the limitation list.
