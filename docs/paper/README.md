# Preprint materials

Everything needed to read, verify or regenerate the preprint:

> **Bias Propagation and Effective Boundaries of Debiasing in Recommender Systems: An Oracle-Based Diagnosis Study**
> Yulin Zhao · *Preprints* **2026**, 202610.2582 · [DOI 10.20944/preprints202610.2582.v2](https://doi.org/10.20944/preprints202610.2582.v2)

| File | What it is |
|---|---|
| `manuscript.md` | **Source text of v2** (corrected). Every number in it is machine-checked against `data/` by `scripts/audit_paper_numbers.py`, which runs in CI on every push. |
| `AeroRec-preprints.org-v2.pdf` | **Version of record** — the PDF typeset and published by Preprints.org (v2, posted 2026-11-06, 14 pages including cover, CC BY 4.0). **Cite this one.** |
| `AeroRec-preprints.org-v1.pdf` | **Superseded v1** (posted 2026-10-27) — kept as-is on purpose: it is the physical evidence for the `0.3576` value corrected in v2, and it matches what the `preprint-v1` tag points at. Its DOI remains resolvable and both versions are cross-linked on the platform. |
| `AeroRec-preprint.pdf` | Author-typeset version, laid out to the ChinaXiv template (structured abstract, three-line tables, sequential numeric citations). Same results, different house style. |
| `figures/fig1_framework.png` | Fig. 1 — Oracle-based diagnostic framework (schematic: data → policy → estimators → boundary) |
| `figures/fig2_propagation.png` | Fig. 2 — Ranking-level Spearman vs exposure skew τ (30 seeds, mean ± std; sign flip at τ ≈ 2–3) |
| `figures/fig3_boundary.png` | Fig. 3 — SNIPS error and gain vs `ESS/N`; the boundary sits at `ESS/N ≈ 0.358` |
| `figures/fig4_injection.png` | Fig. 4 — Second bias injection vs mixing base weight `w_base` (signed effect) |
| `figures/fig5_estimand.png` | Fig. 5 — Naive vs debiased estimates against `T_exp` and `T_pop` (the estimand, not the estimator) |
| `figures/fig6_userbias.png` | Fig. 6 — User-item granularity: pooled CTR drift under self-selection (β sweep) |

## How v2 was submitted

Preprints.org takes source files, not PDFs — the typeset PDF above is produced by the platform. The upload
manuscript is therefore regenerated from `manuscript.md` rather than stored:

```bash
python scripts/build_preprint_docx.py    # -> docs/paper/AeroRec-preprints.org-v2-draft.docx
```

The intermediate `.docx` is not committed: it is a deterministic derivative of `manuscript.md` (fixed script,
no hand edits), and the published PDF supersedes it. Its figures, tables and citation superscripts were checked
against the published PDF when v2 was posted (6/6 figures match at MAE ≤ 0.14, 8/8 tables present).

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
https://doi.org/10.20944/preprints202610.2582.v2
```

```bibtex
@article{zhao2026aerorec,
  title   = {Bias Propagation and Effective Boundaries of Debiasing in Recommender Systems: An Oracle-Based Diagnosis Study},
  author  = {Zhao, Yulin},
  journal = {Preprints},
  year    = {2026},
  number  = {202610.2582},
  doi     = {10.20944/preprints202610.2582.v2},
  url     = {https://doi.org/10.20944/preprints202610.2582.v2}
}
```

## Scope reminder

All data behind these figures is **synthetic**, generated from an analytically known CTR function. The oracle
setting is what makes bias *measurable* rather than estimated — and it is simultaneously the enabling assumption
and the main limitation. See `docs/bias_report.md` for the full narrative and the honest-boundary section of the
root README for the limitation list.
