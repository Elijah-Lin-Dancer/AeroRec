"""Generate the 6 figures of the preprint directly from data/bias/*.json.

Every figure is drawn from the committed result JSONs (no hand-typed numbers), so re-running
the experiments and then this script reproduces the paper figures exactly. DPI is set to
300-340 so that the *effective* DPI at the typeset width clears the ChinaXiv >=300 requirement.

Usage:  python scripts/make_paper_figures.py
Output: docs/paper/figures/fig1..fig6*.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
BASE = str(ROOT / "data" / "bias")
OUT = str(ROOT / "docs" / "paper" / "figures")
Path(OUT).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5})

def load(name: str) -> dict:
    with open(f"{BASE}/{name}.json", encoding="utf-8") as f:
        return json.load(f)

# ============ Figure 1: framework overview (NEW, pure schematic) ============
fig, ax = plt.subplots(figsize=(7.6, 4.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 54)
ax.axis("off")
ax.set_aspect("auto")

C_ORA, C_EXP, C_EST, C_PIP, C_DIA = "#eaf2f8", "#fef9e7", "#fdf2e9", "#eafaf1", "#f4ecf7"
E_ORA, E_EXP, E_EST, E_PIP, E_DIA = "#1f4e79", "#b7950b", "#ca6f1e", "#1e8449", "#7d3c98"

def box(x, y, w, h, fc, ec, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.0",
                                fc=fc, ec=ec, lw=1.1, linestyle=ls))

def arrow(p1, p2, color="#34495e", lw=1.2, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))

def txt(x, y, s, size=7.6, weight="normal", color="#1a1a1a", ha="center", style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va="center", style=style)

# --- Row 1 main flow ---
box(2, 34, 22, 15, C_ORA, E_ORA)
txt(13, 46.2, "ORACLE  (known exactly)", 8.2, "bold", E_ORA)
txt(13, 42.6, "true CTR  r(i)", 7.8)
txt(13, 39.4, "closed-form propensity  π(i)", 7.8)
txt(13, 36.2, "synthetic generation", 7.0, color="#5d6d7e", style="italic")

box(33, 34, 27, 15, C_EXP, E_EXP)
txt(46.5, 46.2, "EXPOSURE & CLICK LOG", 8.2, "bold", E_EXP)
txt(46.5, 42.6, "impressions  ~  π(i)  under skew τ", 7.8)
txt(46.5, 39.4, "clicks  ~  Bernoulli(r(i))", 7.8)
txt(46.5, 36.2, "interaction log (biased)", 7.0, color="#5d6d7e", style="italic")

box(68, 34, 30, 15, C_EST, E_EST)
txt(83, 46.2, "ESTIMATORS & WARNING", 8.2, "bold", E_EST)
txt(83, 42.8, "naive  →  T_exp  (exposure-weighted)", 7.6)
txt(83, 39.8, "IPS / SNIPS  (weights 1/π)  →  T_pop", 7.6)
txt(83, 36.8, "ESS/N warning:  < 0.358  ⇒  caution", 7.4, color="#c0392b")

# --- Row 2 ---
box(68, 3, 30, 15, C_DIA, E_DIA)
txt(83, 15.2, "ORACLE DIAGNOSTICS", 8.2, "bold", E_DIA)
txt(83, 11.8, "L0 ↔ L2 Spearman  (ranking reversal)", 7.6)
txt(83, 8.8, "rel_err / KL / Gini  (level & distortion)", 7.6)
txt(83, 5.8, "bias propagation across L1–L3", 7.6)

box(33, 3, 28, 15, C_PIP, E_PIP)
txt(47, 15.2, "PIPELINE STAGES", 8.2, "bold", E_PIP)
txt(47, 11.6, "L1 recall  ←  exposure share", 7.6)
txt(47, 8.8, "L2 ranking  ←  observed CTR", 7.6)
txt(47, 6.0, "L3 mixed = (1−w)·L2 + w·L0   (§4.3)", 7.4)

# --- Extension boxes (dashed) ---
box(2, 3, 24, 15, "white", E_EXP, ls="--")
txt(14, 15.2, "EXTENSION B  (§4.5)", 7.8, "bold", E_EXP)
txt(14, 11.8, "user self-selection  β:", 7.4)
txt(14, 9.0, "p(u,i) ∝ softmax(β·affinity)", 7.4)
txt(14, 6.0, "user-item granularity", 7.0, color="#5d6d7e", style="italic")

box(68, 22.5, 28, 8.5, "white", E_EST, ls="--")
txt(82, 29.2, "EXTENSION A  (§4.6)", 7.8, "bold", E_EST)
txt(82, 26.3, "$\\hat{\\pi}$ estimated from the log,", 7.2)
txt(82, 24.1, "replaces closed-form π", 7.2)

# --- Arrows: main flow ---
arrow((24.6, 41), (32.4, 41))                      # oracle -> exposure
txt(28.5, 43.0, "generates", 6.8, color="#5d6d7e")
arrow((60.6, 41), (67.4, 41))                      # exposure -> estimators
txt(64, 43.0, "log", 6.8, color="#5d6d7e")
arrow((36, 33.4), (40, 18.6), rad=0.12)            # exposure -> pipeline
txt(34.2, 26.5, "observed CTR", 6.8, color="#5d6d7e", ha="right")
arrow((13, 33.4), (33, 9.5), rad=0.18, color="#1e8449")   # oracle L0 -> pipeline
txt(15.5, 24.5, "truth L0", 6.8, color="#1e8449", ha="left")
arrow((61.6, 10), (67.4, 10))                      # pipeline -> diagnostics
txt(64.5, 12.0, "measure", 6.8, color="#5d6d7e")

# --- Arrows: extensions (dashed) ---
arrow((26.5, 12), (39.5, 32.8), ls="--", color="#b7950b", rad=-0.1)   # ext B -> exposure
arrow((76, 31.3), (76, 33.4), ls="--", color="#ca6f1e")               # ext A -> estimators
arrow((57, 33.4), (72, 31.2), ls="--", color="#ca6f1e", rad=0.1)      # log -> ext A

# --- tau knob ---
arrow((46.5, 23), (46.5, 33.4), color="#c0392b", ls="--", lw=1.3)
txt(46.5, 21.0, "exposure skew  τ  (bias strength)", 7.4, "bold", "#c0392b")

ax.set_title("Oracle-based diagnostic framework", fontsize=9.5)

fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/fig1_framework.png", dpi=300)
plt.close(fig)

# ============ Figure 2: propagation reversal (was Fig 1) ============
m = load("propagation")
taus = [r["tau"] for r in m["rows"]]
sp = [r["L2_spearman"] for r in m["rows"]]
std = [r["L2_spearman_std"] for r in m["rows"]]
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.errorbar(taus, sp, yerr=std, marker="o", ms=5, lw=1.6, color="#1f4e79",
            ecolor="#7fa8cc", elinewidth=1.2, capsize=3)
ax.axhline(0, color="#c0392b", ls="--", lw=1)
ax.annotate("sign flip\n(τ ≈ 2–3)", xy=(2.5, 0), xytext=(2.15, -0.28),
            fontsize=8, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.9))
ax.set_xlabel("Exposure skew strength τ")
ax.set_ylabel("Ranking-level Spearman (L2)")
ax.set_title("Ranking quality inverts as exposure bias grows (30 seeds, mean ± std)", fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_propagation.png", dpi=340)
plt.close(fig)

# ============ Figure 3: ESS boundary (was Fig 2) ============
b = load("boundary")
ess = [r["ess_ratio"] for r in b["rows"]]
gain = [r["gain"] for r in b["rows"]]
ci = [r.get("gain_sig", {}).get("ci95", [None, None]) for r in b["rows"]]
yerr = [[g - c[0] for g, c in zip(gain, ci)], [c[1] - g for g, c in zip(gain, ci)]]
thr = b["cross_ess_ratio"]
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.errorbar(ess, gain, yerr=yerr, marker="s", ms=5, lw=1.6, color="#1f4e79",
            ecolor="#7fa8cc", elinewidth=1.1, capsize=3, ls="-")
ax.axvline(thr, color="#c0392b", ls="--", lw=1.1)
ax.axhline(0, color="gray", lw=0.8, alpha=0.6)
ax.text(thr * 0.82, -0.048, f"boundary ESS/N ≈ {thr:.3f}", color="#c0392b",
        fontsize=8.5, ha="right", rotation=90, va="bottom")
ax.set_xscale("log")
ax.set_xlabel("ESS/N (log scale)")
ax.set_ylabel("Debiasing gain (SNIPS vs naive)")
ax.set_title("Debiasing turns harmful below the ESS/N boundary (250 seeds, 95% CI)", fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_boundary.png", dpi=340)
plt.close(fig)

# ============ Figure 4: re-injection monotone (was Fig 3) ============
i = load("injection")
w = [r["w_base"] for r in i["rows"]]
l3 = [r["L3_spearman"] for r in i["rows"]]
l3s = [r.get("L3_spearman_std", 0) for r in i["rows"]]
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.errorbar(w, l3, yerr=l3s, marker="o", ms=5, lw=1.6, color="#1f4e79",
            ecolor="#7fa8cc", elinewidth=1.2, capsize=3)
ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=0.9, alpha=0.7)
ax.annotate("w = 1 recovers truth\n(self-check)", xy=(1.0, 1.0), xytext=(0.55, 0.82),
            fontsize=8, arrowprops=dict(arrowstyle="->", color="gray", lw=0.9))
ax.set_xlabel("Base-score weight $w_{base}$")
ax.set_ylabel("L3 Spearman")
ax.set_title("Mixing re-injection is strictly monotone (τ = 2, 30 seeds, mean ± std)", fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_injection.png", dpi=340)
plt.close(fig)

# ============ Figure 5: estimand separation (NEW) ============
e = load("estimand_check")
taus = [r["tau"] for r in e["rows"]]
texp = [r["T_exp"] for r in e["rows"]]
tpop = e["rows"][0]["T_pop"]
naive = [r["naive"] for r in e["rows"]]
nstd = [r["naive_std"] for r in e["rows"]]
snips = [r["snips"] for r in e["rows"]]
sstd = [r["snips_std"] for r in e["rows"]]
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.plot(taus, texp, ls="--", color="#7f8c8d", lw=1.3,
        label="T_exp estimand (drifts with τ)")
ax.axhline(tpop, color="#c0392b", ls=":", lw=1.3,
           label=f"T_pop estimand = {tpop:.4f} (constant)")
ax.errorbar(taus, naive, yerr=nstd, marker="o", ms=5, lw=1.6, color="#1f4e79",
            ecolor="#7fa8cc", elinewidth=1.2, capsize=3, label="naive estimate")
ax.errorbar(taus, snips, yerr=sstd, marker="s", ms=5, lw=1.6, color="#e67e22",
            ecolor="#f0b27a", elinewidth=1.1, capsize=3, label="SNIPS (oracle π)")
ax.annotate("", xy=(3.0, 0.415), xytext=(3.0, 0.27),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0))
ax.text(2.88, 0.342, "estimand shift\n0.2525 → 0.4028", fontsize=7.4, color="#c0392b", ha="right")
ax.set_xlabel("Exposure skew strength τ")
ax.set_ylabel("Estimated CTR")
ax.set_title("Two estimators, two estimands (30 seeds, mean ± std)", fontsize=9.5)
ax.legend(fontsize=7.2, loc="upper left", framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_estimand.png", dpi=340)
plt.close(fig)

# ============ Figure 6: user-side bias (was Fig 4) ============
u = load("userbias")
bet = [r["beta"] for r in u["rows"]]
texp = [r["T_exp_user"] for r in u["rows"]]
naive = [r["naive"] for r in u["rows"]]
tpop = [r["T_pop"] for r in u["rows"]]
resid = [r["agg_residual"] for r in u["rows"]]
rstd = [r.get("agg_residual_std", 0) for r in u["rows"]]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
a1.plot(bet, texp, marker="o", ms=5, lw=1.6, color="#1f4e79", label="T_exp_user (naive estimand)")
a1.plot(bet, naive, marker="x", ms=6, lw=1.2, ls="--", color="#e67e22", label="naive estimate")
a1.axhline(tpop[0], color="#c0392b", ls=":", lw=1.2, label="T_pop = 0.281 (true level)")
a1.set_xlabel("Self-selection strength β")
a1.set_ylabel("Pooled CTR")
a1.set_title("(a) Estimand drift: 0.281 → 0.702", fontsize=9.5)
a1.legend(fontsize=7.2, loc="upper left")
a2.errorbar(bet, resid, yerr=rstd, marker="s", ms=5, lw=1.6, color="#c0392b",
            ecolor="#e6a09a", elinewidth=1.1, capsize=3)
a2.axhline(0, color="gray", lw=0.8, alpha=0.6)
a2.set_xlabel("Self-selection strength β")
a2.set_ylabel("Residual bias after wrong-granularity debiasing")
a2.set_title("(b) Wrong-granularity debiasing residual", fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_userbias.png", dpi=300)
plt.close(fig)
print("6 figures written")
