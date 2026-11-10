"""Generate the 6 figures of the preprint directly from data/bias/*.json.

Every figure is drawn from the committed result JSONs (no hand-typed numbers), so re-running
the experiments and then this script reproduces the paper figures exactly. DPI is set to
300-340 so that the *effective* DPI at the typeset width clears the ChinaXiv >=300 requirement.

The reader-facing strings are Chinese (for the Chinese-language thesis).  Matplotlib ships no
CJK font, and on many systems the CJK fonts exist only as .ttc collections whose *first* face
is Japanese (Noto Sans CJK JP) -- rendering Simplified Chinese with that face yields Japanese
glyph variants for ~70 common characters.  We therefore look for a Simplified-Chinese (SC)
face: one already registered with matplotlib, or one extracted from a system .ttc with
fontTools (a hard matplotlib dependency) and registered here.

Usage:  python scripts/make_paper_figures.py
Output: docs/paper/figures/fig1..fig6*.png
"""
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
BASE = str(ROOT / "data" / "bias")
OUT = str(ROOT / "docs" / "paper" / "figures")
Path(OUT).mkdir(parents=True, exist_ok=True)


# ============ Chinese font setup ============
# Simplified-Chinese faces first: their glyph shapes follow mainland Chinese
# orthography, which is what a Chinese-language thesis needs.
CJK_SC_FAMILIES = (
    "Noto Sans CJK SC", "Source Han Sans SC", "Noto Sans SC", "Noto Sans CJK TC",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei",
    "PingFang SC", "Heiti SC", "Noto Serif CJK SC", "Source Han Serif SC",
)
# Last-resort CJK faces: they do contain every Chinese character we need, but some
# glyphs follow Japanese/Traditional regional forms.
CJK_FALLBACK_FAMILIES = ("Noto Sans CJK JP", "Noto Serif CJK JP")


def _registered_families() -> set[str]:
    return {f.name for f in fm.fontManager.ttflist}


def _extract_sc_faces_from_ttc(family: str) -> bool:
    """Register the Regular/Bold faces of *family* hidden inside system .ttc files.

    matplotlib only ever reads face 0 of a .ttc, which for Noto Sans CJK is the
    Japanese variant.  fontTools (a required matplotlib dependency) can pull the SC
    face out; we cache the result under the matplotlib cache directory.  Only the
    two faces we actually use are extracted, to keep the cache small.
    """
    try:
        from fontTools.ttLib import TTCollection
    except ImportError:  # pragma: no cover - fontTools ships with matplotlib
        return False

    try:
        cache = Path(matplotlib.get_cachedir()) / "aerorec-cjk"
        cache.mkdir(parents=True, exist_ok=True)
    except OSError:  # pragma: no cover - unwritable cache dir
        return False

    found = False
    for path in fm.findSystemFonts():
        if not path.lower().endswith(".ttc"):
            continue
        try:
            collection = TTCollection(path, lazy=True)
            faces = list(collection.fonts)
        except Exception:
            continue
        for face in faces:
            try:
                name = face["name"].getDebugName(1)
                style = face["name"].getDebugName(2) or "Regular"
            except Exception:
                continue
            if name != family or style not in ("Regular", "Bold"):
                continue
            out = cache / f"{name.replace(' ', '')}-{style}.otf"
            if not out.exists():
                try:
                    face.save(str(out))
                except Exception:
                    continue
            try:
                fm.fontManager.addfont(str(out))
            except Exception:
                continue
            found = True
    return found


def _install_cjk_font() -> None:
    """Best-effort install of a CJK font when the machine has none."""
    if shutil.which("apt-get"):
        subprocess.run(["apt-get", "install", "-y", "fonts-noto-cjk"],
                       check=False, capture_output=True)


def setup_cjk_font() -> str:
    """Make a Chinese-capable font available and return the family name to use."""
    registered = _registered_families()
    for name in CJK_SC_FAMILIES:
        if name in registered:
            return name
    # Not registered with matplotlib: it may still sit inside a .ttc collection.
    for name in CJK_SC_FAMILIES:
        if _extract_sc_faces_from_ttc(name):
            registered = _registered_families()
            if name in registered:
                return name
    _install_cjk_font()
    registered = _registered_families()
    for name in CJK_SC_FAMILIES:
        if name in registered:
            return name
    for name in CJK_FALLBACK_FAMILIES:
        if name in registered:
            print(f"warning: no Simplified-Chinese face found, falling back to {name}")
            return name
    raise RuntimeError(
        "未找到可用的中文字体（Noto Sans CJK SC / 思源黑体 / 文泉驿 等）。"
        "请先安装，例如：apt-get install -y fonts-noto-cjk")


CJK_FONT = setup_cjk_font()

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
                     # DejaVu Sans keeps rendering Latin/digits exactly as in the
                     # English figures; the CJK family is the fallback that supplies
                     # the Chinese glyphs (matplotlib resolves missing glyphs down
                     # the font.family list).
                     "font.family": ["DejaVu Sans", CJK_FONT],
                     "axes.unicode_minus": False})

# Strings that mix mathtext ($...$) with Chinese are handed to the mathtext parser
# as a whole, and mathtext resolves its "default" face with findfont() -- i.e. the
# *first* family only, with no fallback chain.  Putting the CJK family first for
# those few labels keeps the Chinese glyphs; the math symbols themselves still come
# from the dejavusans math fontset, so they look unchanged.
MATH_TEXT_FAMILY = [CJK_FONT, "DejaVu Sans"]


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

def txt(x, y, s, size=7.6, weight="normal", color="#1a1a1a", ha="center", style="normal",
        family=None):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va="center",
            style=style, fontfamily=family)

# --- Row 1 main flow ---
box(2, 34, 22, 15, C_ORA, E_ORA)
txt(13, 46.2, "真值基准（精确已知）", 8.2, "bold", E_ORA)
txt(13, 42.6, "生成器真值点击率  r(i)", 7.8)
txt(13, 39.4, "闭式倾向分  π(i)", 7.8)
txt(13, 36.2, "合成数据生成", 7.0, color="#5d6d7e", style="italic")

box(33, 34, 27, 15, C_EXP, E_EXP)
txt(46.5, 46.2, "曝光与点击日志", 8.2, "bold", E_EXP)
txt(46.5, 42.6, "曝光量  ~  π(i)  （偏斜 τ）", 7.8)
txt(46.5, 39.4, "点击  ~  Bernoulli(r(i))", 7.8)
txt(46.5, 36.2, "交互日志（有偏）", 7.0, color="#5d6d7e", style="italic")

box(68, 34, 30, 15, C_EST, E_EST)
txt(83, 46.2, "估计器与告警", 8.2, "bold", E_EST)
txt(83, 42.8, "朴素估计  →  T_exp  （曝光加权）", 7.6)
txt(83, 39.8, "IPS / 自归一化 IPS（权重 1/π）→  T_pop", 7.6)
txt(83, 36.8, "ESS/N 告警：  < 0.358  ⇒  需谨慎", 7.4, color="#c0392b")

# --- Row 2 ---
box(68, 3, 30, 15, C_DIA, E_DIA)
txt(83, 15.2, "偏差诊断", 8.2, "bold", E_DIA)
txt(83, 11.8, "L0 ↔ L2 斯皮尔曼相关系数（排序反转）", 7.6)
txt(83, 8.8, "rel_err / KL / 基尼（水平与失真）", 7.6)
txt(83, 5.8, "L1–L3 之间的偏差传播", 7.6)

box(33, 3, 28, 15, C_PIP, E_PIP)
txt(47, 15.2, "流水线阶段", 8.2, "bold", E_PIP)
txt(47, 11.6, "L1 召回阶段  ←  曝光占比", 7.6)
txt(47, 8.8, "L2 排序阶段  ←  观测点击率", 7.6)
txt(47, 6.0, "L3 混排阶段 = (1−w)·L2 + w·L0   (§4.3)", 7.4)

# --- Extension boxes (dashed) ---
box(2, 3, 24, 15, "white", E_EXP, ls="--")
txt(14, 15.2, "扩展 B  (§4.5)", 7.8, "bold", E_EXP)
txt(14, 11.8, "用户自选择  β：", 7.4)
txt(14, 9.0, "p(u,i) ∝ softmax(β·亲和度)", 7.4)
txt(14, 6.0, "用户-物品粒度", 7.0, color="#5d6d7e", style="italic")

box(68, 22.5, 28, 8.5, "white", E_EST, ls="--")
txt(82, 29.2, "扩展 A  (§4.6)", 7.8, "bold", E_EST)
txt(82, 26.3, "$\\hat{\\pi}$ 由日志估计得到，", 7.2, family=MATH_TEXT_FAMILY)
txt(82, 24.1, "替代闭式 π", 7.2)

# --- Arrows: main flow ---
arrow((24.6, 41), (32.4, 41))                      # oracle -> exposure
txt(28.5, 43.0, "生成", 6.8, color="#5d6d7e")
arrow((60.6, 41), (67.4, 41))                      # exposure -> estimators
txt(64, 43.0, "日志", 6.8, color="#5d6d7e")
arrow((36, 33.4), (40, 18.6), rad=0.12)            # exposure -> pipeline
txt(34.2, 26.5, "观测点击率", 6.8, color="#5d6d7e", ha="right")
arrow((13, 33.4), (33, 9.5), rad=0.18, color="#1e8449")   # oracle L0 -> pipeline
txt(15.5, 24.5, "真值排序 L0", 6.8, color="#1e8449", ha="left")
arrow((61.6, 10), (67.4, 10))                      # pipeline -> diagnostics
txt(64.5, 12.0, "度量", 6.8, color="#5d6d7e")

# --- Arrows: extensions (dashed) ---
arrow((26.5, 12), (39.5, 32.8), ls="--", color="#b7950b", rad=-0.1)   # ext B -> exposure
arrow((76, 31.3), (76, 33.4), ls="--", color="#ca6f1e")               # ext A -> estimators
arrow((57, 33.4), (72, 31.2), ls="--", color="#ca6f1e", rad=0.1)      # log -> ext A

# --- tau knob ---
arrow((46.5, 23), (46.5, 33.4), color="#c0392b", ls="--", lw=1.3)
txt(46.5, 21.0, "曝光偏斜  τ  （偏差强度）", 7.4, "bold", "#c0392b")

ax.set_title("基于真值基准的偏差诊断框架", fontsize=9.5)

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
ax.annotate("符号翻转\n(τ ≈ 2–3)", xy=(2.5, 0), xytext=(2.15, -0.28),
            fontsize=8, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.9))
ax.set_xlabel("曝光偏斜强度 τ")
ax.set_ylabel("排序阶段斯皮尔曼相关系数 (L2)")
ax.set_title("曝光偏差增大时排序质量反转（30 个随机种子，均值 ± 标准差）", fontsize=9.5)
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
ax.text(thr * 0.82, -0.048, f"边界 ESS/N ≈ {thr:.3f}", color="#c0392b",
        fontsize=8.5, ha="right", rotation=90, va="bottom")
ax.set_xscale("log")
ax.set_xlabel("有效样本量比 ESS/N（对数刻度）")
ax.set_ylabel("去偏净收益（自归一化 IPS 对比朴素估计）")
ax.set_title("ESS/N 边界以下去偏反而有害（250 个随机种子，95% 置信区间）", fontsize=9.5)
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
ax.annotate("w = 1 恢复真值\n（自校验）", xy=(1.0, 1.0), xytext=(0.55, 0.82),
            fontsize=8, arrowprops=dict(arrowstyle="->", color="gray", lw=0.9))
ax.set_xlabel("基础分权重 $w_{base}$", fontfamily=MATH_TEXT_FAMILY)
ax.set_ylabel("L3 斯皮尔曼相关系数")
ax.set_title("混排二次注入严格单调（τ = 2，30 个随机种子，均值 ± 标准差）", fontsize=9.5)
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
        label="T_exp 估计目标（随 τ 漂移）")
ax.axhline(tpop, color="#c0392b", ls=":", lw=1.3,
           label=f"T_pop 估计目标 = {tpop:.4f}（恒定）")
ax.errorbar(taus, naive, yerr=nstd, marker="o", ms=5, lw=1.6, color="#1f4e79",
            ecolor="#7fa8cc", elinewidth=1.2, capsize=3, label="朴素估计")
ax.errorbar(taus, snips, yerr=sstd, marker="s", ms=5, lw=1.6, color="#e67e22",
            ecolor="#f0b27a", elinewidth=1.1, capsize=3, label="自归一化 IPS（真值 π）")
ax.annotate("", xy=(3.0, 0.415), xytext=(3.0, 0.27),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0))
ax.text(2.88, 0.342, "估计目标漂移\n0.2525 → 0.4028", fontsize=7.4, color="#c0392b", ha="right")
ax.set_xlabel("曝光偏斜强度 τ")
ax.set_ylabel("估计点击率")
ax.set_title("两个估计量、两个估计目标（30 个随机种子，均值 ± 标准差）", fontsize=9.5)
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
a1.plot(bet, texp, marker="o", ms=5, lw=1.6, color="#1f4e79", label="T_exp_user（朴素估计目标）")
a1.plot(bet, naive, marker="x", ms=6, lw=1.2, ls="--", color="#e67e22", label="朴素估计")
a1.axhline(tpop[0], color="#c0392b", ls=":", lw=1.2, label="T_pop = 0.281（真值水平）")
a1.set_xlabel("自选择强度 β")
a1.set_ylabel("合并点击率")
a1.set_title("(a) 估计目标漂移：0.281 → 0.702", fontsize=9.5)
a1.legend(fontsize=7.2, loc="upper left")
a2.errorbar(bet, resid, yerr=rstd, marker="s", ms=5, lw=1.6, color="#c0392b",
            ecolor="#e6a09a", elinewidth=1.1, capsize=3)
a2.axhline(0, color="gray", lw=0.8, alpha=0.6)
a2.set_xlabel("自选择强度 β")
a2.set_ylabel("错误粒度去偏后的残余偏差")
a2.set_title("(b) 错误粒度去偏残余", fontsize=9.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_userbias.png", dpi=300)
plt.close(fig)
print("6 figures written")
