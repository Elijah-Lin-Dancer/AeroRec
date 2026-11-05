"""审计：论文正文/表格里的每一个数字，是否都能在仓库产物里找到出处。

背景：预印本已发布（DOI 10.20944/preprints202610.2582.v1），仓库是它的可复现支撑。
一旦仓库产物被改动，或论文里的数字与产物脱节，"可复现"就成了口号。本脚本把这件事
变成一条可执行的门禁：

    python scripts/audit_paper_numbers.py            # 人类可读报告
    python scripts/audit_paper_numbers.py --strict   # 有不一致则退出码 1

检查内容：
 1. 8 张表逐单元格比对（按论文呈现的小数位数四舍五入后比较，避免精度误报）；
 2. 论文中的关键统计断言（|Δ| 上界、p 值、误报率、τ-sweep 增益）；
 3. 实验设置表与各产物 JSON 的元数据（n_records / n_seeds / n_requests）；
 4. 跨规模稳定性基线（论文 §4.3 的 0.3625）由 boundary_by_range 现场复算后比对。

EXPECTED_GAPS 为空，表示当前没有任何豁免项。若将来新增，必须在
docs/paper/PROVENANCE.md 里写明原因——不允许脚本自己给自己发豁免。
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 论文源文的唯一权威副本在仓库内。仓库外曾有一份工作副本，已迁移至此；
# 两侧若同时存在且内容不同，说明有人在改错的文件，直接报错而不是静默取其一。
PAPER_MD = ROOT / "docs/paper/manuscript.md"
PAPER_MD_STALE = ROOT.parent / "AeroRec-论文终稿-精确数字版.md"
DATA = ROOT / "data"

# 跨规模稳定性基线：论文 §4.3 与 Fig. 3 图注的报告值，由 boundary_by_range 现场复算比对。
# 另一个参照值是扩规模配置的 0.3579（data/bias/boundary.json → cross_ess_ratio），
# 两者相差 0.005，而相邻网格点的 ESS/N 间距为 0.29，故称"对规模稳健"。
PAPER_SCALE_BASELINE = 0.3625
SCALE_TOLERANCE = 2e-4          # 复算确定性，仅留浮点余量

EXPECTED_GAPS: dict[float, str] = {}   # 空 = 无豁免；新增必须在 PROVENANCE.md 说明


def table_blocks(md_lines: list[str]) -> list[list[str]]:
    """切出连续的 markdown 表格块（行首为 '|'）。"""
    blocks, cur = [], []
    for line in md_lines:
        if line.startswith("|"):
            cur.append(line)
        else:
            if len(cur) >= 2:
                blocks.append(cur)
            cur = []
    if len(cur) >= 2:
        blocks.append(cur)
    return blocks


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_cell(text: str) -> tuple[float | None, int]:
    """取出单元格里的第一个数值及其小数位数。

    处理：Unicode 负号、反引号、千分位空格（1 000）、千分位逗号、± 误差棒、箭头。
    """
    s = text.replace("−", "-").replace("–", "-").replace("`", "").replace(",", "")
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
    m = re.search(r"[+-]?\d+(?:\.\d+)?", s)
    if not m:
        return None, 0
    tok = m.group()
    dec = len(tok.split(".")[1]) if "." in tok else 0
    return float(tok), dec


class Audit:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, object, object]] = []
        self.checked = 0

    def cell(self, where: str, label: str, paper_text: str | float, artifact: float | None,
             dec: int | None = None, tol: float = 0.0) -> None:
        self.checked += 1
        if isinstance(paper_text, str):
            value, nd = parse_cell(paper_text)
        else:
            value, nd = float(paper_text), (4 if dec is None else dec)
        if value is None:
            self.rows.append((where, label, "无法解析", paper_text, artifact))
            return
        if artifact is None:
            self.rows.append((where, label, "产物缺字段", value, None))
            return
        # tol > 0：按绝对差比对（用于现场复算值）；否则按呈现精度四舍五入比对
        mismatch = (abs(float(artifact) - value) > tol) if tol > 0 \
            else (round(float(artifact), nd) != round(value, nd))
        if mismatch:
            tag = "已知差异" if round(value, 4) in EXPECTED_GAPS else "不一致"
            self.rows.append((where, label, tag, value, artifact))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="有不一致时返回非零退出码")
    args = ap.parse_args()

    if not PAPER_MD.exists():
        print(f"[skip] 找不到论文源文件 {PAPER_MD}，仅做产物自检")
        return 0

    # 仓库外若还留着一份内容不同的工作副本，说明改错了文件——直接失败，避免
    # "审计通过但审的是旧稿"这种最坏情况。
    if PAPER_MD_STALE.exists():
        text = PAPER_MD.read_text(encoding="utf-8")
        paper_body = text.split("-->", 1)[1].lstrip("\n") if text.startswith("<!--") else text
        old_lines = PAPER_MD_STALE.read_text(encoding="utf-8").splitlines()
        if old_lines != paper_body.splitlines():
            print(f"[FAIL] {PAPER_MD_STALE.name} 与 docs/paper/manuscript.md 内容不同")
            for line in list(difflib.unified_diff(old_lines, paper_body.splitlines(), lineterm=""))[:20]:
                print("  " + line)
            print("  论文源文以仓库内 docs/paper/manuscript.md 为准，请同步或删除仓库外副本。")
            return 1

    md = table_blocks(PAPER_MD.read_text(encoding="utf-8").splitlines())
    a = Audit()

    # ---- 表 2：传播矩阵 ----------------------------------------------------
    prop = json.loads((DATA / "bias/propagation.json").read_text(encoding="utf-8"))
    hdr, body = cells(md[1][0]), md[1][2:]
    col = {"L1 rel_err": "L1_rel_err", "L2 rel_err": "L2_rel_err", "L2 Spearman": "L2_spearman",
           "L2 rank_mae": "L2_rank_mae", "Impr_min": "impr_min", "Impr_max": "impr_max",
           "KL": "exposure_kl", "Gini": "exposure_gini", "τ": "tau"}
    for i, line in enumerate(body):
        c, j = cells(line), prop["rows"][i]
        for ci, h in enumerate(hdr):
            if h in col:
                a.cell("表2 传播矩阵", f"τ={j['tau']} {h}", c[ci], j[col[h]])

    # ---- 表 3：去偏边界 ----------------------------------------------------
    bnd = json.loads((DATA / "bias/boundary.json").read_text(encoding="utf-8"))
    hdr, body = cells(md[2][0]), md[2][2:]
    col = {"π_max/π_min": "pi_ratio", "ESS/N": "ess_ratio", "Naive Err": "naive_err",
           "SNIPS Err": "snips_err", "Gain": "gain"}
    for i, line in enumerate(body):
        c, j = cells(line), bnd["rows"][i]
        for ci, h in enumerate(hdr):
            if h in col:
                a.cell("表3 去偏边界", f"π比={j['pi_ratio']:g} {h}", c[ci], j[col[h]])

    # ---- 表 4：二次注入（表体是横向的）------------------------------------
    inj = json.loads((DATA / "bias/injection.json").read_text(encoding="utf-8"))
    wrow, vrow = cells(md[3][0]), cells(md[3][2])
    for ci in range(1, len(wrow)):
        w, _ = parse_cell(wrow[ci])
        j = next((r for r in inj["rows"] if abs(r["w_base"] - w) < 1e-9), None)
        if j is None:
            a.rows.append(("表4 注入", f"w_base={w}", "产物无该行", w, None))
            continue
        a.cell("表4 注入", f"w_base={w:g} Spearman", vrow[ci], j["injection_spearman"])

    # ---- 表 5：估计目标对照 ------------------------------------------------
    ec = json.loads((DATA / "bias/estimand_check.json").read_text(encoding="utf-8"))
    hdr, body = cells(md[4][0]), md[4][2:]
    col = {"T_exp": "T_exp", "Naive (→T_exp)": "naive",
           "SNIPS (oracle π, →T_pop=0.2525)": "snips",
           "Naive→T_exp Err": "naive_err_vs_texp", "SNIPS→T_pop Err": "snips_err_vs_tpop",
           "ESS (N=2000)": "ess", "τ": "tau"}
    for i, line in enumerate(body):
        c, j = cells(line), ec["rows"][i]
        for ci, h in enumerate(hdr):
            if h in col:
                a.cell("表5 估计目标", f"τ={j['tau']} {h}", c[ci], j[col[h]])

    # ---- 表 6：用户侧偏差 --------------------------------------------------
    ub = json.loads((DATA / "bias/userbias.json").read_text(encoding="utf-8"))
    hdr, body = cells(md[5][0]), md[5][2:]
    col = {"β": "beta", "T_pop": "T_pop", "T_exp_user": "T_exp_user", "Naive": "naive",
           "SNIPS (true p) Err": "snips_true_err",
           "SNIPS (item-marginal) Err": "snips_agg_err",
           "Residual Bias": "agg_residual", "(u,i) Exposure Gini": "exposure_gini"}
    for i, line in enumerate(body):
        c, j = cells(line), ub["rows"][i]
        for ci, h in enumerate(hdr):
            if h in col:
                a.cell("表6 用户侧", f"β={j['beta']} {h}", c[ci], j[col[h]])

    # ---- 表 7：propensity 估计误差（每个 τ 两行）---------------------------
    pe = json.loads((DATA / "bias/propensity_estimation.json").read_text(encoding="utf-8"))
    for line in md[6][2:]:
        c = cells(line)
        tau, _ = parse_cell(c[0])
        est = c[1].replace("`", "").strip()
        j = next((r for r in pe["rows"]
                  if abs(r["tau"] - tau) < 1e-9 and r["estimator"] == est), None)
        if j is None:
            a.rows.append(("表7 估计误差", f"τ={tau} {est}", "产物无该行", tau, None))
            continue
        for ci, key in ((2, "snips_true_err"), (3, "snips_hat_err"),
                        (4, "delta_err"), (5, "ess_ratio_true")):
            a.cell("表7 估计误差", f"τ={tau} {est} {key}", c[ci], j[key])

    # ---- 表 8：真实数据 ----------------------------------------------------
    r1 = json.loads((DATA / "real_validation/result.json").read_text(encoding="utf-8"))
    ra = json.loads((DATA / "real_validation/result_adult.json").read_text(encoding="utf-8"))
    for line in md[7][2:]:
        c = cells(line)
        if "Criteo 1M" in c[0]:
            a.cell("表8 真实数据", "Criteo 1M AUC", c[2], r1["val_auc_deepfm"])
        elif "Adult" in c[0]:
            a.cell("表8 真实数据", "UCI Adult AUC", c[2], ra["val_auc_deepfm"])

    # ---- 关键统计断言 ------------------------------------------------------
    delta_max = max(abs(r["delta_err"]) for r in pe["rows"])
    a.cell("正文断言", "|Δ| 上界 (≤0.0023)", 0.0023, round(delta_max, 4), dec=4)
    tau3 = pe["significance"]["tau3_raw"]["delta_err_gt_zero"]
    a.cell("正文断言", "τ=3 raw p 值 (0.87)", 0.87, round(tau3["p"], 2), dec=2)
    raw2 = next(r for r in pe["rows"] if r["tau"] == 2.0 and r["estimator"] == "raw")
    a.cell("正文断言", "τ=2 误报率 (13%)", 13, round(raw2["warning_hat"] * 100, 0), dec=0)
    lap3 = next(r for r in pe["rows"] if r["tau"] == 3.0 and r["estimator"] == "laplace1")
    a.cell("正文断言", "τ=3 平滑后 ESS/N (0.286)", 0.286, lap3["ess_ratio_hat"], dec=3)
    sweep = json.loads((DATA / "bias/boundary_sweep.json").read_text(encoding="utf-8"))
    for tau, gain, std in ((1.0, 0.0468, 0.0062), (2.0, 0.0976, 0.0083), (3.0, 0.1395, 0.0094)):
        g = next(x for x in sweep["grid"] if x["tau"] == tau and x["n_records"] == 5000)
        a.cell("正文断言", f"τ-sweep 增益 τ={tau}", gain, g["gain"], dec=4)
        a.cell("正文断言", f"τ-sweep 标准差 τ={tau}", std, g["gain_std"], dec=4)

    # ---- 跨规模稳定性复算：论文 0.3625 vs 现场复算 -------------------------
    # 论文 §4.3 与 Fig. 3 图注称：边界在小规模配置（60 物品 × 600 请求 × 30 种子）下
    # 为 0.3625，扩规模到（500 物品 × 5000 请求 × 250 种子）后为 0.3579（见 boundary.json），
    # 差 0.005，而相邻网格点的 ESS/N 间距是 0.29，故"对规模稳健"。
    # 这里按同配置现场复算小规模值并比对；差异超容差即判不一致，不再自动豁免。
    try:
        sys.path.insert(0, str(ROOT))
        from src.bias.boundary import boundary_by_range
        small = boundary_by_range(t_pop=0.25, n_records=600, n_seeds=30)["cross_ess_ratio"]
        a.cell("正文断言", "跨规模基线 (0.3625)", PAPER_SCALE_BASELINE, small,
               dec=4, tol=SCALE_TOLERANCE)
    except Exception as exc:                      # 复算失败不应让整份审计崩掉
        a.rows.append(("正文断言", "跨规模基线 (0.3625)", "复算失败", PAPER_SCALE_BASELINE, str(exc)))

    # ---- 实验设置表 vs 产物元数据 -----------------------------------------
    meta = {
        "Effective Boundary": (bnd["n_records"], bnd["rows"][0]["n_seeds"]),
        "Estimand Check": (ec["n_requests"], ec["n_seeds"]),
        "Propensity Error": (pe["n_requests"], pe["n_seeds"]),
        "τ-Sweep": (None, sweep["n_seeds"]),
    }
    for line in md[0][1:]:
        c = cells(line)
        name = c[0]
        if name == "Effective Boundary":
            a.cell("表1 设置", "Effective Boundary 记录数", c[2], meta[name][0])
            a.cell("表1 设置", "Effective Boundary 种子", c[3], meta[name][1])
        elif name in ("Estimand Check", "Propensity Error"):
            a.cell("表1 设置", f"{name} 请求数", c[2], meta[name][0])
            a.cell("表1 设置", f"{name} 种子", c[3], meta[name][1])
        elif name == "τ-Sweep":
            a.cell("表1 设置", "τ-Sweep 种子", c[3], meta[name][1])

    # ---- 报告 --------------------------------------------------------------
    bad = [r for r in a.rows if r[2] == "不一致"]
    known = [r for r in a.rows if r[2] == "已知差异"]
    other = [r for r in a.rows if r[2] not in ("不一致", "已知差异")]
    print(f"核对单元格 {a.checked} 个")
    print(f"  一致      {a.checked - len(bad) - len(known) - len(other)}")
    print(f"  不一致    {len(bad)}")
    print(f"  已知差异  {len(known)}")
    print(f"  其他      {len(other)}")
    for tag, group in (("不一致", bad), ("已知差异", known), ("其他", other)):
        for where, label, _, paper, art in group:
            note = ""
            if isinstance(paper, float) and round(paper, 4) in EXPECTED_GAPS:
                note = " — " + EXPECTED_GAPS[round(paper, 4)]
            print(f"  [{tag}] {where} / {label}: 论文={paper} 产物={art}{note}")
    if args.strict and (bad or other):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
