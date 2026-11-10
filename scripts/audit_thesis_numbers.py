"""审计：毕设素材文档里引用的关键数字，是否都能在仓库产物里找到出处。

与 `audit_paper_numbers.py` 的分工
----------------------------------
- `audit_paper_numbers.py` 审 **预印本正文**（`docs/paper/manuscript.md`）的
  8 张表与统计断言，逐单元格比对；
- 本脚本审 **毕设素材文档**（仓库外的底账 / 叙事 / 审计方案），
  确认「写进正文要用的关键数字」都能追溯到 `data/**/*.json`。

为什么需要它
------------
毕设正文由人撰写，素材来自底账。若底账里的数字与产物脱节，人会照着错数字写。
本脚本把「底账 → 产物」的可追溯性变成一条可执行检查。

局限（必须如实说明）
--------------------
- 检查清单 `KEY` 是**人工维护**的，不是自动抽取全文数字。
  新增实验后需手动往 `KEY` 里加条目，否则不会被覆盖。
- 只做「值是否存在」的容差比对，**不校验语义对应**
  （例如 0.7421 同时是 LR AUC 和别的什么数时，脚本无法分辨）。
- 产物被某个档位覆盖后，旧档位的数字会变成 MISS——这正是本脚本要暴露的问题。

用法
----
    python scripts/audit_thesis_numbers.py
    python scripts/audit_thesis_numbers.py --strict    # 有缺失则退出码 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 关键数字清单：(文档中的写法, 语义说明)
# 覆盖：端到端主结果 / 消融（full 档）/ 模型对照（full 档）/ Coat / 性能与稳定性
KEY: list[tuple[str, str]] = [
    # --- 端到端主结果（data/eval/end_to_end.json）---
    ("0.7421", "LR AUC"),
    ("0.8019", "DeepFM AUC"),
    ("0.6321", "DeepFM+混排 真实 CTR"),
    ("0.7155", "Oracle 上界"),
    ("0.4922", "热门基线 真实 CTR"),
    ("0.1652", "随机基线"),
    ("0.2844", "相对热门提升 +28.4%"),
    ("0.8835", "占 Oracle 88.3%"),
    ("0.6231", "热门榜 平滑 CTR（虚高）"),
    ("0.2521", "同榜 真实 CTR"),
    # --- 端到端 NDCG（end_to_end.json:true_ctr_ranking.ndcg_at_k）---
    # 真实 CTR 只看"点得多不多"，NDCG 才看"好的排前面没有"。
    # 两者占比之差（90.2% vs 88.4%）是"瓶颈在召回不在排序"的论据，第 5 章要用。
    ("0.9016", "NDCG@10 DeepFM+混排"),
    ("0.2068", "NDCG@10 随机"),
    ("0.6716", "NDCG@10 热门"),
    # --- 消融 micro 档（data/eval/ablation_micro.json，100 用户）---
    # micro 档用于证明"结论不随数据规模翻转"。与 full 档同用户数（100），
    # 只变数据规模——用户数相同才能把差异归因到规模而不是样本量。
    ("0.5383", "micro 全四路（基线）"),
    ("0.5564", "micro 仅 popular"),
    ("0.6125", "micro 混排关同品类打散"),
    ("0.2058", "micro 混排关 CTR 加权"),
    ("0.2210", "micro 混排关归一化"),
    # --- 消融 full 档（data/eval/ablation_full.json）---
    ("0.4987", "全四路（基线）"),
    ("0.5115", "仅 popular"),
    ("0.3847", "去掉 popular"),
    ("0.3313", "仅 itemcf"),
    ("0.5006", "去掉 itemcf"),
    ("0.4085", "仅 two_tower"),
    ("0.5069", "去掉 two_tower"),
    ("0.3601", "仅 rule"),
    ("0.5214", "去掉 rule"),
    ("0.6809", "召回消融 Oracle"),
    ("0.5632", "混排：关同品类打散"),
    ("0.2145", "混排：关 CTR 加权"),
    ("0.2128", "混排：关归一化"),
    # --- 模型对照 full 档（data/eval/model_comparison_full.json）---
    ("0.8131", "LR + 手工交叉"),
    ("0.8124", "DeepFM + 手工交叉"),
    # --- Coat（data/bias/boundary_coat.json）---
    ("0.3177", "Coat ESS/N 口径A（记录级）"),
    ("0.4890", "Coat ESS/N 口径B（逐用户）"),
    ("0.5371", "Coat ESS/N 口径C（候选池）"),
    ("0.0884", "Coat naive 点击率误差"),
    ("0.0263", "Coat SNIPS 点击率误差"),
    ("0.0620", "Coat 误差下降 gain"),
    ("0.0810", "Coat A 组去偏收益"),
    ("-0.0032", "Coat B 组（对照）收益"),
    # --- 性能：串行 micro（data/bench.json）---
    ("1.448", "micro P50 延迟 ms"),
    ("1.845", "micro 平均延迟 ms"),
    ("541.9", "micro 单线程 QPS"),
    ("128.1", "micro 峰值 RSS MB"),
    ("0.201", "micro 引擎冷启动 s"),
    # --- 性能：lite 训练形态（data/bench_lite.json）---
    ("293.846", "lite 引擎冷启动 s"),
    ("196.673", "lite P50 延迟 ms"),
    ("926.7", "lite 峰值 RSS MB"),
    # --- 并发（data/bench_concurrency.json）---
    ("565.7", "1 线程聚合 QPS"),
    ("380.6", "16 线程聚合 QPS"),
]


def collect_numbers() -> list[tuple[float, str]]:
    """读出 data/**/*.json 里所有数值及其路径。"""
    out: list[tuple[float, str]] = []
    for f in sorted(ROOT.glob("data/**/*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        def walk(o, path: str) -> None:
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, f"{path}[{i}]")
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                out.append((float(o), f"{f.relative_to(ROOT)}:{path}"))

        walk(d, "")
    return out


def find(target: str, nums: list[tuple[float, str]]) -> list[str]:
    """按文档写法的小数位数取容差，避免「1.4476 vs 1.448」这类精度误报。

    容差放宽到「末位 ±1 个单位」（4 位小数 → ±0.0005），原因见
    `data/eval/end_to_end.json` 的 `meta.note`：真实 CTR 口径的 Oracle 池含
    `rng.choice` 随机采样，**同一台机器重复运行**末位也会漂（三次实测
    0.6323 / 0.6321 / 0.6322）。若按 ±0.00005 严格比对，重跑一次门禁就红，
    逼人去改文档数字——那是在迁就噪声，不是在保证正确。
    放宽后仍能抓住真正的手误（如 0.4987 写成 0.4897，差 0.009 > 0.0005）。
    AUC 类指标不受此影响：`DeepFMRanker.__init__` 已播种，逐位可复现。
    """
    dec = len(target.split(".")[-1]) if "." in target else 0
    tol = 0.5 * 10 ** (-(dec - 1)) if dec >= 1 else 0.5
    val = float(target)
    return [src for v, src in nums if abs(v - val) <= tol]


def main() -> int:
    ap = argparse.ArgumentParser(description="审计毕设素材数字的产物支撑")
    ap.add_argument("--strict", action="store_true", help="有缺失则以退出码 1 结束")
    args = ap.parse_args()

    nums = collect_numbers()
    print(f"产物中共采集到 {len(nums)} 个数值\n")

    missing: list[tuple[str, str]] = []
    for target, desc in KEY:
        hit = find(target, nums)
        if hit:
            print(f"  OK   {target:>9}  {desc:<26} ← {hit[0]}")
        else:
            missing.append((target, desc))
            print(f"  MISS {target:>9}  {desc:<26} ✗ 无产物支撑")

    print()
    if missing:
        print(f"⚠️  {len(missing)} 项无产物支撑：{[m[0] for m in missing]}")
        print("   可能原因：① 产物被其他档位覆盖（检查是否有 *_micro.json / *_full.json）；"
              "② 文档数字写错；③ 实验未重跑。")
        if args.strict:
            return 1
    else:
        print(f"✅ {len(KEY)} 项关键数字全部有产物支撑")
    return 0


if __name__ == "__main__":
    sys.exit(main())
