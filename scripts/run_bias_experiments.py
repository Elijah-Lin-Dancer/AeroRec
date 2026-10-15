"""偏差传播实验：跑出「链路各级偏差」矩阵与二次注入曲线。

用法：
    python scripts/run_bias_experiments.py                    # 跑全部（默认参数）
    python scripts/run_bias_experiments.py --requests 300     # 缩小规模快速验证

产出：
    data/bias/propagation.json     传播矩阵（多 τ）
    data/bias/injection.json       混排二次注入扫描（多 w_base）
    data/bias/estimand_check.json  估计目标对照（naive→T_exp / SNIPS→T_pop）
    data/bias/boundary_sweep.json  去偏收益 τ×N 网格
    stdout                         人类可读表格

设计说明
--------
本实验**不需要训练任何模型**：它研究的是「估计量在偏差下的行为」，
用真实 CTR 作 ground-truth 即可，因此秒级完成、完全可复现。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.bias.boundary import boundary_by_range, boundary_sweep
from src.bias.estimand import estimand_check
from src.bias.propagation import mixing_second_injection, propagation_matrix


def _synth_true_ctr(n_items: int = 60, seed: int = 0):
    """构造一份有真实结构的「物品真实 CTR」（少数物品明显更好）。

    刻意让 CTR 有真实的分层：用对数正态，使"热门/冷门"差距自然出现——
    否则曝光偏差没有可扭曲的对象。
    """
    rng = np.random.default_rng(seed)
    base = rng.lognormal(mean=np.log(0.22), sigma=0.5, size=n_items)
    return np.clip(base, 0.01, 0.9)[None, :]


def main():
    ap = argparse.ArgumentParser(description="偏差传播与二次注入实验")
    ap.add_argument("--items", type=int, default=500)
    ap.add_argument("--requests", type=int, default=5000)
    ap.add_argument("--slots", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seeds", dest="n_seeds", type=int, default=30,
                    help="传播/注入实验的种子数（跨种子均值±std；boundary 恒用 max(20, requests//20)）")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.join("data", "bias"), exist_ok=True)
    true_ctr = _synth_true_ctr(args.items, seed=0)

    taus = [0.0, 0.5, 1.0, 2.0, 3.0]
    mat = propagation_matrix(true_ctr, n_items=args.items, taus=taus,
                             n_requests=args.requests, slots=args.slots,
                             alpha=args.alpha, w_base=0.0, seed=args.seed,
                             n_seeds=args.n_seeds)

    print("=" * 100)
    print(f"链路偏差传播矩阵（{args.items} 物品，{args.requests} 请求 × {args.slots} 坑位，"
          f"α={args.alpha}，{args.n_seeds} 种子均值±std）")
    print("水平误差 = 逐物品相对误差 rel_err；排序误差 = 与真值的秩相关 spearman")
    print("-" * 100)
    hdr = (f"{'τ(曝光偏差)':>11} | {'L1_rel':>7} {'L2_rel':>7} {'L3_rel':>7} {'排序级增量':>9} | "
           f"{'L1_sp':>7} {'L2_sp':>7} {'L3_sp':>7} | {'最少曝光':>8} {'曝光KL':>7} {'Gini':>6}")
    print(hdr)
    print("-" * 100)
    for r in mat["rows"]:
        print(f"{r['tau']:>11.1f} | {r['L1_rel_err']:>7.3f} {r['L2_rel_err']:>7.3f} "
              f"{r['L3_rel_err']:>7.3f} {r['L2_incremental_rel']:>9.3f} | "
              f"{r['L1_spearman']:>7.3f} {r['L2_spearman']:>7.3f} {r['L3_spearman']:>7.3f} | "
              f"{r['impr_min']:>8.1f} {r['exposure_kl']:>7.4f} {r['exposure_gini']:>6.3f}")
    print("=" * 100)

    w_bases = [0.0, 0.2, 0.4, 0.8, 1.0]
    inj = mixing_second_injection(true_ctr, n_items=args.items, tau=2.0,
                                  w_bases=w_bases, n_requests=args.requests,
                                  slots=args.slots, alpha=args.alpha,
                                  seed=args.seed, n_seeds=args.n_seeds)
    print()
    print("=" * 100)
    print(f"混排二次注入扫描（固定 τ=2.0，{args.n_seeds} 种子均值±std）—— w_base 是基础分（热度）权重")
    print("-" * 100)
    print(f"{'w_base':>8} | {'排序级rel':>9} {'混排级rel':>9} {'注入增量':>9} | "
          f"{'排序级sp':>9} {'混排级sp':>9} {'注入增量':>9}")
    print("-" * 100)
    for r in inj["rows"]:
        print(f"{r['w_base']:>8.1f} | {r['L2_rel_err']:>9.3f} {r['L3_rel_err']:>9.3f} "
              f"{r['injection_rel']:>+9.3f} | {r['L2_spearman']:>9.3f} {r['L3_spearman']:>9.3f} "
              f"{r['injection_spearman']:>+9.3f}")
    print("=" * 100)

    p1 = os.path.join("data", "bias", "propagation.json")
    p2 = os.path.join("data", "bias", "injection.json")
    p3 = os.path.join("data", "bias", "boundary.json")
    for path, payload in ((p1, mat), (p2, inj)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- 去偏有效边界 ----
    print()
    bnd = boundary_by_range(t_pop=float(true_ctr.mean()), n_records=20000,
                            n_seeds=max(20, args.requests // 20))
    print("=" * 100)
    print("去偏有效边界：**不是** τ 决定成败，而是 π 的动态范围（→ ESS/N）")
    print("-" * 100)
    print(f"{'π 范围':>18} {'极值比':>10} {'ESS/N':>8} {'naive_err':>10} {'snips_err':>10} {'gain':>9}")
    print("-" * 100)
    for r in bnd["rows"]:
        print(f"[{r['pi_min']:.0e}, {r['pi_max']:.1f}]{'':>4} {r['pi_ratio']:>10.0f} "
              f"{r['ess_ratio']:>8.4f} {r['naive_err']:>10.4f} {r['snips_err']:>10.4f} "
              f"{r['gain']:>+9.4f}")
    print("=" * 100)
    print(f"  → {bnd['conclusion']}")
    print("  → 落地：ESS/N 无需真值即可计算，可作为真实系统的去偏预警指标。")
    with open(p3, "w", encoding="utf-8") as f:
        json.dump(bnd, f, ensure_ascii=False, indent=2)

    # ---- 估计目标对照（报告 §4 历史口径补测：60 物品、400 请求）----
    print()
    truth60 = _synth_true_ctr(60, seed=0)
    ec = estimand_check(truth60, taus=taus, n_requests=400, slots=args.slots,
                        seed=args.seed, n_seeds=args.n_seeds)
    print("=" * 100)
    print("估计目标对照：naive → T_exp（曝光加权总体）vs SNIPS → T_pop（全体总体），"
          f"60 物品、400 请求 × {args.slots} 坑位，{args.n_seeds} 种子")
    print("-" * 100)
    print(f"{'τ':>5} | {'T_pop':>7} {'T_exp':>7} | {'naive':>7} {'snips':>7} | "
          f"{'naive→T_exp':>11} {'SNIPS→T_pop':>11} | {'ESS/N':>7}")
    print("-" * 100)
    for r in ec["rows"]:
        print(f"{r['tau']:>5.1f} | {r['T_pop']:>7.4f} {r['T_exp']:>7.4f} | "
              f"{r['naive']:>7.4f} {r['snips']:>7.4f} | "
              f"{r['naive_err_vs_texp']:>11.4f} {r['snips_err_vs_tpop']:>11.4f} | "
              f"{r['ess_ratio']:>7.4f}")
    print("=" * 100)
    p4 = os.path.join("data", "bias", "estimand_check.json")
    with open(p4, "w", encoding="utf-8") as f:
        json.dump(ec, f, ensure_ascii=False, indent=2)

    # ---- τ-sweep 网格（报告 §6 历史口径补测：60 物品、40 种子）----
    print()
    sw = boundary_sweep(truth60.ravel(), taus=taus, n_records_list=[1000, 2000, 5000, 10000],
                        n_seeds=40, seed=args.seed)
    print("=" * 100)
    print("去偏收益 τ-sweep（60 物品、40 种子平均）：**τ 越大去偏赚得越多**（估计目标分离的收益）")
    print("-" * 100)
    print(f"{'τ':>5} {'N':>7} | {'naive_err':>10} {'snips_err':>10} {'gain':>9} {'±std':>8} {'ESS/N':>7}")
    print("-" * 100)
    for r in sw["grid"]:
        print(f"{r['tau']:>5.1f} {r['n_records']:>7d} | {r['naive_err']:>10.4f} "
              f"{r['snips_err']:>10.4f} {r['gain']:>+9.4f} {r['gain_std']:>8.4f} "
              f"{r['ess_ratio']:>7.4f}")
    print("=" * 100)
    p5 = os.path.join("data", "bias", "boundary_sweep.json")
    with open(p5, "w", encoding="utf-8") as f:
        json.dump(sw, f, ensure_ascii=False, indent=2)

    # ---- 结论摘要（自动生成，避免手工写错）----
    first, last = mat["rows"][0], mat["rows"][-1]
    print()
    print("关键结论：")
    print(f"  1) 曝光偏差从 τ={first['tau']} 到 τ={last['tau']}，"
          f"曝光 Gini 由 {first['exposure_gini']:.3f} 升至 {last['exposure_gini']:.3f}，"
          f"KL 由 {first['exposure_kl']:.4f} 升至 {last['exposure_kl']:.4f}。")
    print(f"  2) 排序质量（spearman）由 {first['L2_spearman']:+.3f} 跌到 {last['L2_spearman']:+.3f}"
          f"{'（已反转）' if last['L2_spearman'] < 0 else ''}——偏差的主要危害是**把排序搞反**，"
          f"而非整体偏高。")
    print(f"  3) 最少曝光物品的曝光次数由 {first['impr_min']} 降到 {last['impr_min']}，"
          f"这些冷门物品被 α 平滑拽向先验 1.0，因此被系统性高估。")
    print(f"  4) 混排 w_base 由 0 升到 1 时，混排级 rel_err 由 "
          f"{inj['rows'][0]['L3_rel_err']:.3f} 降到 {inj['rows'][-1]['L3_rel_err']:.3f}、"
          f"spearman 由 {inj['rows'][0]['L3_spearman']:+.3f} 升到 {inj['rows'][-1]['L3_spearman']:+.3f}"
          f"（w_base=1 时混排级等于真值，误差严格为 0，可作为实现自检）。")
    print(f"\n结果已写入：{p1}、{p2}、{p3}、{p4} 与 {p5}    耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
