"""在线服务的去偏判据接入（`ess_warning_from_log` 的可执行化）。

为什么需要这一层
----------------
`src/bias/boundary.py:ess_warning_from_log` 已经能把倾向分翻译成一句建议，
但它要求调用方**先有倾向分**。真实系统里没有真值倾向分——那是合成数据的特权。
因此本模块补上中间那一步：**从日志估计倾向分**，并把判据接进在线链路。

倾向分估计口径（与 `src/bias/pi_estimation.py` 一致）
--------------------------------------------------
    π̂_i = (impr_i + λ) / (Σ_j impr_j + λ·C)
其中 `impr_i` 是物品 i 在日志中的**曝光次数**，`C` 是候选池大小，
`λ` 为 Laplace 平滑强度（λ=0 是原始频率估计，λ=1 把未见物品拉向均匀先验）。

`λ=0` 的坑必须讲清楚：偏斜大时冷门物品 `impr≈0` → `π̂≈0` → `1/π̂` 权重爆炸。
所以在线侧默认 `λ=1`。

⚠️ 口径提醒（Coat 实测已验证）
--------------------------------
ESS/N 对**计算口径**高度敏感：同一份 Coat 数据，四个口径给出
0.3177 / 0.4890 / 0.5371 / 0.93–0.95，横跨去偏有效边界 0.3579 的两侧。
本模块**必须**显式报告用的是哪个口径。本模块用**「口径 A（记录级）」**——
即先估物品粒度 π̂，再对**该批实际曝光记录**的倾向分整体计算 ESS/N，
与合成标定 `data/bias/boundary.json` 同口径。

实测反例（同一份偏斜日志）：
    候选池级口径 ESS/N = 0.9907  →  判定「可以放心去偏」
    记录级口径   ESS/N = 0.1129  →  判定「不建议去偏」
两个口径结论完全相反。候选池里大量从未曝光的物品经平滑后权重高度一致，
反而把 ESS/N 拉高，而它们根本不在日志里、不该参与判断。

诚实边界
--------
阈值 0.4（`ess_warning_from_log` 的默认阈值）标定于**合成**环境。
本模块不声称该数值在真实航旅系统上可直接使用——它声称的是
**机制可迁移**：判据的输入（ESS/N）与输出（该不该去偏）在真实系统上可算。
数值本身需在目标系统上重新标定。
"""
from __future__ import annotations

import numpy as np

from src.bias.boundary import ess_warning_from_log

__all__ = ["estimate_propensity_from_log", "bias_guard"]


def estimate_propensity_from_log(item_ids, n_candidates: int,
                                  lam: float = 1.0) -> np.ndarray:
    """从日志的曝光频次估计倾向分 π̂。

    参数
    ----
    item_ids : 一维序列，日志中被曝光的物品 id（**重复即多次曝光**）
    n_candidates : 候选池大小 C
    lam : Laplace 平滑强度（默认 1）

    返回
    ----
    长度 C 的 π̂ 数组，和为 1。

    说明
    ----
    这是**物品粒度**的估计：同一物品对所有用户共享一个曝光概率。
    它无法还原用户-物品粒度的差异（如「A 用户系统性地被推贵重航线」），
    那种差异需要 `src/bias/userbias.py` 那一套粒度更细的诊断。
    """
    c = int(n_candidates)
    if c <= 0:
        raise ValueError("n_candidates 必须为正")
    impr = np.zeros(c, dtype=np.float64)
    ids = np.asarray(item_ids, dtype=np.int64)
    if ids.size:
        ids = ids[(ids >= 0) & (ids < c)]
        np.add.at(impr, ids, 1.0)
    total = impr.sum()
    denom = total + lam * c
    if denom <= 0:
        return np.full(c, 1.0 / c)
    return (impr + lam) / denom


def bias_guard(item_ids, n_candidates: int, lam: float = 1.0) -> dict:
    """在线链路的去偏开关建议：只看日志，给出「该不该去偏」。

    这是 `ess_warning_from_log` 的**可执行封装**——补上了「从日志估倾向分」
    这一步，使调用方无需自己准备倾向分。

    ⚠️ 口径（这里最容易出错，必须显式说明）
    ---------------------------------------
    ESS/N 按 **Coat 实验的「口径 A（记录级）」** 计算：先估出**物品粒度**的 π̂，
    再把每条**实际曝光记录**的倾向分取出来组成序列，对**这批记录**整体算 ESS/N。

    为什么不直接对候选池全体物品算？实测反例（同一份偏斜日志）：
        候选池级口径 ESS/N = 0.9907  →  判定「可以放心去偏」
        记录级口径   ESS/N = 0.1129  →  判定「不建议去偏」
    **两个口径给出完全相反的结论。** 原因是候选池里包含大量从未曝光的物品，
    它们的 π̂ 经平滑后权重 1/π̂ 高度一致，反而把平均权重「摊平」，
    把 ESS/N 拉高——而这些物品根本不在日志里，不该参与判断。
    记录级口径才与合成标定（`data/bias/boundary.json`）同口径。

    返回
    ----
    {
      "ess": float,                 # 有效样本量
      "ess_ratio": float,           # ESS/N（记录级口径）
      "pi_min": float, "pi_max": float,
      "recommendation": str,        # 人可读建议（中文）
      "should_debias": bool,        # 是否建议启用 IPS/SNIPS
      "lam": float,                 # 实际使用的平滑强度
      "n_records": int,             # 日志曝光条数
      "n_candidates": int,
      "caliber": str,               # ESS/N 计算口径（必报）
      "threshold_note": str,        # 阈值的来源与迁移性说明
    }
    """
    ids = np.asarray(item_ids, dtype=np.int64)
    pi_item = estimate_propensity_from_log(ids, n_candidates, lam=lam)
    # 记录级口径：只取实际出现在日志里的记录的倾向分（重复曝光即重复记录）
    valid = ids[(ids >= 0) & (ids < int(n_candidates))]
    pi_records = pi_item[valid] if valid.size else pi_item
    warn = ess_warning_from_log(pi_records)
    # ess_warning_from_log 的阈值是 0.4（合成环境标定，见 boundary.py 模块说明）
    should_debias = bool(warn["ess_ratio"] > 0.4)
    out = {
        "ess": warn["ess"],
        "ess_ratio": warn["ess_ratio"],
        "pi_min": warn["pi_min"],
        "pi_max": warn["pi_max"],
        "recommendation": warn["recommendation"],
        "should_debias": should_debias,
        "lam": float(lam),
        "n_records": int(valid.size),
        "n_candidates": int(n_candidates),
        "caliber": ("记录级 ESS/N（口径 A）：先估物品粒度 π̂，"
                    "再对**该批实际曝光记录**的倾向分整体计算；"
                    "与合成标定 data/bias/boundary.json 同口径"),
        "threshold_note": ("阈值 0.4 标定于合成环境；本模块保证的是机制可迁移"
                           "（输入可算、输出可执行），数值需在目标系统重新标定。"
                           "Coat 实测显示 ESS/N 随口径横跨边界两侧，故必须连同口径一起报告。"),
    }
    return out
