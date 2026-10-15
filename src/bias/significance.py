"""显著性检验：**纯 numpy** 的 bootstrap 置信区间与 p 值。

为什么不用 scipy
----------------
研究路径只依赖 numpy（`repro.yml` 的 bias-experiments job 在只装 numpy 的环境
里重跑全部实验，任何额外依赖都会让 CI 变红）。因此假设检验用 **bootstrap**
（百分位置信区间 + 重抽样 p 值）实现：

  - 不依赖任何参数分布假设（per-seed 指标的分布形状未知）；
  - 纯 numpy、固定 bootstrap 种子，**完全可复现**（零漂移门禁可逐字节校验）；
  - 论文可辩护：bootstrap CI / p 值是标准非参数方法（Efron & Tibshirani 1993）。

口径
----
输入是**逐种子**的指标值（n_seeds 次独立重复），检验目标是它们的均值：

  - `ci95`   : bootstrap 均值的 95% 百分位置信区间；
  - `p`      : 单侧 p 值（alternative="less" → H1: mean < 0；
               "greater" → H1: mean > 0；"two-sided" → H1: mean ≠ 0）；
  - p 值用 (计数 + 1) / (B + 1) 平滑，避免 p=0 的过度自信。
"""
from __future__ import annotations

import numpy as np

# 固定 bootstrap 种子：与主实验 seed 解耦但确定性，保证产物逐字节可复现。
BOOTSTRAP_SEED = 424242
N_BOOT = 10_000


def bootstrap_mean_test(values, alternative: str = "two-sided",
                        n_boot: int = N_BOOT,
                        seed: int = BOOTSTRAP_SEED) -> dict:
    """对逐种子指标值做「均值是否偏离 0」的 bootstrap 检验。

    参数
    ----
    values : 逐种子的指标值（长度 = n_seeds）。
    alternative : "less" / "greater" / "two-sided"。
    n_boot : 重抽样次数；seed : bootstrap 专用种子（固定 → 可复现）。

    返回
    ----
    {mean, ci95, p, n, method, alternative}，全部为 JSON 可序列化类型。
    样本数 < 2 时返回 n=0 的空结果（不虚构显著性）。
    """
    x = np.asarray(values, dtype=np.float64)
    n = int(x.size)
    if n < 2 or n_boot < 1:
        return {"n": n, "mean": round(float(x.mean()) if n else 0.0, 4),
                "ci95": None, "p": None, "n_boot": 0,
                "method": "bootstrap_percentile", "alternative": alternative}
    rng = np.random.default_rng(seed)
    # (n_boot, n) 的重抽样索引 → 每次重抽样的均值
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot_means = x[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    if alternative == "less":
        # H1: mean < 0 → p = P(bootstrap mean >= 0)
        p = (float((boot_means >= 0).sum()) + 1.0) / (n_boot + 1.0)
    elif alternative == "greater":
        p = (float((boot_means <= 0).sum()) + 1.0) / (n_boot + 1.0)
    else:
        obs = float(x.mean())
        # 双侧：以 |bootstrap mean − obs| 度量偏离（中心化重抽样）
        centered = boot_means - obs
        p = (float((np.abs(centered) >= abs(obs)).sum()) + 1.0) / (n_boot + 1.0)
    return {"n": n, "mean": round(float(x.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p": round(p, 4), "n_boot": int(n_boot),
            "method": "bootstrap_percentile", "alternative": alternative}


def paired_diff_test(a, b, alternative: str = "two-sided",
                     n_boot: int = N_BOOT, seed: int = BOOTSTRAP_SEED) -> dict:
    """**同种子配对**差值 d_i = a_i − b_i 的均值检验。

    前提：a、b 来自同一组种子（CRN 下同 seed 的两次配置共享随机流），
    配对差消除了种子间方差，检验功效高于独立两样本。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, "paired test requires same-shape seed series"
    return bootstrap_mean_test(a - b, alternative=alternative,
                               n_boot=n_boot, seed=seed)
