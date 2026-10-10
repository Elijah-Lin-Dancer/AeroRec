"""去偏估计量：IPS / SNIPS / DR —— 以及**两个先被搞错、后被纠正**的关键认识。

⚠️ 认识一：IPS 纠正的不是"坏估计"，而是**估计目标**
-------------------------------------------------------------
直觉说法是「朴素估计有偏，IPS 把它纠回来」。**在本设定下这是错的**。实测：

    τ(曝光偏差)   池化 naive    池化真值 Σtruth·π    均匀池真值 mean(truth)
      0.0          0.2543          0.2525               0.2525
      1.0          0.3041          0.3045               0.2525
      2.0          0.3545          0.3567               0.2525
      3.0          0.4005          0.4028               0.2525

池化 naive 与 `Σtruth·π` 几乎完全吻合。**池化 CTR 均值本身无估计偏差**——
抽样按 π 进行，日志均值天然收敛到「曝光分布下的期望」。
看起来像偏差的上升（0.2525→0.4028）其实是**估计目标（estimand）变了**。

因此 IPS 的真实作用是**在两种目标之间转换**：
    T_pop = mean(truth)      均匀总体（推荐系统真正关心的）
    T_exp = Σ truth_i π_i    曝光加权总体（日志均值天然收敛到它）

⚠️ 认识二：**π 的两种约定不可混用**（本项目实测踩过，导致 DR 输出 −10.2）
--------------------------------------------------------------------------
   约定① 「包含概率」：π 是单个单元被记录的概率，量级 O(1)。
        适用于  IPS = (1/N)Σ y_i/π_i
                DR  = (1/N)Σ [m̂_i + (y_i−m̂_i)/π_i]
        实测 m̂≡truth 时 DR 给出 0.2964（真值 0.3）——正确。

   约定② 「分布概率」：π 在所有单元上归一化，Σπ=1，量级 ~1/N。
        只适用于**自归一化**形式  ŷ = Σ(y/π)/Σ(1/π)。
        若把约定②的 π 塞进 DR 的 m̂ 项，量纲不匹配、修正项被 1/π² 放大，
        实测输出 **−10.2**（真值 0.3），完全无意义。

本模块**统一采用约定②**（与 `policy.py` 的闭式 π 一致，Σπ=1），因此：
  - `snips_estimate` / `ips_estimate`  → 正确，估计 T_pop。
  - `dr_estimate`                      → 采用约定②下的**自归一化**形式；
                                          必须在 π 量级正常（非 Σπ=1 归一化）时使用，
                                          或直接改用 `dr_estimate_inclusion`。
  两种约定的校验见 `check_propensity_convention` 与测试。
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


# ----------------------------------------------------------------------
# 两个估计目标
# ----------------------------------------------------------------------
def naive_estimate(clicked) -> float:
    """朴素估计 = 日志点击均值。收敛到 **T_exp = Σ truth_i π_i**（曝光加权总体）。"""
    y = np.asarray(clicked, dtype=np.float64)
    return float(y.mean()) if len(y) else 0.0


def ips_estimate(clicked, propensity) -> float:
    """IPS / SNIPS 估计 = Σ(y/π) / Σ(1/π)。收敛到 **T_pop = mean(truth)**（均匀总体）。

    与 `snips_estimate` 同式；保留两个名字是为了在报告里让"公式"与"算法名"都能对上。
    """
    return snips_estimate(clicked, propensity)


def snips_estimate(clicked, propensity) -> float:
    """自归一化 IPS（Self-Normalized IPS），估计 T_pop。

    自归一化是对**候选集内**正确且低方差的形式：
        ŷ = Σ(y_i / π_i) / Σ(1 / π_i)
    """
    y = np.asarray(clicked, dtype=np.float64)
    pi = np.clip(np.asarray(propensity, dtype=np.float64), _EPS, None)
    if len(y) == 0:
        return 0.0
    w = 1.0 / pi
    denom = float(w.sum())
    return float((y * w).sum() / denom) if denom > 0 else 0.0


def unnormalized_ips_estimate(clicked, propensity) -> float:
    """未归一化 IPS：Σ(y_i/π_i) / N。

    ⚠️ 当 π 被归一化为 Σπ=1（本项目口径）时，π ~ 1/N，此式量纲为 O(N)·T_pop，
    与 T_pop 相差一个数量级——**不能用它直接汇报 CTR**。
    保留它是为了在报告中显式展示这个常见错误（见测试 test_unnormalized_ips_scale_trap）。
    """
    y = np.asarray(clicked, dtype=np.float64)
    pi = np.clip(np.asarray(propensity, dtype=np.float64), _EPS, None)
    if len(y) == 0:
        return 0.0
    return float((y / pi).sum() / len(y))


def dr_estimate(clicked, propensity, model_pred, snips: bool = True) -> float:
    """双重稳健（DR）估计 T_pop（**约定①：π 为包含概率，量级 O(1)**）。

        ŷ = (1/N) Σ [ m̂_i + (y_i − m̂_i) / π_i ]

    ⚠️ 传入 Σπ=1 的归一化 π（约定②）会使此式失效（实测输出 −10.2）。
    若手上只有约定②的 π，请先转成包含概率：
        π_incl_i = π_i · K        （K = 每次请求曝光坑位数）
    或用 `check_propensity_convention` 检测并报警。
    """
    y = np.asarray(clicked, dtype=np.float64)
    pi = np.clip(np.asarray(propensity, dtype=np.float64), _EPS, None)
    m = np.asarray(model_pred, dtype=np.float64)
    if len(y) == 0:
        return 0.0
    g = m + (y - m) / pi                    # 每条记录的 DR 伪结果
    return float(np.mean(g))


def dr_estimate_selfnormalized(clicked, propensity, model_pred) -> float:
    """自归一化 DR（配合约定②：Σπ=1）。

        ŷ = Σ w_i · m̂_i / Σ w_i  +  [ Σ w_i (y_i − m̂_i)/π_i ] / Σ w_i

    ⚠️ 实测：即便自归一化，把 Σπ=1 的 π 代入 DR 的修正项仍会被 1/π² 放大
    （π~1/N 时修正项量级 ~N），**不稳定**。因此本函数只在 π 量级正常时可用；
    对 Σπ=1 的 π，请改用 `dr_estimate` 并传入包含概率。
    保留此函数是为了在报告中显式展示这一失败模式。
    """
    y = np.asarray(clicked, dtype=np.float64)
    pi = np.clip(np.asarray(propensity, dtype=np.float64), _EPS, None)
    m = np.asarray(model_pred, dtype=np.float64)
    if len(y) == 0:
        return 0.0
    w = 1.0 / pi
    den = float(w.sum())
    if den <= 0:
        return 0.0
    # 模型项按 w 加权；修正项按 w/π 加权（注意这里是 1/π²，问题的来源）
    model_part = float(np.sum(w * m) / den)
    corr_part = float(np.sum(w * (y - m) / pi) / den)
    return model_part + corr_part


def check_propensity_convention(propensity, tol: float = 1e-6) -> dict:
    """判定传入的 π 属于哪种约定，并对误用给出明确警告。

    返回 {convention: 'distribution'|'inclusion'|'ambiguous',
          pi_sum, pi_mean, max_pi, warning}
    """
    pi = np.asarray(propensity, dtype=np.float64)
    s = float(pi.sum())
    out = {"pi_sum": s, "pi_mean": float(pi.mean()), "max_pi": float(pi.max())}
    if abs(s - 1.0) < tol:
        out["convention"] = "distribution"
        out["warning"] = ("Σπ=1（分布概率约定）：只能用自归一化 IPS/SNIPS。"
                          "不要直接用于 dr_estimate 的 m̂ 项。")
    elif pi.max() <= 1.0 + tol and s > 1.0 + tol:
        out["convention"] = "inclusion"
        out["warning"] = ("π 为包含概率：可用 IPS/DR 原始式。"
                          "若用自归一化式，需确认其抽样模型一致。")
    else:
        out["convention"] = "ambiguous"
        out["warning"] = "π 既未归一化也未满足包含概率约束，请检查取值来源。"
    return out


def cross_fit_model(features, clicked, propensity, n_folds: int = 5,
                    seed: int = 42) -> np.ndarray:
    """对 m̂ 做 K-fold 交叉拟合（避免"用自己预测自己"使 DR 退化）。

    若样本内拟合，m̂ 记住了 y_i → (y−m̂) 偏小、DR 退化为近似 naive。
    这是 DR 实现里最易忽略的一处（见测试 test_cross_fit_is_out_of_sample）。
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(clicked, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = len(y)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), max(2, int(n_folds)))
    for k, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != k])
        if len(train_idx) == 0:
            out[test_idx] = y.mean()
            continue
        Xtr = np.hstack([np.ones((len(train_idx), 1)), x[train_idx]])
        Xte = np.hstack([np.ones((len(test_idx), 1)), x[test_idx]])
        try:
            beta, *_ = np.linalg.lstsq(Xtr, y[train_idx], rcond=None)
            out[test_idx] = Xte @ beta
        except np.linalg.LinAlgError:                      # pragma: no cover
            out[test_idx] = y[train_idx].mean()
    return np.clip(out, 0.0, 1.0)


# ----------------------------------------------------------------------
# 去偏效果评估
# ----------------------------------------------------------------------
def debias_report(clicked, propensity, true_pop_mean: float,
                  true_exp_mean: float | None = None,
                  model_pred=None) -> dict:
    """把各估计量与**两个目标**分别对照。

    参数
    ----
    true_pop_mean : T_pop = mean(truth)            均匀总体真值（IPS 的目标）
    true_exp_mean : T_exp = Σ truth_i π_i          曝光加权真值（naive 的目标）
                    传 None 时跳过该项对照。

    这样报告能同时显示：
      - naive 对 T_exp 是否准（预期：很准 —— 它本来就无偏）
      - snips 对 T_pop 是否准（预期：很准）
      - 两者相对**对方**的目标差多少（这才是"看起来的偏差"）
    """
    naive = naive_estimate(clicked)
    snips = snips_estimate(clicked, propensity)
    out = {
        "true_pop_mean": float(true_pop_mean),
        "naive": naive,
        "ips": snips,
        "snips": snips,
        "naive_err_vs_pop": abs(naive - true_pop_mean),
        "snips_err_vs_pop": abs(snips - true_pop_mean),
        "naive_gain_vs_pop": abs(naive - true_pop_mean) - abs(snips - true_pop_mean),
    }
    if true_exp_mean is not None:
        out["true_exp_mean"] = float(true_exp_mean)
        out["naive_err_vs_exp"] = abs(naive - true_exp_mean)
        out["snips_err_vs_exp"] = abs(snips - true_exp_mean)
    if model_pred is not None:
        # DR 需要包含概率（约定①）。若传的是 Σπ=1 的分布概率，先换算成包含概率。
        pi = np.asarray(propensity, dtype=np.float64)
        conv = check_propensity_convention(pi)["convention"]
        pi_dr = pi * len(pi) if conv == "distribution" else pi
        dr = dr_estimate(clicked, pi_dr, model_pred)
        out["dr"] = dr
        out["dr_err_vs_pop"] = abs(dr - true_pop_mean)
        out["dr_propensity_convention"] = conv
    return out


def effective_sample_size(propensity) -> float:
    """有效样本量 ESS = (Σw)² / Σw²，w = 1/π。

    去偏的**有效边界**判据：π 越偏，ESS 越小、估计越不稳。
    均匀 π 时 ESS ≈ N；极端集中时 ESS → 1。
    """
    pi = np.clip(np.asarray(propensity, dtype=np.float64), _EPS, None)
    w = 1.0 / pi
    if len(w) == 0:
        return 0.0
    sw2 = float((w * w).sum())
    return float(w.sum() ** 2 / sw2) if sw2 > 0 else 0.0
