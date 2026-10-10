"""偏差诊断模块（DeviationAuditor）。

本模块把推荐链路上的**偏差**形式化为可注入、可测量、可去偏的实验变量。

核心资产
--------
与真实日志最根本的差别在于：本项目的数据是**合成**的，因此
「每个 (用户, 物品) 对的真实 CTR」是**解析已知的**（`compute_true_ctr`，见
`src/data/generator_common.py`）。真实数据集只能*估计*偏差，这里可以*精确测量*。
这是「真实数据做不到、而合成数据做得到」的那件事，也是本模块的立足点。

三个层次
--------
1. `policy`     —— 曝光策略：给定打分，如何决定谁被曝光；**propensity 闭式可得**。
2. `metrics`    —— 偏差度量：把某个估计量相对 ground-truth 的偏离量化。
3. `estimators` —— 去偏估计量：IPS / SNIPS / DR，检验能否把有偏观测还原到真值。
"""
from __future__ import annotations

__all__ = ["policy", "metrics", "estimators", "propagation", "boundary", "report"]
