"""排序层：DeepFM CTR 模型 + 5 步混排 + 降级兜底。

注意：本文件使用**惰性导入**（PEP 562 的模块级 `__getattr__`）。

为什么要惰性
------------
在线 Demo 运行在内存受限的免费实例上（Streamlit Community Cloud，1GB）。
实测框架固有导入开销：torch 643MB、sklearn 176MB、pandas 约 150MB。
在线侧走的是纯 numpy 推理路径（`src/rank/numpy_deepfm.py`），**不需要 torch**。

若本文件在包初始化时就 `from src.rank.deepfm import DeepFM`，
则任何 `import src.rank.anything` 都会连带把 torch 拉进内存，
使在线侧白白付出 643MB——这正是本项目踩过的坑。

改为惰性后：
    from src.rank.mixer import Mixer          # 不触发 torch
    from src.rank.numpy_deepfm import ...     # 不触发 torch
    from src.rank import DeepFM               # 显式取用才触发 torch（离线侧照旧）
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅类型检查时导入，运行时不触发
    from src.rank.deepfm import DeepFM
    from src.rank.trainer import DeepFMRanker
    from src.rank.mixer import Mixer
    from src.rank.degrade import DegradeRanker, compute_recent_ctr

__all__ = ["DeepFM", "DeepFMRanker", "Mixer", "DegradeRanker", "compute_recent_ctr"]

# 名称 -> 所属子模块（惰性加载映射）
_LAZY = {
    "DeepFM": "src.rank.deepfm",
    "DeepFMRanker": "src.rank.trainer",
    "Mixer": "src.rank.mixer",
    "DegradeRanker": "src.rank.degrade",
    "compute_recent_ctr": "src.rank.degrade",
}


def __getattr__(name: str):
    """按需加载：只有真正访问 `src.rank.DeepFM` 时才导入对应子模块。"""
    mod_path = _LAZY.get(name)
    if mod_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(mod_path)
    value = getattr(mod, name)
    globals()[name] = value  # 缓存，后续访问不再走 __getattr__
    return value


def __dir__():
    return sorted(__all__)
