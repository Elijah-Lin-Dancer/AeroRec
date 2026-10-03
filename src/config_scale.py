"""档位解析器：根据环境变量 `AEROREC_MODE` 选择数据规模档。

为什么要单独一个模块
--------------------
本项目有三档数据规模，只为在不同内存约束下跑通**同一条链路**：

  full   10 万 × 10 万 × 500 万曝光   ~1.5GB   本地完整复现，**论文级可引用数值来源**
  lite    2 万 ×  2 万 × 100 万曝光   ~930MB   本地 / 2GB 内存在线实例（在线训练）
  micro   1 万 ×  1 万 ×  50 万曝光   ~130MB   1GB 内存的免费在线实例（离线训练 + numpy 推理）

三档**共用同一套算法实现**，差异仅在 `N_USERS / N_ITEMS / IMPRESSIONS_PER_USER`
与生成产物目录。为避免在每个模块里写 `if MODE == ...` 分支，
这里做一次集中解析，其余模块统一 `from src.config_scale import *`。

用法
----
    AEROREC_MODE=micro  streamlit run app.py     # 默认（1GB 免费档）
    AEROREC_MODE=lite   streamlit run app.py     # 2GB 档
    AEROREC_MODE=full   streamlit run app.py     # 本地完整复现（需大内存）

注意：`config.py`（full 档）不经过本解析器——它被 `src/config_lite.py` 继承，
作为"基础定义层"（类目、城市、时段、生成系数等）。本解析器只在
micro / lite 两个"在线档"之间做选择。
"""
from __future__ import annotations

import os

MODE = os.environ.get("AEROREC_MODE", "micro").lower()

if MODE == "lite":
    from src.config_lite import *  # noqa: F401,F403
elif MODE in ("micro", "online", "cloud"):
    from src.config_micro import *  # noqa: F401,F403
    MODE = "micro"
else:
    raise ValueError(
        f"未知的 AEROREC_MODE={MODE!r}；可选：'micro'（默认）/ 'lite'。"
        f"如需 full 档请直接运行离线脚本（python -m src.eval.evaluate 等）。"
    )
