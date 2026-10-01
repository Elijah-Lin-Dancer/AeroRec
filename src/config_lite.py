"""轻量档（lite）配置：供 2GB 内存的在线 Demo 实例使用（如 Hugging Face Spaces）。

设计原则
--------
**只改"体量"，不改"方法"。** 全链路的每一环都完整保留：
  热门/规则召回 → 特征构建 → DeepFM(CTR) + LR(CVR) 多目标排序
  → 5 步混排 → 降级兜底 → 模拟 A/B。

与完整档（`src/config.py`，10 万用户 × 10 万物品 × 500 万曝光）相比：

  完整档（离线 / 可引用口径）        轻量档（在线 Demo 口径）
  N_USERS  100,000              →    20,000
  N_ITEMS  100,000              →    20,000
  曝光     5,000,000            →    1,000,000

为什么需要单独一档
------------------
1. pandas 3.0 + pyarrow 后端下，构造百万行的表本身就要约 170MB 额外开销；
2. 导入 PyTorch 约需 425MB；
两者叠加会超出 1GB 免费实例。轻量档改用纯 numpy 通路并把规模控制在
2 万 × 2 万 × 100 万，整机峰值约 900MB，可稳定跑在 2GB 实例上。

诚实边界（重要）
----------------
两档使用**同一套生成公式与随机种子**，CTR/CVR 分布与隐式非线性交叉结构一致；
但规模缩小会放大热门基线的**稀疏性偏差**（见 `src/serve/engine_lite.py` 说明），
因此**轻量档的 A/B 数值不可引用**。申请材料中的量化结论一律以完整档为准。
"""
from pathlib import Path

from src.config import *  # noqa: F401,F403  继承类目/城市/时段/生成系数等全部定义

# ---- 覆盖：数据规模（核心改动点）----
N_USERS = 20_000
N_ITEMS = 20_000
ITEMS_PER_CAT = N_ITEMS // 5      # 每类目 4 千
IMPRESSIONS_PER_USER = 50         # 2万 × 50 = 100 万曝光日志

# ---- 覆盖：生成产物路径（与完整档隔离，避免互相覆盖）----
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
PROC_DIR = DATA_DIR / "processed_lite"
RAW_DIR = DATA_DIR / "raw_lite"
ITEMS_PATH = PROC_DIR / "items.csv"
USERS_PATH = PROC_DIR / "users.csv"
TRIPS_PATH = PROC_DIR / "trips.csv"
LOGS_PATH = PROC_DIR / "impression_logs.csv"

LITE = True
