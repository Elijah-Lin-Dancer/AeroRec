"""轻量档（lite）配置：供在线 Demo（Hugging Face Spaces 免费档 2GB 内存）使用。

设计原则
--------
**只改"体量"，不改"方法"。** 全链路的每一环都完整保留：
  四路召回 → 融合 → DeepFM(CTR) + LR(CVR) 多目标排序 → 5 步混排 → 降级兜底 → 模拟 A/B。

与完整档（`src/config.py`，10 万用户 × 10 万物品 × 500 万曝光）相比，
本档把规模压到 2 万 × 2 万 × 100 万，以适配 2GB 内存的免费实例
（实测峰值 RSS 约 930MB）：

  完整档（离线 / 论文口径）           轻量档（在线 Demo 口径）
  N_USERS  100,000              →     20,000
  N_ITEMS  100,000              →     20,000
  曝光     5,000,000            →     1,000,000

诚实边界：两档都使用**同一套生成公式与随机种子**，因此 CTR/CVR 的真实分布、
隐式非线性交叉结构、以及"DeepFM 优于 LR""多目标排序优于单目标"的结论方向一致；
但**具体数值会因规模不同而略有差异**（完整档真实 CTR 口径 +33.0% vs 热门，
轻量档为 +3.1%），README 与展示页给出的量化结果均为完整档（10万×10万×500万）的实测值，
在线 Demo 仅用于交互体验，不用于复现论文数值。
"""
from pathlib import Path

from src.config import *  # noqa: F401,F403  继承类目 / 城市 / 时段 / 生成系数等全部定义

# ---- 覆盖：数据规模（唯一改动点）----
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
