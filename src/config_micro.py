"""极轻档（micro）配置：供内存受限的在线免费实例（Streamlit Community Cloud，1GB 内存）使用。

为什么会有这一档（工程取舍说明）
--------------------------------
本项目最早为在线 Demo 准备的是 `lite` 档（2 万 × 2 万 × 100 万曝光），
实测峰值 RSS 约 930MB。原计划托管在 Hugging Face Spaces 免费档（2GB），
余量充足。但 2025 年后 Hugging Face 调整政策：

  - Static Space 免费，但**跑不了 Python**（Streamlit 需要常驻进程）；
  - Gradio / Docker Space **需要付费**（PRO，$9/月）；
  - 且官方已**废弃 Streamlit SDK**，推荐改用 Docker SDK。

于是转向 Streamlit Community Cloud（官方免费托管），但其免费档内存上限为 **1GB**。
930MB 的 `lite` 档 + Streamlit 运行时（约 100–150MB）会紧贴甚至越过该上限，
存在 OOM 风险。与其赌运气，不如**再降一档规模、留出确定的安全余量**：

  完整档 full （论文口径）   10 万 × 10 万 × 500 万曝光   ~1.5GB
  轻量档 lite （本地/大内存） 2 万 ×  2 万 × 100 万曝光   ~930MB
  极轻档 micro（在线免费档） 1 万 ×  1 万 ×  50 万曝光   ~130MB

关键取舍：**只降"体量"，不降"方法"，更不动任何可引用结论。**
micro 档与 full/lite 共用同一套生成公式、随机种子与全部算法实现
（四路召回 → 融合 → DeepFM(CTR) + LR(CVR) 多目标排序 → 5 步混排
 → 降级兜底 → `compare_rankers` 真实 CTR 对比 + Oracle 上界），
因此链路完整性与方法正确性不受影响。

诚实边界
--------
在线 micro 档的**一切数值都只是交互演示**，不作为任何结论：
数据规模越小、越稀疏，相对增益越低（这是预期现象，不是模型变差）。
论文级可引用结论一律以完整档（10万×10万×500万）离线复现为准。
"""
from pathlib import Path

from src.config_lite import *  # noqa: F401,F403  继承 lite 档全部定义（含 full 档类目/城市/时段等）

# ---- 覆盖：数据规模（唯一改动点，相对 lite 再减半）----
N_USERS = 10_000
N_ITEMS = 10_000
ITEMS_PER_CAT = N_ITEMS // 5      # 每类目 2 千
IMPRESSIONS_PER_USER = 50         # 1万 × 50 = 50 万曝光日志

# ---- 覆盖：生成产物路径（与 full / lite 三档互相隔离，避免磁盘缓存串档）----
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
PROC_DIR = DATA_DIR / "processed_micro"
RAW_DIR = DATA_DIR / "raw_micro"
ITEMS_PATH = PROC_DIR / "items.csv"
USERS_PATH = PROC_DIR / "users.csv"
TRIPS_PATH = PROC_DIR / "trips.csv"
LOGS_PATH = PROC_DIR / "impression_logs.csv"

LITE = True
MICRO = True
