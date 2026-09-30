"""全局配置：路径、随机种子、数据规模、类目、CTR/CVR 生成参数。"""
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "src"
DATA_DIR = BASE / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

# 可复现性
SEED = 42

# 数据规模（实测环境：32 核 / 123GB 内存 / 256GB 磁盘，CPU 版 torch）
N_USERS = 100_000
N_ITEMS = 100_000                 # 10 万物品
ITEMS_PER_CAT = N_ITEMS // 5     # 每类目 2 万
IMPRESSIONS_PER_USER = 50        # 每个用户 50 次曝光 → 500 万日志
MAX_TRIPS_PER_USER = 4           # 未来行程数上限（出行约束用）

# 出行消费五大推荐类目
CATEGORIES = ["航线", "酒店", "商城", "营销", "直播"]

# 城市（出发/到达/常驻）
CITIES = ["北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", "西安", "重庆", "南京"]
# 航班时段（出行特有：商务×早班机的隐式交叉）
TIME_SLOTS = ["早班", "午间", "晚班"]
# 曝光上下文
HOUR_SLOTS = ["高峰", "平峰"]

# 服务参数
TOP_K = 10

# ---- CTR 生成（含"隐式"非线性交叉项，模型不可直接拿到乘积）----
# true_ctr = sigmoid( CTR_BASE + aff[cat] + q_w*quality + cat_bias[cat]
#     + g1*(price_sensitivity * (1-price_norm))   # 价格敏感 × 低价
#     + g2*(is_business * is_early_morning)        # 商务 × 早班机
#     + g3*miles_fit )                             # 里程预算契合
# CTR_BASE 为负偏置，把整体点击率校准到 ~12% 的现实区间
CTR_BASE = -2.0
QUALITY_W = 0.6
GAMMA_PRICE = 0.8          # g1
GAMMA_BIZ_MORNING = 0.7    # g2
GAMMA_MILES = 0.5          # g3
CAT_BIAS_STD = 0.3

# ---- CVR 生成（点击后转化）----
# cvr = sigmoid( beta0 + beta1*miles_fit + beta2*(1-price_norm) + noise )
BETA0 = -2.5               # 压低基准转化率到 ~10%
BETA_MILES = 0.5
BETA_PRICE = 0.6
CVR_NOISE_STD = 0.3

# 特征库版本
FEATURE_VERSION = "v1"

# 生成产物路径（CSV，可复现，不入库）
ITEMS_PATH = PROC_DIR / "items.csv"
USERS_PATH = PROC_DIR / "users.csv"
TRIPS_PATH = PROC_DIR / "trips.csv"
LOGS_PATH = PROC_DIR / "impression_logs.csv"
