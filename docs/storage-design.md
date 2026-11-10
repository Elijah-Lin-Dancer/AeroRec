# 存储设计

> 本文档对应毕设第 6 章「系统设计与实现」的存储层小节。
> 本项目是**单机可复现的研究原型**，存储设计的目标不是「支撑百万 QPS」，
> 而是**让每一条实验结论都能追溯到具体的数据形态与文件**。
> 这是研究型系统与生产系统的取舍差异，需在论文中明确声明。

## 1. 存储形态选择：文件式而非数据库

**选择：CSV（离线全量）+ NPZ（在线权重）+ JSON（实验结果），不引入数据库。**

理由（按重要性排序）：

| 理由 | 说明 |
|---|---|
| **可复现性优先** | CSV 是纯文本，`git diff` 可读、`sha256` 可校验；数据库导出物是二进制，跨版本不可比对 |
| **数据是派生物，不是资产** | 全量数据由 `src/data/synthetic_generator.py` 以固定种子生成，**不需要持久化事务保障**。删了能重算，这是合成数据相对真实业务数据的关键优势 |
| **零运维依赖** | 不引入 MySQL/PostgreSQL 意味着评审或答辩现场 `git clone` 后即可完整复现，不需要装数据库、导 SQL、配账号 |
| **规模在单机内存可承受范围** | 全量日志 500 万行 / 178 MB，pandas 读入约 3.5 s（实测），无需分库分表 |
| **许可约束** | Coat 原始数据为 CC BY-NC，与仓库 MIT 不兼容，只能用文件形式放在**仓库外**，靠环境变量指向 |

**代价（必须写进论文的局限）**：
- 无并发写、无事务、无索引，**不支持多用户同时写入**；
- 每次启动要全量读 CSV，冷启动成本随数据量线性增长；
- 不做 schema 演进管理，字段变更靠重新生成。

> 论文表述建议：**「本系统定位为可复现的研究原型，存储层刻意选择文件式以换取
> 可审计性，而非追求生产级的并发与事务能力」**。这比声称「我们设计了高可用存储」
> 更诚实，也更符合实际代码。

## 2. 数据分层与文件布局

```
data/
├── processed/              # L1 全量（full 档，10万×10万×500万曝光，178 MB）
│   ├── users.csv
│   ├── items.csv
│   ├── trips.csv
│   └── impression_logs.csv
├── processed_lite/         # L1 缩减档（2万×2万×100万，36 MB）—— 用于快速验证
├── raw/                    # 外部原始数据（Coat 等），**不入库**
├── bias/                   # 偏差实验产物（7 个 JSON）
│   ├── boundary.json          边界标定（阈值 0.3579）
│   ├── boundary_sweep.json    规模稳健性扫描
│   ├── propagation.json       偏差传播
│   ├── injection.json         二次注入
│   ├── estimand_check.json    两估计目标辨析
│   ├── propensity_estimation.json  π̂ 误差敏感性
│   ├── userbias.json          用户侧偏差
│   └── boundary_coat.json     ★ Coat 真实倾向分验证（毕设新增）
├── eval/                   # 系统级实验产物
│   ├── end_to_end.json        端到端主结果（full 档）
│   ├── ablation_full.json     ★ 消融·论文口径（召回 9 组 + 混排 6 组，100 用户）
│   ├── ablation_micro.json    ★ 消融·micro 档（100 用户，仅变数据规模，用于结论稳健性对照）
│   ├── model_comparison_full.json   ★ 模型对照·论文口径（A/B/C/E + 召回族）
│   └── model_comparison_micro.json  ★ 模型对照·快速档
├── bench.json              # 串行推理基准（micro）
├── bench_lite.json         # ★ 训练形态对照基准（lite）
├── bench_concurrency.json  # ★ 并发压测
└── stability.json          # ★ 稳定性测试

artifacts/micro/            # 在线推理工件（入库，供零 torch 部署）
├── deepfm_ctr.npz          # DeepFM 权重（numpy 格式，非 torch state_dict）
├── cvr_lr.npz              # CVR 的 LR 系数 + OneHot 类别
└── meta.json               # 特征维度、训练配置、指标
```

> ★ 标记为 2026-11-10 毕设工作新增。

## 3. 四表结构（离线全量）

### 3.1 `users.csv`（用户表，10 万行）

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` | int | 主键，0-based 连续 |
| `aff_航线` / `aff_酒店` / `aff_商城` / `aff_营销` / `aff_直播` | float [0,1] | 五品类亲和度，**决定 `aff_cat` 交叉项** |
| `miles_balance` | int | 里程余额，与 `items.miles_required` 构成 `miles_fit` |
| `home_idx` | int | 常驻城市索引（用于规则召回的出发地匹配） |
| `price_sensitivity` | float [0,1] | 价格敏感度，**与 `price_norm` 构成乘积型交叉** |
| `schedule_flexibility` | float [0,1] | 时刻灵活度 |
| `is_business` | 0/1 | 商务旅客，**与 `is_early` 构成乘积型交叉** |
| `active_days_7d` | int | 近 7 日活跃天数（活跃度代理） |

### 3.2 `items.csv`（物品表，10 万行）

| 字段 | 类型 | 说明 |
|---|---|---|
| `item_id` | int | 主键 |
| `category_idx` / `category` | int / str | 品类索引与名称（5 类：航线/酒店/商城/营销/直播） |
| `quality` | float [0,1] | 质量分，真值 CTR 的线性项之一 |
| `cash_price` | float | 现金价，归一化后进 `price_norm` |
| `is_miles_ticket` | 0/1 | 是否里程票 |
| `miles_required` | int | 所需里程 |
| `origin_idx` / `dest_idx` | int | 出发/到达城市（规则召回用） |
| `time_slot` | int | 时刻槽（0 = 早班，用于 `is_early`） |
| `stock` | int | 库存。**注意：当前生成器为均匀曝光，不按库存过滤**，此字段存在但未形成约束（见 §5 局限） |
| `popularity` | float | 热度（热门召回与 `w_base` 使用） |
| `name` | str | 展示名 |

### 3.3 `trips.csv`（行程表，用户 × 行程）

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` | int | 外键 → users |
| `trip_id` | int | 行程主键 |
| `origin_idx` / `dest_idx` | int | 出发/到达城市 |
| `depart_offset` | int | 出发时间偏移（规则召回按时间排序） |

### 3.4 `impression_logs.csv`（曝光日志，500 万行，178 MB）

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` / `item_id` | int | 联合主键（一次曝光 = 一行） |
| `category_idx` / `category` | int / str | 冗余存储，避免分析时 join |
| `hour_slot` / `is_weekend` | int | 上下文特征 |
| `true_ctr` | float | **生成器真值 CTR**（合成数据特权，真实日志没有） |
| `clicked` | 0/1 | 伯努利采样自 `true_ctr` |
| `converted` | 0/1 | 转化标签（CVR 模型训练用） |

> 设计取舍：`category` 与 `category_idx` **冗余存储**在同一张日志表里。
> 冗余违反第三范式，但避免了每次分析都要 join `items`。
> **实测（500 万行日志 × 10 万行物品）**：
> 直接取冗余列 **0.7 ms**，`merge` 取同名列 **1814.7 ms**——**差 2779 倍**。
> 研究场景下**查询速度优先于范式纯洁**，且该冗余字段在生成时一次性写入、
> 不存在更新不一致的风险（物品品类不随时间变化）。

## 4. 在线存储：为什么把 torch 权重导成 numpy

`artifacts/micro/deepfm_ctr.npz` 不是 torch 的 `state_dict`，而是**拆成纯 numpy 数组**的
扁平结构（`linear_dense_w` / `sparse_embs__0` / `dnn_weights__0` …）。

| 维度 | torch `state_dict` | numpy `.npz`（本方案） |
|---|---|---|
| 依赖 | 必须装 torch（**磁盘实测 1722 MB**；进程 RSS 增量约 643 MB） | 只需 numpy |
| 进程 RSS | 约 900 MB（lite 档实测 926.7 MB） | **128.1 MB** |
| 冷启动 | 需反序列化 + 建图 | `np.load` 直接读数组 |
| P50 延迟 | 196.7 ms（lite 档实测） | **1.45 ms** |
| 可审计性 | pickle 二进制，不可读 | npz 可逐数组打印检查 |

**实测对照见 `data/bench.json` vs `data/bench_lite.json`**：
同一套模型，仅存储形态不同，冷启动差 **1462 倍**，延迟差 **136 倍**。

> 这是第 7 章「性能优化」一节最硬的证据：**优化点不在算法，在存储形态**。

## 5. 已知局限（必须写进论文）

1. **无事务与并发写**：文件式存储不支持多写入者。若要做在线学习（持续写日志），
   需改为追加式日志文件或引入 SQLite。
2. **`stock` 字段未生效**：`items.stock` 存在，但 `generate_logs` 用
   `rng.integers` 均匀采样，**不按库存过滤**。这意味着「库存约束导致 overlap 进一步失效」
   这一假设**当前未被实现**，不能声称已验证。若要验证，需先在生成器里补可行性过滤。
3. **Coat 原始数据不入库**：CC BY-NC 许可与仓库 MIT 不兼容，只能放仓库外，
   由 `COAT_DIR` 环境变量指向。这导致**他人复现 Coat 实验需自行下载**。
4. **全量数据每次重新生成**：不做增量更新，种子固定故结果可复现，
   但意味着**无法模拟数据漂移**。
5. **JSON 产物无 schema 版本号**：字段增删靠人工核对。若长期演进应加 `schema_version`。

## 6. 与「50% 工程实践」要求的对应

学校要求工程实践占比 ≥50%。本章的存储设计对应以下可验证的工程产出：

| 工程点 | 证据 |
|---|---|
| 存储形态选型与取舍论证 | 本文档 §1、§4（含实测对照） |
| 四表结构设计与字段职责 | §3（含冗余存储的性能论证） |
| 在线/离线双形态分离 | `artifacts/micro/*.npz` vs `data/processed/*.csv` |
| 性能优化有据可依 | 冷启动 1462× / 延迟 136× 的实测数字 |
| 局限的诚实披露 | §5 五条 |
