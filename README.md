# AeroRec · 出行智能推荐中台

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/cloud)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 面向出行消费场景（机票 / 酒店 / 商城 / 营销 / 直播）的开源智能推荐中台 Demo。
> 工程化复现工业级推荐链路：**四路召回 → DeepFM CTR 排序 → 5 步混排 → 降级兜底 → 真实 CTR 评估**。

**在线体验**：`（部署后替换为 https://<你的 App URL>.streamlit.app ）`

---

## 一、这个项目在做什么

把一条**完整可跑的推荐链路**从零搭出来，并把每一层的**工程判断**显式写出来：

```
用户请求
  └─ 召回层：热门兜底 ∪ 规则召回（里程兑换 / 常驻地 / 多程返程）
       └─ 排序层：DeepFM(CTR) + LR(CVR) 多目标加权打分
            └─ 混排层：归一化 → 加权 → 分层 → 同品类打散 → 去重截断
                 └─ 降级层：CTR 预估异常 → 回退历史平均 CTR
                      └─ 评估层：真实 CTR 口径 + Oracle 上界 / 模拟 A/B
```

**不做的事**：不提出新算法。DeepFM 是标准 DeepFM（Guo et al., 2017）。
本项目的价值在**工程实现与评估严谨性**，不在算法创新。

---

## 二、三档数据规模（同一套算法，不同资源约束）

| 档位 | 规模 | 峰值内存 | 用途 |
|---|---|---|---|
| `full` | 10 万用户 × 10 万物品 × 500 万曝光 | ~1.5 GB | 本地完整复现，**论文级可引用数值来源** |
| `lite` | 2 万 × 2 万 × 100 万曝光 | ~930 MB | 本地 / 2GB 内存实例（在线训练） |
| `micro` | 1 万 × 1 万 × 50 万曝光 | **~130 MB** | 1GB 内存的免费在线实例（离线训练 + numpy 推理） |

三档**共用同一套生成公式、随机种子与全部算法实现**，差异仅在规模与产物路径。

```bash
AEROREC_MODE=micro  streamlit run app.py   # 默认（1GB 免费档）
AEROREC_MODE=lite   streamlit run app.py   # 轻量档
AEROREC_MODE=full   streamlit run app.py   # 完整档（需大内存）
```

---

## 三、核心结果（完整档，**可引用口径**）

评估用**已知的潜在真实 CTR** 计算各方法 Top-K 的平均点击率，不依赖任何
「平滑历史 CTR」估计，因此**不受稀疏性偏差影响**。

完整档实测（100 用户 / Top-10）：

| 方法 | 平均真实 CTR | 相对热门 | 占 Oracle 上界 |
|---|---|---|---|
| 随机 | 0.1652 | -66.4% | 23.4% |
| 热门（基线） | 0.4922 | — | 68.4% |
| **DeepFM + 多目标混排** | **0.6546** | **+33.0%** | **91.0%** |
| Oracle 上界 | 0.7194 | +46.2% | 100% |

> **Oracle 上界的构造**（关键）：逐用户候选池 = 热门 ∪ 规则 ∪ 随机采样，
> 在池内按该用户的真实 CTR 降序取 Top-K。池**必须包含被比较方法的结局**，
> 否则 Oracle 不构成真正的上界。这样构造可保证 Oracle 真实 CTR
> **严格 ≥ 任何从同一候选空间取 Top-K 的方法**。

---

## 四、一个必须公开的坑：为什么有两套评估口径

### ① 真实 CTR 对比（✅ 可引用）

见上一节。

### ② 模拟在线 A/B（⚠️ 仅供形式对照，数字勿引用）

热门基线按「历史日志的**贝叶斯平滑 CTR**（α=5）」排序，而本项目日志是
**随机曝光**生成的：曝光次数少的物品一旦偶然被点击，平滑 CTR 就被显著高估
（α=5 不足以抵消这种噪声）。这属于推荐系统中的**位置偏差 / 数据稀疏偏差**。

完整档实测：

```
热门榜 top200 平滑 CTR = 0.6231
热门榜 top200 真实 CTR = 0.2528      → 偏差 +0.37
平滑榜 top10 与真实 CTR top10 重合数 = 2/10
```

**后果**：用「热门基线」作对照会**高估**推荐系统的相对收益
（模拟口径 +18.9% vs 真实 CTR 口径 +7.1% —— 注意此处数值随数据规模变化）。

**处置**：本项目没有把这个偏差藏起来，而是——

1. 把它做成页面上的显式提示；
2. 单独拆出「真实 CTR 对比」页作为可引用口径；
3. 把 `Mixer` 的 `w_base` 默认值从 `0.4` 改为 `0.0`，避免候选池的热度信息在混排阶段**被二次注入**；
4. 在 `src/eval/ab.py` 的模块文档里写清来龙去脉。

> 先知道自己的指标哪里不可信，再谈提升。这本身就是本项目想展示的判断力。

---

## 五、在线 Demo 的一个工程取舍：把训练与推理解耦

在线目标平台（Streamlit Community Cloud 免费档）内存上限 **1GB**。
实测各依赖的**固有导入开销**（与数据规模无关）：

| 组件 | 累计内存 |
|---|---|
| 裸 Python | 9 MB |
| + numpy | 25 MB |
| + scikit-learn | 176 MB |
| + **torch** | **643 MB** |
| + streamlit | 676 MB |

**即：只要 `import torch` 就固定花掉 643MB。** 实测把数据从 100 万曝光压到
50 万曝光，峰值内存只从 847MB 降到 840MB —— **瓶颈是框架运行时，不是数据**。

于是把**训练与推理解耦**：

1. **离线（本机）** 用 torch 训练 DeepFM(CTR) 与 LR(CVR)；
2. 权重**导出为 numpy 数组**（`python -m src.build_online_artifacts micro`）；
3. **在线侧只加载权重**，用纯 numpy 复现前向，不依赖 torch / sklearn。

| | 解耦前（在线训练） | 解耦后（加载 numpy 权重） |
|---|---|---|
| 峰值内存 | 840 MB | **~130 MB** |
| 启动耗时 | 90 s | **~1 s** |
| 相对 1GB 上限余量 | 1.2×（有 OOM 风险） | **~8×** |

**数学等价性已逐样本校验**：

```
PyTorch vs numpy 前向最大偏差 = 1.19e-07
```

即：在线实例演示的**仍是同一个已训练模型**，只是换了一种运行时执行。
完整档离线评估仍使用 PyTorch 原版，权重与结论不受影响。

---

## 六、目录结构

```
app.py                        # Streamlit 入口（四个页面）
requirements.txt              # 完整依赖（含 torch / sklearn，本地 full/lite 档用）
requirements_streamlit.txt    # 在线依赖（仅 numpy + streamlit，省 ~800MB）
artifacts/micro/              # 离线预训练权重（numpy 格式）
  deepfm_ctr.npz              #   DeepFM 权重
  cvr_lr.npz                  #   LR(CVR) 系数 + OneHot 类别边界
  meta.json                   #   规模 / 种子 / 特征维度元信息
src/
  config.py                   # 基础定义（类目 / 城市 / 时段 / 生成系数）
  config_scale.py             # 档位解析（AEROREC_MODE → micro / lite）
  config_micro.py             # micro 档配置
  config_lite.py              # lite 档配置
  build_online_artifacts.py   # 离线训练 + 导出权重（需 torch）
  data/
    generator_common.py       # pandas-free 的生成基础量与真实 CTR/CVR 公式
    synthetic_generator.py    # full 档生成器（pandas）
    synthetic_generator_lite.py # lite/micro 档生成器（numpy）
  feature/lite_store.py       # numpy 特征库（dense / sparse 矩阵构造）
  rank/
    __init__.py               # 惰性导入（PEP 562）——避免 torch 被连带拉起
    deepfm.py                 # DeepFM 网络定义（PyTorch）
    trainer.py                # DeepFMRanker 训练封装
    numpy_deepfm.py           # DeepFM 的纯 numpy 前向（与 PyTorch 数值等价）
    numpy_lr.py               # LR + OneHot 的纯 numpy 实现
    mixer.py                  # 5 步混排
    degrade.py                # 降级兜底
  eval/
    metrics.py                # AUC / Recall@k / NDCG@k（sklearn 缺失时纯 numpy 兜底）
    ab.py                     # 真实 CTR 对比 + Oracle 上界 + 模拟 A/B
    evaluate.py               # 完整档离线评估入口
  serve/
    engine_lite.py            # lite 档引擎（在线训练）
    engine_micro.py           # micro 档引擎（纯 numpy 推理，零 torch）
```

---

## 七、快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成离线推理权重（仅 micro 档需要，约 2 分钟，需 torch）
python -m src.build_online_artifacts micro

# 3. 启动
streamlit run app.py
```

数据由脚本**运行时合成**，不入库。首次打开约 1 秒完成数据生成与权重加载。

---

## 八、四个页面

| 页面 | 内容 | 口径 |
|---|---|---|
| 推荐演示 | 选用户 → 出 Top-K，每项带预估 CTR / CVR 与基础分 | — |
| 排序流程 | 展示 5 步混排的中间态 | — |
| **真实 CTR 对比** | 各方法 Top-K 平均真实 CTR + 逐用户候选池 Oracle 上界 | ✅ **可引用** |
| 模拟 A/B | 对照（热门）vs 实验（DeepFM+混排），含 z 检验 | ⚠️ 仅供参考 |

---

## 九、诚实边界

- 数据为本项目**合成**（受控实验），不使用、也不声称使用任何真实业务数据。
- 生成公式中注入了隐式非线性交叉项（价格敏感 × 低价、商务 × 早班机、里程契合），
  模型只能拿到原始特征，交叉需自行学习 —— 此为受控实验设置。
- 模型均使用**已有方法的实现**（DeepFM 为标准 DeepFM，Guo et al., 2017），
  本项目**不提出新算法**。
- 在线实例为 micro 档（1万×1万×50万），**这里的一切数值都只是体验演示**，不作为结论。
  规模越小、数据越稀疏，相对增益越低 —— 这是预期现象，不是模型变差。
- 实习经历仅作为项目动机来源，实习中未涉及深度排序模型的训练。
- 评估口径的局限性已在代码与页面中显式标注（见 `src/eval/ab.py` 模块文档）。

---

## 十、License

MIT
