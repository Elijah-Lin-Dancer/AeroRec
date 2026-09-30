# AeroRec · 出行智能推荐中台

> 面向出行消费场景（机票 / 酒店 / 商城 / 营销 / 直播）的开源智能推荐中台。
> 工程化复现工业级推荐链路：**召回 → CTR 动态排序 → 降级兜底 → 服务化 → 可解释 → 模拟 A/B**。
> 本科 AI 出国申请作品 · 工程型 · 交叉学科（AI + 出行/交通）

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 背景与动机

出行类 App（机票、酒店、辅营）的核心增长杠杆之一是**个性化推荐中台**：在首页、会员中心、行程页等位置，
把"对的商品"在"对的坑位"推给"对的用户"。工业实践中，这与多品类混排、CTR 动态排序、服务降级、A/B 实验密不可分。

本项目**受工业实习中的推荐中台工程实践启发**，用一个**可复现的合成数据集**演示完整工程链路，
便于评审者一键运行、审阅与复现。**本项目不使用、也不声称使用任何真实业务数据。**

## 核心功能

- [x] 多场景推荐：航线 / 酒店 / 商城 / 营销 / 直播 五大类目混排
- [x] 热门兜底召回 + 物品协同过滤（ItemCF）召回基线
- [x] Two-Tower 向量召回（PyTorch 双塔 + 负采样）
- [x] 规则召回：里程兑换 / 里程票零库存 / 常驻地热门航线 / 多段行程优先返程
- [x] 四路召回融合（位置分加权 + 降级兜底）
- [x] 轻量特征库：集中产出召回 / 排序 / 冷启动共用特征（dense + sparse）
- [x] CTR 动态排序：归一化 → CTR 加权 → 分层排序 → 同品类打散 → 分页去重
- [x] DeepFM 排序模型（PyTorch，FM + DNN）
- [x] 降级兜底：CTR 服务超时 → 历史平均 CTR 兜底
- [x] 离线评估：LR vs DeepFM 的 AUC 对比 + 推荐列表真实 CTR 对比
- [ ] 服务化 API（FastAPI）与可解释面板（W4）
- [ ] 模拟 A/B 报告（W4）
- [ ] 架构图 / 复现文档 / demo 视频（W5–W6）

## 系统架构

```
Web/Dashboard ──HTTP/JSON──▶ Serving(FastAPI)
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
        Recall 召回          Rank 排序            Degrade 降级
        (热门/协同/双塔/规则)  (CTR 模型)          (7日均值兜底)
            └───────────────────┼───────────────────┘
                                ▼
               Data: 合成出行交互生成器(可复现)
```

## 快速开始

```bash
pip install -r requirements.txt
python -m src.pipeline_demo        # 数据 + 四路召回 + 融合 + 样例推荐
python -m src.eval.evaluate        # LR vs DeepFM AUC + 排序层真实 CTR 对比
python -m pytest tests/ -q         # 冒烟测试
```

## 数据与复现

数据由 `src/data/synthetic_generator.py` 生成（固定随机种子，可一键复现）：

- **物品**：5 类目共 100,000 个（每类目 2 万），各带质量分 `quality`、现金价、里程价、出发/到达城市、航班时段、库存。
- **用户**：100,000 个，对各品类有亲和度 `aff_*`，附里程余额、常驻城市、价格敏感度、商务标签。
- **未来行程**：约 14.8 万条，支撑"多段行程按日期优先返程"等出行业务规则。
- **曝光日志**：500 万条，每条按"潜在真实 CTR"抽样点击、点击后按 CVR 抽样转化。

因已知潜在真实 CTR，离线评估（AUC、线上增益）具备可比性。生成产物为 CSV，位于 `data/processed/`（已 gitignore，用脚本重生）。

## 方法

1. **召回（已完成）**：热门兜底 + ItemCF + Two-Tower 向量召回 + 出行规则召回，四路融合产出候选集。
2. **排序（已完成）**：复刻工业 5 步流程——分数归一化 → CTR 加权 → 分层排序 → 同品类打散 → 分页去重；CTR 模型为 DeepFM（PyTorch）。
3. **降级（已完成）**：CTR 服务超时 → 降级为历史平均 CTR 兜底，保证核心链路不挂。

## 离线评估结果（受控合成实验）

| 指标 | 基线 | DeepFM / 排序层 | 提升 |
|---|---|---|---|
| CTR 模型 AUC（测试集 6 万） | Logistic Regression 0.742 | **DeepFM 0.798** | +5.6pp |
| 推荐列表真实 CTR（Top10 均值） | 热门 0.205 / 随机 0.155 | **DeepFM+混排 0.583** | 相对热门 **+184%** |

> 合成数据含隐式非线性交叉（价格敏感×低价、商务×早班机、里程契合），DeepFM 的 FM/DNN 结构
> 能自动学到这些交叉，故 AUC 与真实 CTR 均显著优于线性基线——此为受控实验结论，非真实线上指标。

## 目录结构

```
src/
  config.py              全局配置
  data/synthetic_generator.py   合成数据生成器
  feature/store.py             轻量特征库
  recall/baselines.py          热门 + ItemCF 召回
  recall/two_tower.py         Two-Tower 向量召回
  recall/rule.py              出行规则召回（里程 / 多段行程）
  recall/fusion.py            多路召回融合
  rank/deepfm.py              DeepFM 排序模型
  rank/trainer.py             CTR 训练器
  rank/mixer.py               5 步混排
  rank/degrade.py             降级兜底
  eval/metrics.py             离线指标
  eval/evaluate.py            LR vs DeepFM AUC + 真实 CTR 对比
  serve/                       (W4) FastAPI + 面板
tests/test_data.py         数据 / 特征 / 召回 冒烟测试
tests/test_rank.py         排序层 / 混排 / 降级 冒烟测试
```

## 诚实边界

- 数据为本项目**合成**（受控实验），仅用于演示工程链路；不涉任何真实业务数据，也不与任何真实企业系统关联。
- 数据生成公式中注入了隐式非线性交叉项，用于让深度模型可学；模型只能拿到原始特征，交叉需自行学习——此设计明确标注为受控实验设置。
- CTR / 召回模型使用标准模型（scikit-learn / PyTorch 实现的 Two-Tower、DeepFM 等均为已有方法的实现），本项目**不提出新算法**。
- 实习经历仅作为项目动机来源（业务规则与流程启发），实习中未涉及深度排序模型的训练。

## 待办（Roadmap）

- [x] W2：Two-Tower 召回 + 规则召回 + 召回融合
- [x] W3：DeepFM 排序 + 5 步混排流程 + 降级策略 + 离线评估（AUC / 真实 CTR）
- [ ] W4：FastAPI 服务化 + 可解释面板 + 模拟 A/B 报告
- [ ] W5–W6：README 完善（架构图 / 复现 / 量化结果 / demo 视频）+ 在线 Demo（GitHub Pages + Streamlit Cloud）

---

*注：本文档随开发推进更新，勾选项为已完成模块。*
