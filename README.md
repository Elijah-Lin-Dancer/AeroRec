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
- [x] 轻量特征库：集中产出召回 / 排序 / 冷启动共用特征（dense + sparse）
- [ ] CTR 动态排序：归一化 → CTR 加权 → 分层排序 → 同品类打散 → 分页去重（W2–W3）
- [ ] DeepFM 排序模型（PyTorch）+ Two-Tower 向量召回 + 规则召回（W2–W3）
- [ ] 降级兜底：CTR 服务超时 → 近 7 天平均 CTR 兜底（W2）
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
        (热门+协同)        (CTR 模型)          (7日均值兜底)
            └───────────────────┼───────────────────┘
                                ▼
               Data: 合成出行交互生成器(可复现)
```

## 快速开始

```bash
pip install -r requirements.txt
python -m src.pipeline_demo        # 生成数据 + 训练召回基线 + 样例推荐
python -m pytest tests/ -q         # 冒烟测试
```

## 数据与复现

数据由 `src/data/synthetic_generator.py` 生成（固定随机种子，可一键复现）：

- **物品**：5 类目共 100,000 个（每类目 2 万），各带质量分 `quality`、现金价、里程价、出发/到达城市、航班时段、库存。
- **用户**：100,000 个，对各品类有亲和度 `aff_*`，附里程余额、常驻城市、价格敏感度、商务标签。
- **未来行程**：约 14.8 万条，支撑"多段行程按日期优先返程"等出行业务规则。
- **曝光日志**：500 万条，每条按"潜在真实 CTR"抽样点击、点击后按 CVR 抽样转化。

因已知潜在真实 CTR，离线评估（AUC、线上增益）具备可比性。生成产物为 CSV，位于 `data/processed/`（已 gitignore，用脚本重生）。

## 方法（规划）

1. **召回**：热门兜底 + ItemCF，产出候选集（后续接入 Two-Tower 向量召回与规则召回）。
2. **排序（W2–W3）**：复刻工业 5 步流程——分数归一化 → CTR 加权 → 分层排序 → 同品类打散 → 分页去重；CTR 模型升级为 DeepFM（PyTorch）。
3. **降级（W2）**：CTR 服务超时 → 降级为近 7 天平均 CTR 兜底，保证核心链路不挂。

## 目录结构

```
src/
  config.py              全局配置
  data/synthetic_generator.py   合成数据生成器
  feature/store.py             轻量特征库
  recall/baselines.py          热门 + ItemCF 召回
  rank/                        (W2) CTR 排序
  serve/                       (W4) FastAPI + 面板
  eval/                        (W4) 离线评估 + A/B
tests/test_data.py         数据层 + 特征库冒烟测试
```

## 诚实边界

- 数据为本项目**合成**（受控实验），仅用于演示工程链路；不涉任何真实业务数据，也不与任何真实企业系统关联。
- 数据生成公式中注入了隐式非线性交叉项，用于让深度模型可学；模型只能拿到原始特征，交叉需自行学习——此设计明确标注为受控实验设置。
- CTR 模型使用标准模型（scikit-learn / PyTorch 实现的 DeepFM、Two-Tower 等均为已有方法的实现），本项目**不提出新算法**。
- 实习经历仅作为项目动机来源（业务规则与流程启发），实习中未涉及深度排序模型的训练。

## 待办（Roadmap）

- [ ] W2：Two-Tower 召回 + 规则召回 + 召回融合 + 降级策略
- [ ] W3：DeepFM 排序 + 5 步混排流程 + 离线评估（AUC / Recall@k）
- [ ] W4：FastAPI 服务化 + 可解释面板 + 模拟 A/B 报告
- [ ] W5–W6：README 完善（架构图/复现/量化结果/demo 视频）+ 在线 Demo（GitHub Pages + Streamlit Cloud）

---
*注：本文档随开发推进更新，勾选项为已完成模块。*
