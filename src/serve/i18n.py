"""Streamlit Demo 的中英双语文案表（i18n）。

设计
----
- 单一真源：所有面向用户的文案集中在 `TEXT`，键为语义 key，值为 `{"en": ..., "zh": ...}`。
- 会话安全：**不使用模块级可变语言状态**（Streamlit 同进程多会话共享模块变量会串语言）。
  改由 `make_t(lang)` 返回一个闭包 `t(key, **fmt)`，`lang` 由调用方（app.py）从
  `st.session_state` 读取后传入，从而每个浏览器会话各持其语言。
- 缺失键：回退为返回 key 本身，便于开发期发现漏配。

用法
----
    from src.serve.i18n import LANGS, LANG_LABELS, make_t
    t = make_t("en")          # 或 make_t(st.session_state["lang"])
    t("hero.title")
    t("demo.summary_user", user_id=7, top_k=10)
"""
from __future__ import annotations

LANGS = ("en", "zh")
LANG_LABELS = {"en": "English", "zh": "中文"}
DEFAULT_LANG = "en"

# 语言无关的常量（资源名、模式名、方法名等），避免在页面里散落硬编码
MODE_LABELS = {
    "en": {"micro": "micro (lightweight)", "lite": "lite", "full": "full"},
    "zh": {"micro": "micro（极轻档）", "lite": "lite（轻量档）", "full": "full（完整档）"},
}


TEXT: dict[str, dict[str, str]] = {
    # ---------------------------------------------------------------- meta
    "app.page_title": {
        "en": "AeroRec · Travel Recommender Platform",
        "zh": "AeroRec · 出行智能推荐中台",
    },
    "hero.title": {
        "en": "AeroRec · Travel Recommendation Platform",
        "zh": "AeroRec · 出行智能推荐中台",
    },
    "hero.sub": {
        "en": "An engineering-grade recommendation pipeline "
              "(controlled experiment on synthetic data)",
        "zh": "工程化复现工业级推荐链路（合成数据受控实验）",
    },
    "chip.recall": {"en": "4-way Recall", "zh": "四路召回"},
    "chip.rank": {"en": "DeepFM CTR Ranking", "zh": "DeepFM CTR 排序"},
    "chip.mix": {"en": "5-step Mixing", "zh": "5 步混排"},
    "chip.degrade": {"en": "Graceful Fallback", "zh": "降级兜底"},
    "chip.true_ctr": {"en": "True-CTR Evaluation", "zh": "真实 CTR 评估"},
    # 页头下方的「已发表预印本」条：这是全站权重最高的可信度信号，
    # 放在 hero 内（而非页脚）是为了让首屏即见 DOI。
    "hero.paper": {
        "en": "📄 **Published preprint** · *Bias Propagation and Effective Boundaries of Debiasing "
              "in Recommender Systems* · Preprints 2026, 202610.2582 · "
              "[DOI 10.20944/preprints202610.2582.v2](https://doi.org/10.20944/preprints202610.2582.v2)",
        "zh": "📄 **已发表预印本** ·《Bias Propagation and Effective Boundaries of Debiasing "
              "in Recommender Systems》· Preprints 2026, 202610.2582 · "
              "[DOI 10.20944/preprints202610.2582.v2](https://doi.org/10.20944/preprints202610.2582.v2)",
    },

    # ------------------------------------------------------------- sidebar
    "brand.sub": {"en": "Travel Recommender · Demo", "zh": "出行智能推荐中台 · Demo"},
    "sidebar.language": {"en": "Language", "zh": "语言"},
    "nav.label": {"en": "Navigation", "zh": "导航"},
    "nav.overview": {"en": "Overview (start here)", "zh": "项目总览（先看这页）"},
    "nav.demo": {"en": "Demo", "zh": "推荐演示"},
    "nav.bias": {"en": "Bias Diagnosis", "zh": "偏差诊断"},
    "nav.pipeline": {"en": "Ranking Pipeline", "zh": "排序流程"},
    "nav.true_ctr": {"en": "True-CTR Comparison", "zh": "真实 CTR 对比"},
    "nav.ab": {"en": "Simulated A/B", "zh": "模拟 A/B"},
    "nav.model": {"en": "Model & Validation", "zh": "模型与验证"},
    "sidebar.mode": {"en": "Mode: **{mode}**", "zh": "当前模式：**{mode}**"},
    "sidebar.research_title": {
        "en": "**Research thread**: bias-propagation diagnosis + debiasing effective boundary",
        "zh": "**研究主线**：偏差传播诊断 + 去偏有效边界",
    },
    "sidebar.link.tech_report": {"en": "tech report", "zh": "tech report"},
    "sidebar.link.html_report": {"en": "visual report (HTML)", "zh": "可视化报告（HTML）"},
    "sidebar.link.repo": {"en": "full repository", "zh": "完整仓库"},
    "sidebar.link.highlight": {"en": "research one-pager (EN)", "zh": "研究 one-pager（EN）"},
    # 侧边栏顶部的预印本入口：访客可能只想「看一眼论文」，不该逼他先去 Overview 页找。
    "sidebar.paper_title": {
        "en": "**📄 Preprint** · Preprints 2026, 202610.2582",
        "zh": "**📄 论文预印本** · Preprints 2026, 202610.2582",
    },
    "sidebar.link.paper_doi": {"en": "DOI · publisher page", "zh": "DOI · 出版社页面"},
    "sidebar.link.paper_pdf": {"en": "published PDF", "zh": "发布版 PDF"},

    # --------------------------------------------------- scope disclosure
    "scope.title": {
        "en": "🧪 Scope & boundary disclosure (expand · affects the credibility of every page)",
        "zh": "🧪 口径与边界声明（点开查看 · 影响所有页面的结论可信度）",
    },
    "scope.data_info": {
        "en": "**Data note**: all data in this app is **synthetically generated** by "
              "`src/data/synthetic_generator.py` (fixed random seed, one-click reproducible). "
              "It **does not use — and does not claim to use — any real business data**.\n\n"
              "Synthetic data is not a shortcut: it makes the *latent true CTR* a **known quantity**, "
              "which in turn allows an **Oracle upper bound** and an evaluation protocol that "
              "**does not depend on smoothed estimates** — the very premise behind this project's "
              "quantitative claims. The cost: what is shown here is the **effectiveness of the method "
              "under controlled conditions**, not **online transferability**.",
        "zh": "**数据说明**：本应用的全部数据由 `src/data/synthetic_generator.py` **合成生成**"
              "（固定随机种子，可一键复现），**不使用、也不声称使用任何真实业务数据**。\n\n"
              "选用合成数据并非取巧：它让「潜在真实 CTR」成为**已知量**，"
              "从而能构造 **Oracle 上界**、给出**不依赖平滑估计**的评估口径，"
              "这是本项目量化结论得以成立的前提。代价是：本页展示的是"
              "**方法在受控条件下的有效性**，不代表**线上可迁移性**。",
    },
    "scope.scale_micro": {
        "en": "**This instance is the lightweight tier**: 10k users × 10k items × 500k impressions "
              "(the full tier is 100k × 100k × 5M), measured peak process memory ~297 MB, running on "
              "a free instance (memory floor 690 MB) — for **interactive experience only**. "
              "Because the data scale is far smaller than the full tier, **numbers on this page are not "
              "paper-grade conclusions**; citable conclusions should follow the offline reproduction "
              "of the full tier (see the GitHub repository README and report pages).",
        "zh": "**本实例为极轻档**：1 万用户 × 1 万物品 × 50 万曝光（完整档为 10 万 × 10 万 × 500 万），"
              "实测进程峰值内存 约 297MB，运行于免费实例（内存下限 690MB），仅用于**交互体验**。"
              "因数据规模远小于完整档，**本页数值不可作为论文级结论**；"
              "可引用结论请以完整档离线复现为准（见 GitHub 仓库 README 与展示页）。",
    },
    "scope.scale_lite": {
        "en": "**This instance is the lite tier**: 20k users × 20k items × 1M impressions "
              "(the full tier is 100k × 100k × 5M), measured peak process memory ~960 MB, running on "
              "a 2 GB instance — for **interactive experience only**. "
              "Because the data scale is far smaller than the full tier, **numbers on this page are not "
              "paper-grade conclusions**; citable conclusions should follow the offline reproduction "
              "of the full tier (see the GitHub repository README and report pages).",
        "zh": "**本实例为轻量档**：2 万用户 × 2 万物品 × 100 万曝光（完整档为 10 万 × 10 万 × 500 万），"
              "实测进程峰值内存 约 960MB，运行于 2GB 内存的实例，仅用于**交互体验**。"
              "因数据规模远小于完整档，**本页数值不可作为论文级结论**；"
              "可引用结论请以完整档离线复现为准（见 GitHub 仓库 README 与展示页）。",
    },
    "scope.inference_note": {
        "en": "ℹ️ The model in this instance is **pre-trained offline + numpy inference**: training is "
              "done locally, weights are exported as numpy arrays and loaded; the online side does not "
              "depend on torch/sklearn — an engineering trade-off under the free-instance memory limit. "
              "See `src/build_online_artifacts.py` in the repository.\n\n"
              "(Platform note: Streamlit Community Cloud free instances have a resource *range* — "
              "CPU 0.078–2 cores, memory 690 MB–2.7 GB, no GPU; **the app sleeps after 12 h of "
              "inactivity** and wakes on access, which may take ~10 s.)",
        "zh": "ℹ️ 本实例的模型为**离线预训练 + numpy 推理**：训练阶段在本机完成，"
              "权重导出为 numpy 数组后加载，在线侧不依赖 torch/sklearn，"
              "这是在「免费实例内存受限」条件下的工程取舍，详见仓库 `src/build_online_artifacts.py`。"
              "\n\n"
              "（平台说明：Streamlit Community Cloud 免费实例的资源为区间：CPU 0.078～2 核、"
              "内存 690MB～2.7GB、无 GPU；**无流量 12 小时后休眠**，访问即自动唤醒，"
              "唤醒时可能需等待约十秒。）",
    },

    # -------------------------------------------------------- page: overview
    # 设计意图：绝大多数访客既不会读论文、也不会翻仓库源码。这一页必须在
    # **一屏到两屏内**回答四个问题：这是什么 / 做了什么 / 关键结论是什么 /
    # 为什么值得在意。故文案全部写成「结论先行 + 一句证据」，不铺参数。
    "ov.header": {"en": "Overview: what this is, and why it matters",
                  "zh": "项目总览：这是什么，为什么值得看"},
    "ov.lead": {
        "en": "**AeroRec is a travel-recommendation platform that reproduces an industrial pipeline "
              "end-to-end** — 4-way recall → DeepFM CTR ranking → 5-step mixing → graceful degradation → "
              "serving — and, on top of it, a **research thread on how biased exposure quietly destroys "
              "rankings** (published as a preprint, Preprints 2026, 202610.2582).\n\n"
              "The enabling trick is that the data is **synthetic with an analytically known latent CTR**. "
              "That turns bias from something you *estimate* into something you can **measure** — which is "
              "why the numbers below are citable rather than illustrative.",
        "zh": "**AeroRec 是一个端到端复现工业级链路的出行推荐平台**——四路召回 → DeepFM CTR 排序 → "
              "5 步混排 → 降级兜底 → 服务化；在此之上还有一条**关于「有偏曝光如何悄悄摧毁排序」的研究主线**"
              "（已发表预印本，Preprints 2026, 202610.2582）。\n\n"
              "让这一切成立的关键，是数据为**合成数据、潜在真实 CTR 解析已知**："
              "于是偏差从「只能估计」变成「可以精确测量」，"
              "下面这些数字因此可引用，而不只是示意。",
    },

    # 四个「做了什么」的卡片
    "ov.m_gain": {"en": "True CTR vs popularity", "zh": "真实 CTR vs 热门"},
    "ov.m_gain_help": {
        "en": "Full tier, 100 users × Top-10, computed from the known latent true CTR — citable, and "
              "unaffected by smoothing-sparsity bias.",
        "zh": "完整档，100 用户 × Top-10，由已知潜在真实 CTR 直接算出——可引用，且不受平滑稀疏性偏差影响。",
    },
    "ov.m_oracle": {"en": "of the Oracle ceiling", "zh": "占 Oracle 上界"},
    "ov.m_oracle_help": {
        "en": "Per-user best possible Top-10 within the candidate pool: the remaining headroom is the pool, "
              "not the model.",
        "zh": "逐用户候选池内理论最优 Top-10：剩下的差距来自候选池，不是模型。",
    },
    "ov.m_ess": {"en": "Debiasing boundary", "zh": "去偏有效边界"},
    "ov.m_ess_help": {
        "en": "Below ESS/N ≈ 0.358, IPS/SNIPS gains turn negative. Needs no ground truth → deployable as an "
              "online warning.",
        "zh": "ESS/N ≈ 0.358 以下，IPS/SNIPS 收益转负。不需要 ground-truth → 可作为线上预警落地。",
    },
    # 中文全角括号占位更宽，带「100万行」会在四列布局里被截断，故把口径移进 help。
    "ov.m_real": {"en": "Real-data AUC (Criteo)", "zh": "真实数据 AUC"},
    "ov.m_real_help": {
        "en": "Same DeepFM implementation, zero model changes; LR baseline 0.7716, constant 0.5000.",
        "zh": "同一套 DeepFM 实现、模型零改动；LR 基线 0.7716，常数基线 0.5000。",
    },
    "ov.what_title": {"en": "What it actually does", "zh": "它到底做了什么"},
    "ov.c1_title": {"en": "① The full industrial chain", "zh": "① 完整工业链路"},
    "ov.c1_body": {
        "en": "4-way recall (**popularity / ItemCF / Two-Tower / travel rules**: mileage redemption, "
              "multi-leg return priority) → fusion → **DeepFM CTR + LR CVR** multi-objective ranking → "
              "**5-step mixing** (normalize / weight / stratify / de-cluster / dedup) → graceful fallback to "
              "historical mean CTR → FastAPI + Streamlit. The same algorithm code runs three data tiers.",
        "zh": "四路召回（**热门 / ItemCF / Two-Tower / 出行规则**：里程兑换、多段行程优先返程）→ 融合 → "
              "**DeepFM CTR + LR CVR** 多目标排序 → **5 步混排**（归一化 / 加权 / 分层 / 打散 / 去重）→ "
              "异常时降级为历史平均 CTR → FastAPI + Streamlit。同一套算法代码跑三档数据规模。",
    },
    "ov.c2_title": {"en": "② Serving discipline, actually measured", "zh": "② 服务侧纪律，实测而非估算"},
    "ov.c2_body": {
        "en": "Training and inference are **decoupled**: weights are exported to numpy, so the online side "
              "**never imports torch** (`torch + sklearn + numpy + streamlit` = 673 MB baseline, i.e. it cannot "
              "fit a free 690 MB instance at all). Measured: engine peak **129 MB**, single-thread "
              "**P50 0.70 ms / P99 0.95 ms**, ~1378 QPS. PyTorch vs numpy forward max deviation **1.19e-07** — "
              "same trained model, different runtime.",
        "zh": "训练与推理**解耦**：权重导出为 numpy，在线侧**永不 import torch**"
              "（`torch+sklearn+numpy+streamlit` 基线 673MB，在免费实例 690MB 下限上根本装不下）。"
              "实测：引擎峰值 **129MB**、单线程 **P50 0.70ms / P99 0.95ms**、约 1378 QPS；"
              "PyTorch 与 numpy 前向最大偏差 **1.19e-07**——同一个已训练模型，只换了运行时。",
    },
    "ov.c3_title": {"en": "③ An evaluation protocol worth trusting", "zh": "③ 一套站得住脚的评估口径"},
    "ov.c3_body": {
        "en": "Rankings are scored against the **known latent true CTR**, plus a per-user **Oracle upper "
              "bound** over the candidate pool (popular ∪ rule ∪ random, so it is strictly ≥ every compared "
              "method). **No smoothed-history estimate is involved**, so the gain is not an artifact of "
              "sparsity bias — the flaw that makes the naive \"simulated A/B\" number untrustworthy.",
        "zh": "用**已知的潜在真实 CTR** 给各方法的 Top-K 打分，并给出逐用户候选池的 **Oracle 上界**"
              "（池 = 热门 ∪ 规则 ∪ 随机，故严格 ≥ 所有被比方法）。"
              "**不依赖任何平滑历史估计**，因此增益不是稀疏性偏差的产物——"
              "而正是这个偏差让「模拟 A/B」的那组数字不可引用。",
    },
    "ov.c4_title": {"en": "④ Real-data transferability", "zh": "④ 真实数据迁移性"},
    "ov.c4_body": {
        "en": "The *same* DeepFM implementation, **zero model changes**: validation AUC **0.7733** on Criteo "
              "Display Advertising Challenge (1 M rows) and **0.9183** on UCI Adult. This claims "
              "*the implementation learns real signal* — **not** that Criteo weights can rank travel items "
              "(different feature schema). We report honestly that the edge over an LR baseline on Criteo is "
              "only 0.0017.",
        "zh": "**同一套** DeepFM 实现、**模型零改动**：Criteo Display Advertising Challenge（100 万行）"
              "验证 AUC **0.7733**，UCI Adult **0.9183**。这证明的是*实现能在真实数据上学到信号*，"
              "**不是**「Criteo 权重可以直接给出行物品排序」（特征 schema 不同）。"
              "同时如实标注：Criteo 上相对 LR 基线仅高 0.0017。",
    },

    # 研究主线三个发现
    "ov.k_title": {"en": "The research thread, in three findings", "zh": "研究主线：三个关键发现"},
    "ov.k1_title": {"en": "The harm is inversion, not inflation", "zh": "危害是排序反转，不是整体偏高"},
    "ov.k1_body": {
        "en": "As exposure skew grows (π ∝ popularity^τ), ranking-level Spearman against the true CTR falls "
              "from **+0.87 (τ=0) to −0.51 (τ=3)**: at τ=3 the leaderboard is *reversed* — cold items, pulled "
              "toward the smoothing prior by a few accidental clicks, outrank genuinely good ones "
              "(30 seeds, bootstrap p < 0.001). A mean-only view would call this \"modest bias\".",
        "zh": "曝光越偏斜（π ∝ 热度^τ），排序级与真实 CTR 的 Spearman 从 **+0.87（τ=0）跌到 −0.51（τ=3）**："
              "τ=3 时排行榜被*反转*——冷门物品靠少数偶然点击被平滑先验拉高，反超真正优质的物品"
              "（30 种子，bootstrap p < 0.001）。只看「平均 CTR 偏高多少」会得出「偏差可控」的错误结论。",
    },
    "ov.k2_title": {"en": "Debiasing has an effective boundary", "zh": "去偏存在有效边界"},
    "ov.k2_body": {
        "en": "Below **ESS/N ≈ 0.358**, IPS/SNIPS gains turn **negative** — the variance cost of importance "
              "weights exceeds the correction benefit (stable across an 8× scale-up). The boundary is set by "
              "the **propensity dynamic range**, not by bias strength. Crucially `ESS/N` needs **no ground "
              "truth**, so it ships as a deployable warning signal: `ess_warning_from_log(log, threshold=0.358)`.",
        "zh": "**ESS/N ≈ 0.358** 以下，IPS/SNIPS 收益转**负**——重要性权重的方差代价超过纠偏收益"
              "（8 倍扩规模后仍稳定）。边界由 **propensity 动态范围**决定，不是由偏差强度决定。"
              "关键在于 `ESS/N` **不需要 ground-truth**，因此可作为线上预警落地："
              "`ess_warning_from_log(log, threshold=0.358)`。",
    },
    "ov.k3_title": {"en": "User-side bias needs user-item granularity", "zh": "用户侧偏差必须在 user-item 粒度建模"},
    "ov.k3_body": {
        "en": "Self-selection drifts the pooled-CTR estimand from **0.281 to 0.702** (a 2.5× apparent "
              "inflation). Debiasing with **item-marginal** propensities leaves a significant residual "
              "(**+0.06 → +0.22**, p < 0.001) — only correct **user-item-level** propensity debiasing keeps "
              "error at 0.004–0.037.",
        "zh": "用户自选让池化 CTR 的估计目标从 **0.281 漂到 0.702**（虚高 2.5 倍）。"
              "用**物品边际** propensity 去偏会留下显著残余（**+0.06 → +0.22**，p < 0.001）——"
              "只有**user-item 粒度**的正确 propensity 去偏才能把误差压在 0.004–0.037。",
    },

    # 为什么值得在意
    "ov.why_title": {"en": "Why it matters", "zh": "为什么值得在意"},
    "ov.why_body": {
        "en": "- **It reframes \"bias\"** from *a number that runs too high* into a **ranking-integrity "
              "failure** — which is the thing actually worth monitoring in production.\n"
              "- **It turns \"should we debias?\" into a decision rule** that needs no ground truth, hence "
              "usable on real logs where the truth is unknowable.\n"
              "- **It shows the engineering the claims rest on**: decoupled serving, three data tiers, "
              "**133 tests**, and a **byte-for-byte drift gate** on 7 result JSONs re-run by CI on every push.\n"
              "- **It keeps the boundary explicit**: synthetic data proves *method validity under controlled "
              "conditions*, not *online transferability* — and says so everywhere.",
        "zh": "- **它把「偏差」重新定义**：不是*某个数字偏高*，而是**排序完整性的失效**——"
              "这才是生产环境真正该监控的东西。\n"
              "- **它把「要不要去偏」变成一条判据**，且不需要 ground-truth，因此在真值不可知的真实日志上也能用。\n"
              "- **它把结论所依赖的工程摆出来**：训练推理解耦、三档数据规模、**133 项测试**、"
              "以及 CI 每次提交都对 7 个结果 JSON 做**逐字节零漂移门禁**。\n"
              "- **它不藏边界**：合成数据证明的是*方法在受控条件下的有效性*，不是*线上可迁移性*——并且到处都这么说。",
    },

    # 事实表
    "ov.facts_title": {"en": "At a glance", "zh": "一眼看全"},
    "ov.fact.item": {"en": "Item", "zh": "项目"},
    "ov.fact.value": {"en": "Value", "zh": "内容"},
    "ov.fact.paper_k": {"en": "Preprint", "zh": "预印本"},
    "ov.fact.paper_v": {
        "en": "Preprints 2026, 202610.2582 · DOI 10.20944/preprints202610.2582.v2 (posted 2026-11-06)",
        "zh": "Preprints 2026, 202610.2582 · DOI 10.20944/preprints202610.2582.v2（2026-11-06 发布，v1 为 2026-10-27）",
    },
    "ov.fact.chain_k": {"en": "Pipeline", "zh": "链路"},
    "ov.fact.chain_v": {
        "en": "4-way recall → DeepFM CTR + LR CVR → 5-step mixing → fallback → FastAPI / Streamlit",
        "zh": "四路召回 → DeepFM CTR + LR CVR → 5 步混排 → 降级兜底 → FastAPI / Streamlit",
    },
    "ov.fact.tiers_k": {"en": "Data tiers", "zh": "数据档位"},
    "ov.fact.tiers_v": {
        "en": "full 100k×100k×5M (citable) · lite 20k×20k×1M · micro 10k×10k×500k (this app)",
        "zh": "full 10万×10万×500万（可引用）· lite 2万×2万×100万 · micro 1万×1万×50万（本应用）",
    },
    "ov.fact.result_k": {"en": "Citable result (full tier)", "zh": "可引用结论（完整档）"},
    "ov.fact.result_v": {
        "en": "DeepFM+mixing: +28.5% true CTR vs popularity, reaching 88.4% of the Oracle bound",
        "zh": "DeepFM+混排：真实 CTR 相对热门 +28.5%，达到 Oracle 上界的 88.4%",
    },
    "ov.fact.bias_k": {"en": "Bias research", "zh": "偏差研究"},
    "ov.fact.bias_v": {
        "en": "7 experiments, 30-seed mean ± std, numpy-only bootstrap B=10000, all in `data/bias/*.json`",
        "zh": "7 个实验，30 种子均值±std，纯 numpy bootstrap B=10000，全部落盘 `data/bias/*.json`",
    },
    "ov.fact.real_k": {"en": "Real-data validation", "zh": "真实数据验证"},
    "ov.fact.real_v": {
        "en": "Criteo 1M AUC 0.7733 (LR 0.7716) · Criteo 200k 0.7586±0.0007 (3 seeds) · UCI Adult 0.9183",
        "zh": "Criteo 100万行 AUC 0.7733（LR 0.7716）· Criteo 20万行 0.7586±0.0007（3 种子）· UCI Adult 0.9183",
    },
    "ov.fact.gate_k": {"en": "Quality gates", "zh": "质量门禁"},
    "ov.fact.gate_v": {
        "en": "133 tests · 7 result JSONs byte-for-byte drift-gated in CI · ruff + mypy · "
             "assertion that the online engine never loads torch",
        "zh": "133 项测试 · CI 对 7 个结果 JSON 逐字节零漂移门禁 · ruff + mypy · "
              "断言在线引擎不加载 torch",
    },

    # 导航地图
    "ov.map_title": {"en": "Where to look next", "zh": "接下来看哪里"},
    "ov.map.page": {"en": "Page", "zh": "页面"},
    "ov.map.what": {"en": "What you'll see", "zh": "你能看到什么"},
    "ov.map.demo": {"en": "Live Top-K for any user, with per-item predicted CTR / CVR and the reason",
                    "zh": "任意用户的实时 Top-K，含逐条预估 CTR / CVR 与推荐理由"},
    "ov.map.bias": {"en": "Drag exposure skew τ and watch the ranking invert in real time; then the ESS/N boundary table",
                    "zh": "拖动曝光偏斜 τ，实时看排序如何被反转；再看 ESS/N 边界表"},
    "ov.map.pipeline": {"en": "The 5 mixing steps opened up, stage by stage",
                    "zh": "把 5 步混排逐步拆开，看每一级的中间态"},
    "ov.map.true_ctr": {"en": "The citable comparison: every method vs the Oracle ceiling",
                    "zh": "可引用的对比：各方法 vs Oracle 上界"},
    "ov.map.ab": {"en": "A simulated A/B, kept deliberately — and labelled non-citable",
                    "zh": "模拟 A/B，特意保留并明确标注为不可引用"},
    "ov.map.model": {"en": "DeepFM architecture, real-data AUC, and a live numpy forward pass",
                    "zh": "DeepFM 结构、真实数据 AUC、以及一次真实的 numpy 前向"},

    # 引用
    "ov.cite_title": {"en": "If you use this work, please cite", "zh": "如果这项工作的内容对你有用，请引用"},
    "ov.cite_body": {
        "en": "Zhao, Y. Bias Propagation and Effective Boundaries of Debiasing in Recommender Systems: "
              "An Oracle-Based Diagnosis Study. *Preprints* **2026**, 202610.2582. "
              "https://doi.org/10.20944/preprints202610.2582.v2",
        "zh": "Zhao, Y. Bias Propagation and Effective Boundaries of Debiasing in Recommender Systems: "
              "An Oracle-Based Diagnosis Study. *Preprints* **2026**, 202610.2582. "
              "https://doi.org/10.20944/preprints202610.2582.v2",
    },
    "ov.cite_bibtex": {"en": "BibTeX", "zh": "BibTeX"},

    # 页尾的诚实边界（Overview 必须自带，否则「精炼介绍」会变成「过度宣传」）
    "ov.boundary": {
        "en": "**Honest boundary** — all data here is synthetic (script-generated, seeded, reproducible); "
              "**no real business data is used or claimed**. What the numbers establish is *method validity "
              "under controlled conditions*, **not** online transferability. The full limitation list lives in "
              "the [README](https://github.com/Elijah-Lin-Dancer/AeroRec#honest-limitations).",
        "zh": "**诚实边界**——本页全部数据为合成（脚本生成、固定种子、可复现），"
              "**不使用、也不声称使用任何真实业务数据**。这些数字确立的是*方法在受控条件下的有效性*，"
              "**不是**线上可迁移性。完整局限清单见 "
              "[README](https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/README.zh.md#诚实边界与局限性)。",
    },

    # ------------------------------------------------------------ page: demo
    "demo.header": {"en": "Recommendation Demo", "zh": "推荐演示"},
    "demo.caption": {
        "en": "Pick a user and Top-K to see the final recommendations from the full pipeline "
              "(recall → ranking → mixing).",
        "zh": "选择用户与 Top-K，观察完整链路（召回 → 排序 → 混排）的最终推荐结果。",
    },
    "demo.user_id": {"en": "User ID", "zh": "用户 ID"},
    "demo.user_id_help": {
        "en": "Range 0–{n}, corresponds to one user in the synthetic data",
        "zh": "取值 0–{n}，对应合成数据中的一位用户",
    },
    "demo.topk": {"en": "Top K", "zh": "Top K"},
    "demo.run": {"en": "Generate recommendations", "zh": "生成推荐"},
    "demo.empty": {"en": "This user has no recommendations.", "zh": "该用户无推荐结果"},
    "demo.subheader": {
        "en": "Top-{top_k} recommendations for user {user_id}",
        "zh": "用户 {user_id} 的 Top-{top_k} 推荐",
    },
    "demo.m_avg_ctr": {"en": "Avg. predicted CTR", "zh": "平均预估 CTR"},
    "demo.m_max_ctr": {"en": "Max predicted CTR", "zh": "最高预估 CTR"},
    "demo.m_categories": {"en": "Categories covered", "zh": "覆盖品类数"},
    "demo.card_ctr": {"en": "pred. CTR", "zh": "预估 CTR"},
    "demo.card_base": {"en": "base score", "zh": "基础分"},
    "demo.expand_fields": {"en": "View full fields (incl. internal scores)", "zh": "查看完整字段（含内部打分）"},
    "demo.footer_caption": {
        "en": "CTR = DeepFM predicted click-through rate; CVR = LR predicted conversion rate; "
              "base score = recall-stage score. Results are produced in one call to `engine.recommend()`, "
              "reproducible (fixed random seed).",
        "zh": "CTR = DeepFM 预估点击率；CVR = LR 预估转化率；基础分 = 召回阶段得分。"
              "结果由 `engine.recommend()` 一次产出，可复现（固定随机种子）。",
    },

    # -------------------------------------------------------- page: pipeline
    "pipe.header": {"en": "5-step Mixing Pipeline (interpretable)", "zh": "5 步混排流程（可解释）"},
    "pipe.caption": {
        "en": "Open up each step to see what it does and what the intermediate state looks like. "
              "This is where *interpretability* lands — not just the result, but the path to it.",
        "zh": "拆开看每一步做了什么、中间态长什么样。这是「可解释」的落点，"
              "不只给结果，还给到达结果的路径。",
    },
    "pipe.s1": {"en": "① Normalize", "zh": "① 归一化"},
    "pipe.s1d": {"en": "Put scores on a common scale so a large-magnitude metric can't dominate the weighting",
                 "zh": "把各分数压到同一量纲，避免量纲大的指标主导加权"},
    "pipe.s2": {"en": "② Multi-objective weighting", "zh": "② 多目标加权"},
    "pipe.s2d": {"en": "w_base·base + w_ctr·(0.7×CTR + 0.3×CVR), default w_base=0",
                 "zh": "w_base·基础分 + w_ctr·(0.7×CTR + 0.3×CVR)，默认 w_base=0"},
    "pipe.s3": {"en": "③ Stratified ranking", "zh": "③ 分层排序"},
    "pipe.s3d": {"en": "Rank by score strata so different tiers alternate", "zh": "按分数分层，让不同档位交替出现"},
    "pipe.s4": {"en": "④ Same-category de-clustering", "zh": "④ 同品类打散"},
    "pipe.s4d": {"en": "Avoid consecutive items of the same category to reduce homogenization",
                 "zh": "同品类不连排，避免结果同质化"},
    "pipe.s5": {"en": "⑤ Dedup & truncate", "zh": "⑤ 去重截断"},
    "pipe.s5d": {"en": "Remove duplicates and truncate to Top-K", "zh": "剔除重复项，截断到 Top-K"},
    "pipe.user_id": {"en": "User ID", "zh": "用户 ID"},
    "pipe.run": {"en": "Show pipeline", "zh": "展示流程"},
    "pipe.trace_caption": {
        "en": "The candidate pool has **{n}** items; the table below shows how the intermediate state "
              "evolves step by step. Default `w_base=0` — the pool is already formed by popularity + "
              "rule recall, so popularity is reflected in the candidate set; the mixing stage does not "
              "re-inject popularity bias (see the empirical note in `src/rank/mixer.py`).",
        "zh": "候选池共 **{n}** 个物品；下表逐步展示中间态的变化。"
              "默认 `w_base=0`，候选池已由热门 + 规则召回构成，热度信息已体现在候选集里，"
              "混排阶段不再二次注入热门偏差（见 `src/rank/mixer.py` 实测说明）。",
    },
    "pipe.stage.candidates": {"en": "Candidate pool (after recall)", "zh": "候选池（召回后）"},
    "pipe.stage.s1": {"en": "① After normalize", "zh": "① 归一化后"},
    "pipe.stage.s2": {"en": "② After weighting", "zh": "② 多目标加权后"},
    "pipe.stage.s3": {"en": "③ After stratified ranking", "zh": "③ 分层排序后"},
    "pipe.stage.s4": {"en": "④ After de-clustering", "zh": "④ 同品类打散后"},
    "pipe.stage.final": {"en": "⑤ Dedup & truncate → final", "zh": "⑤ 去重截断 → 最终结果"},
    "pipe.no_data": {"en": "(no data at this stage)", "zh": "（该阶段无数据）"},
    "pipe.col.item": {"en": "Item", "zh": "物品"},
    "pipe.col.category": {"en": "Category", "zh": "类目"},
    "pipe.col.base": {"en": "Base score", "zh": "基础分"},
    "pipe.col.ctr": {"en": "Pred. CTR", "zh": "预估 CTR"},
    "pipe.col.cvr": {"en": "Pred. CVR", "zh": "预估 CVR"},
    "pipe.final_header": {"en": "Final ranking (in order)", "zh": "最终结果的排序（按顺序）"},
    "pipe.final_caption": {
        "en": "After de-clustering, same-category items are no longer adjacent; trailing duplicates "
              "were truncated at step ⑤.",
        "zh": "经打散后同品类不再连排；尾部重复项已在第 ⑤ 步截断。",
    },

    # ------------------------------------------------------- page: true_ctr
    "tctr.header": {"en": "True-CTR Comparison (citable)", "zh": "真实 CTR 口径对比（结论可引用）"},
    "tctr.info": {
        "en": "**Why this page is the most trustworthy**: it computes each method's Top-K average CTR "
              "directly from the **known latent true CTR**, with no dependence on any smoothed-history "
              "CTR estimate, hence **free from sparsity bias**.\n\n"
              "**Oracle upper bound** = per-user Top-K by true CTR within the candidate pool "
              "(popular ∪ rule ∪ random), therefore it is **strictly ≥ every compared method** — a "
              "meaningful ceiling.",
        "zh": "**为什么这一页最可信**：直接用**已知的潜在真实 CTR** 计算各方法 Top-K 的平均点击率，"
              "不依赖任何「平滑历史 CTR」估计，因此**不受稀疏性偏差影响**。\n\n"
              "**Oracle 上界** = 逐用户的候选池（热门 ∪ 规则 ∪ 随机）内按真实 CTR 取 Top-K，"
              "因此它**严格 ≥ 所有被比较的方法**，这是一个有意义的天花板。",
    },
    "tctr.n_users": {"en": "Sample users", "zh": "样本用户数"},
    "tctr.n_users_help": {"en": "Larger is more stable, but runtime grows linearly",
                          "zh": "越大越稳，但耗时线性增长"},
    "tctr.run": {"en": "Run true-CTR comparison", "zh": "运行真实 CTR 对比"},
    "tctr.spinner": {"en": "Evaluating ...", "zh": "评估中 ..."},
    "tctr.m_rel": {"en": "DeepFM vs popularity", "zh": "DeepFM 相对热门"},
    "tctr.m_rel_help": {"en": "Baseline: “{base}”", "zh": "以「{base}」为基线"},
    "tctr.m_oracle_pct": {"en": "Share of Oracle bound", "zh": "占 Oracle 上界"},
    "tctr.m_oracle_pct_help": {"en": "1.0 = reaches the theoretical bound", "zh": "1.0 = 达到理论上界"},
    "tctr.m_oracle_ctr": {"en": "Oracle-bound true CTR", "zh": "Oracle 上界真实 CTR"},
    "tctr.m_sample": {"en": "Sample size", "zh": "样本量"},
    "tctr.m_sample_val": {"en": "{n} users × Top-{k}", "zh": "{n} 用户 × Top-{k}"},
    "tctr.sub_table": {"en": "Method comparison", "zh": "各方法对比"},
    "tctr.col.method": {"en": "Method", "zh": "方法"},
    "tctr.col.true_ctr": {"en": "Avg. true CTR", "zh": "平均真实 CTR"},
    "tctr.col.rel": {"en": "vs {base}", "zh": "相对{base}"},
    "tctr.col.pct": {"en": "Share of Oracle", "zh": "占 Oracle 上界"},
    "tctr.sub_chart": {"en": "Avg. true CTR (higher is better)", "zh": "平均真实 CTR（越高越好）"},
    "tctr.chart_x": {"en": "Method", "zh": "方法"},
    "tctr.chart_y": {"en": "True CTR", "zh": "真实 CTR"},
    "tctr.success": {
        "en": "**Conclusion**: with {n} users and Top-{k}, DeepFM+mixing improves **{rel:+.1%}** over "
              "{base}, reaching **{pct:.1%}** of the theoretical bound. (The full tier 100k×100k×5M gives "
              "+28.5% / 88.4% offline; the online micro tier has a sparser candidate pool, so the gain is "
              "lower — a scale difference, not an implementation difference.)",
        "zh": "**结论**：{n} 用户、Top-{k} 下，DeepFM+混排 相对 {base} "
              "提升 **{rel:+.1%}**，达到理论上界的 **{pct:.1%}**。"
              "（完整档 10万×10万×500万 的离线结果为 +28.5% / 88.4%，在线 micro 档因候选池更稀疏，"
              "增益低于完整档，这是规模差异，不是实现差异。）",
    },
    "name.random": {"en": "Random", "zh": "随机"},
    "name.popularity": {"en": "Popularity", "zh": "热门"},
    "name.deepfm_mix": {"en": "DeepFM+Mixing", "zh": "DeepFM+混排"},
    "name.oracle": {"en": "Oracle", "zh": "Oracle"},

    # ----------------------------------------------------------- page: ab
    "ab.header": {"en": "Simulated A/B Report", "zh": "模拟 A/B 报告"},
    "ab.caption": {
        "en": "Control = popularity ranking; Treatment = DeepFM+mixing. Each is exposed independently "
              "and clicks/conversions are observed (offline simulation).",
        "zh": "对照组 = 热门排序；实验组 = DeepFM+混排。各自独立曝光并观测点击 / 转化（离线模拟）。",
    },
    "ab.warning": {
        "en": "**This page is methodologically limited; numbers are for reference only.** The popularity "
              "baseline is ranked by *smoothed history-log CTR*, and the log uses random exposure — an "
              "item with few impressions that happens to be clicked gets its smoothed CTR pushed up, "
              "forming **sparsity bias** that **overstates** the treatment gain. Rely on the "
              "*True-CTR Comparison* page instead.",
        "zh": "**本页口径受限，数值仅供参考。** 对照的热门基线按「历史日志平滑 CTR」排序，"
              "而日志为随机曝光，曝光少的物品偶然被点击就会推高平滑 CTR，形成**稀疏性偏差**，"
              "使实验组增益被**高估**。请以「真实 CTR 对比」页的结论为准。",
    },
    "ab.error_lite": {
        "en": "**A/B numbers in the lite tier are especially non-citable.** The lite tier has fewer and "
              "sparser items, amplifying this bias further — it may even show a negative lift.",
        "zh": "**轻量档的 A/B 数值尤其不可引用。** 轻量档物品更少、更稀疏，"
              "该偏差被进一步放大，甚至可能出现负提升。",
    },
    "ab.n_users": {"en": "Simulated users", "zh": "模拟用户数"},
    "ab.n_users_help": {"en": "Larger is more stable, but runtime grows linearly",
                        "zh": "越大越稳，但耗时线性增长"},
    "ab.run": {"en": "Run A/B simulation", "zh": "运行 A/B 模拟"},
    "ab.spinner": {"en": "Simulating ...", "zh": "模拟中 ..."},
    "ab.sub_result": {"en": "Experiment results", "zh": "实验结果"},
    "ab.m_trt_ctr": {"en": "Treatment CTR", "zh": "实验组 CTR"},
    "ab.m_ctrl_ctr": {"en": "Control CTR", "zh": "对照组 CTR"},
    "ab.m_trt_cvr": {"en": "Treatment CVR", "zh": "实验组 CVR"},
    "ab.m_ctr_p": {"en": "CTR significance p", "zh": "CTR 显著性 p"},
    "ab.m_ctr_p_help": {"en": "Two-proportion z-test; p < 0.05 is usually significant",
                        "zh": "两比例 z 检验；p < 0.05 通常视为显著"},
    "ab.delta_vs_ctrl": {"en": "{v:+.1f}% vs control", "zh": "{v:+.1f}% vs 对照"},
    "ab.sub_chart": {"en": "Treatment vs Control", "zh": "实验组 vs 对照组"},
    "ab.chart_metric": {"en": "Metric", "zh": "指标"},
    "ab.chart_ctr": {"en": "CTR (click-through)", "zh": "点击率 CTR"},
    "ab.chart_cvr": {"en": "CVR (conversion)", "zh": "转化率 CVR"},
    "ab.series_ctrl": {"en": "Control (popularity)", "zh": "对照组（热门）"},
    "ab.series_trt": {"en": "Treatment (DeepFM+mixing)", "zh": "实验组（DeepFM+混排）"},
    "ab.caption2": {
        "en": "CVR significance p={cvr_p:.4f}; post-click conversion — control {ctrl:.2%} / "
              "treatment {trt:.2%} (synthetic data, not real online metrics)",
        "zh": "CVR 显著性 p={cvr_p:.4f}；点击后转化率 对照 {ctrl:.2%} / "
              "实验 {trt:.2%}（合成数据，非真实线上指标）",
    },
    "ab.exp_title": {
        "en": "📌 On the credibility of this page's numbers (why a negative lift can appear)",
        "zh": "📌 关于本页数值的可信度说明（为什么可能出现负提升）",
    },
    "ab.exp_body": {
        "en": "**Why a negative lift can appear here?**\n\n"
              "The popularity baseline is ranked by the *smoothed CTR of the history log*. This project's "
              "log is generated with **random exposure**, so an item with very few impressions that is "
              "clicked even once gets its smoothed CTR significantly overstated (Bayesian smoothing α=5 "
              "is not enough to offset this noise). This belongs to the **position/sparsity bias** "
              "problem in recommender systems.\n\n"
              "**Full-tier measurement**: the smoothed CTR of the popularity top-200 = 0.6231 while its "
              "true CTR is only 0.2528 — a bias of **+0.37**; the smoothed top-10 overlaps the true-CTR "
              "top-10 by only **2/10**.\n\n"
              "**Implication**: using the *popularity baseline* as control will **overstate** the "
              "recommender's relative gain.\n\n"
              "**Mitigation adopted**: this project adds the **True-CTR Comparison** page, which computes "
              "each method's Top-K average CTR directly from the known latent true CTR, and gives a "
              "per-user candidate-pool **Oracle upper bound** (pool = popular ∪ rule ∪ random, guaranteed "
              "strictly ≥ all methods). That protocol depends on no smoothed estimate and its "
              "**conclusions are citable**.\n\n"
              "**Conclusion**: do not cite this page's numbers; rely on the *True-CTR Comparison* page.",
        "zh": "**为什么这里可能出现负提升？**\n\n"
              "热门基线按「历史日志的平滑 CTR」排序。本项目日志是**随机曝光**生成的，"
              "因此曝光次数很少的物品只要偶然被点击一次，其平滑 CTR 就会被显著高估"
              "（贝叶斯平滑 α=5 不足以抵消这种噪声）。这属于推荐系统中的"
              "**位置偏差 / 数据稀疏偏差**问题。\n\n"
              "**完整档实测**：热门榜 top200 的平滑 CTR = 0.6231，而其真实 CTR 仅 0.2528，"
              "偏差 **+0.37**；平滑榜 top10 与真实 CTR top10 仅 **2/10** 重合。\n\n"
              "**这意味着**：用「热门基线」作为对照，会**高估**推荐系统的相对收益。\n\n"
              "**已采取的修正**：本项目新增「**真实 CTR 对比**」页面，"
              "直接用已知的潜在真实 CTR 计算各方法 Top-K 的平均点击率，"
              "并给出逐用户候选池的 **Oracle 上界**（池 = 热门 ∪ 规则 ∪ 随机，"
              "保证严格 ≥ 所有方法）。该口径不依赖任何平滑估计，**结论可引用**。\n\n"
              "**结论**：本页数字请勿引用；请以「真实 CTR 对比」页为准。",
    },

    # --------------------------------------------------------- page: bias
    "bias.header": {"en": "Pipeline Bias Diagnosis (interactive)", "zh": "链路偏差诊断（交互）"},
    "bias.caption": {
        "en": "Drag the “exposure-bias strength τ” slider to watch in real time how biased exposure "
              "**inverts** the recommendation ranking — the interactive version of this project's "
              "research thread (bias-propagation diagnosis + debiasing effective boundary).",
        "zh": "拖动「曝光偏差强度 τ」滑块，实时观察有偏曝光如何把推荐排序**搞反**，"
              "这是本项目研究主线（偏差传播诊断 + 去偏有效边界）的交互版。",
    },
    "bias.map_note": {
        "en": "📄 This page is the **interactive version of the bias-propagation matrix** in the "
              "[tech report](https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/docs/bias_report.md): "
              "identical model-free setup and seed — the report tabulates 5 fixed τ points, here you can sweep τ continuously.",
        "zh": "📄 本页是[技术报告](https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/docs/bias_report.md)"
              "中 **偏差传播矩阵的交互版**：同一套免训练设置、同一随机种子，"
              "报告里是 5 个固定 τ 点，这里可连续拖动 τ 扫描。",
    },
    "bias.exp_title": {"en": "What is this? Why only synthetic data can do it",
                       "zh": "这是什么？为什么只有合成数据能做"},
    "bias.exp_body": {
        "en": "The fundamental difficulty of recommenders: in a real log you **know neither the ground "
              "truth nor the exposure policy**, so the biases are entangled and inseparable. This project "
              "uses **synthetic data** to make each (user, item) latent true CTR a **known quantity**, and "
              "the exposure policy's propensity a **closed-form function** of the policy parameters — so "
              "“bias = biased observation − known truth” can be measured exactly.\n\n"
              "The experiment below **trains no model**; using the true CTR as ground truth suffices — it "
              "runs in seconds and is seed-reproducible.",
        "zh": "推荐系统的根本难题：真实日志里你**既不知道真值、也不知道曝光策略**，"
              "偏差混在一起拆不开。本项目用**合成数据**让每个 (用户, 物品) 的潜在真实 CTR "
              "成为**已知量**，曝光策略的 propensity 又是策略参数的**闭式函数**，"
              "于是「偏差 = 有偏观测 − 已知真值」可以精确测量。\n\n"
              "下方实验**不训练任何模型**，用真实 CTR 作 ground-truth 即可，"
              "秒级完成、固定种子可复现。",
    },
    "bias.slider": {"en": "Exposure-bias strength τ (0 = uniform/unbiased → 3 = extremely skewed)",
                    "zh": "曝光偏差强度 τ（0 = 均匀无偏 → 3 = 极度偏斜）"},
    "bias.slider_help": {
        "en": "τ is the exponent of the popularity exposure policy: π_i ∝ popularity_i^τ. τ=0 degrades to uniform exposure.",
        "zh": "τ 是 popularity 曝光策略的指数：π_i ∝ 热度_i^τ。τ=0 退化为均匀曝光。",
    },
    "bias.m_sp": {"en": "Ranking-level spearman", "zh": "排序级 spearman"},
    "bias.m_sp_help": {"en": "Rank correlation with true CTR: ≈1 same order, <0 ranking inverted",
                       "zh": "与真实 CTR 的秩相关：≈1 同序，<0 排序被反转"},
    "bias.m_gini": {"en": "Exposure Gini", "zh": "曝光 Gini"},
    "bias.m_gini_help": {"en": "Exposure concentration: 0 = uniform, →1 = extremely concentrated",
                         "zh": "曝光集中度：0=均匀，→1=极度集中"},
    "bias.m_impr": {"en": "Min impressions (cold items)", "zh": "冷门物品最少曝光"},
    "bias.m_impr_val": {"en": "{v} times", "zh": "{v} 次"},
    "bias.m_impr_help": {"en": "The larger τ, the less cold items are exposed and the more they are overstated by smoothed CTR",
                         "zh": "τ 越大，冷门物品越少被曝光，越易被平滑 CTR 高估"},
    "bias.status_bad": {
        "en": "⚠️ At τ={tau:.1f} the ranking is **inverted** (spearman={sp:+.3f}): cold items are pushed "
              "up by a few accidental clicks, overtaking popular ones.",
        "zh": "⚠️ τ={tau:.1f} 时排序已被**反转**（spearman={sp:+.3f}）："
              "冷门物品被少数偶然点击拉高平滑 CTR，反而排到热门前面。",
    },
    "bias.status_good": {"en": "✅ At τ={tau:.1f} the ranking **matches the truth** (spearman={sp:+.3f}).",
                         "zh": "✅ τ={tau:.1f} 时排序与真值**基本一致**（spearman={sp:+.3f}）。"},
    "bias.status_warn": {"en": "At τ={tau:.1f} the ranking is **weakly correlated** with the truth (spearman={sp:+.3f}).",
                         "zh": "τ={tau:.1f} 时排序与真值**弱相关**（spearman={sp:+.3f}）。"},
    "bias.sub_line": {"en": "Ranking quality inverts as exposure bias grows", "zh": "排序质量随曝光偏差反转"},
    "bias.series_l2": {"en": "Ranking (L2)", "zh": "排序级(L2)"},
    "bias.series_l1": {"en": "Recall (L1)", "zh": "召回级(L1)"},
    "bias.series_l3": {"en": "Mixing (L3)", "zh": "混排级(L3)"},
    "bias.line_caption": {
        "en": "Current τ={tau:.1f} → ranking-level spearman={sp:+.3f} (below the zero line = the “ranking "
              "inverted” region).\n\nReading note — two curves overlap on purpose: with the project default "
              "`w_base = 0` the mixing score **is** the ranking score, so **L3 lies exactly on L2**; and L1 is "
              "derived from the exposure share alone, so it tracks τ monotonically **by construction**. "
              "Neither is a finding — they are definitions, shown so you can see where the damage enters.",
        "zh": "当前 τ={tau:.1f} → 排序级 spearman={sp:+.3f}（零线以下即「排序被反转」区）。\n\n"
              "读图说明——有两条曲线是**按定义重合**的：项目默认 `w_base = 0`，混排得分**就是**排序得分，"
              "故 **L3 严格落在 L2 上**；而 L1 只由曝光份额推导，因此**按构造**对 τ 单调。"
              "这两条都不是发现，是定义——放在图上是让你看清「破坏是从哪一级进来的」。",
    },
    "bias.sub_scatter": {"en": "True CTR vs. smoothed CTR (current τ)", "zh": "真实 CTR vs 平滑 CTR（当前 τ）"},
    "bias.scatter_x": {"en": "True CTR", "zh": "真实CTR"},
    "bias.scatter_y": {"en": "Smoothed CTR (observed)", "zh": "平滑CTR(观测)"},
    "bias.scatter_status": {"en": "Status", "zh": "状态"},
    "bias.over": {"en": "overstated", "zh": "高估"},
    "bias.under": {"en": "understated", "zh": "低估"},
    "bias.scatter_caption": {
        "en": "Points above the diagonal are overstated. Cold items (left) are pulled toward the prior "
              "1.0, hence systematically overstated.",
        "zh": "点落在对角线上方 = 被高估。冷门物品（左）被拽向先验 1.0，故被系统性高估。",
    },
    "bias.sub_bar": {"en": "Exposure concentration (current τ)", "zh": "曝光集中度（当前 τ）"},
    "bias.bar_x": {"en": "Items (by exposure, desc)", "zh": "物品(按曝光降序)"},
    "bias.bar_y": {"en": "Exposure frequency", "zh": "曝光频率"},
    "bias.bar_caption": {
        "en": "The long tail is pressed toward 0: a few popular items monopolize exposure; cold items are barely observed.",
        "zh": "长尾被压到接近 0：少数热门独占曝光，冷门几乎不被观测。",
    },

    # bias 页第二个标签：去偏的有效边界（结论落地点，不能只放在报告里）
    "bias.tab_sweep": {"en": "Exposure sweep (τ)", "zh": "曝光扫描（τ）"},
    "bias.tab_boundary": {"en": "Debiasing boundary (ESS/N)", "zh": "去偏有效边界（ESS/N）"},
    "bnd.info": {
        "en": "The first tab shows *that* bias wrecks the ranking. This one shows **when debiasing stops "
              "paying off** — the project's single most actionable result.\n\n"
              "Sweep the propensity's dynamic range `π_max/π_min`; below **ESS/N ≈ 0.358** the variance cost "
              "of importance weights exceeds the correction benefit and IPS/SNIPS **gain turns negative**. "
              "Note `naive_err` stays flat at 0.0024 across the whole sweep: the boundary is not about "
              "*when naive becomes wrong*, it is about *when paying variance to fix that 0.0024 stops being worth it*.",
        "zh": "上一个标签说明偏差**会**毁掉排序；这一页说明**去偏何时开始不划算**——本项目最可落地的一条结论。\n\n"
              "扫描 propensity 的动态范围 `π_max/π_min`：**ESS/N ≈ 0.358** 以下，重要性权重的方差代价超过纠偏收益，"
              "IPS/SNIPS 的**收益转为负**。注意 `naive_err` 全程恒为 0.0024——"
              "边界说的不是*naive 何时开始不准*，而是*花方差代价去纠正那 0.0024 何时开始不划算*。",
    },
    "bnd.sub_table": {"en": "Measured boundary (n_records = 20000, 250 seeds)",
                      "zh": "实测边界（n_records = 20000，250 种子）"},
    "bnd.col.range": {"en": "Propensity range", "zh": "propensity 范围"},
    "bnd.col.ratio": {"en": "π_max/π_min", "zh": "π_max/π_min"},
    "bnd.col.ess": {"en": "ESS/N", "zh": "ESS/N"},
    "bnd.col.naive": {"en": "Naive err", "zh": "naive 误差"},
    "bnd.col.snips": {"en": "SNIPS err", "zh": "SNIPS 误差"},
    "bnd.col.gain": {"en": "Gain", "zh": "增益"},
    "bnd.col.verdict": {"en": "Verdict", "zh": "判定"},
    "bnd.verdict_ok": {"en": "debias helps", "zh": "去偏有益"},
    "bnd.verdict_flat": {"en": "no measurable change", "zh": "无显著差异"},
    "bnd.verdict_edge": {"en": "◀ boundary", "zh": "◀ 边界"},
    "bnd.verdict_bad": {"en": "debias hurts", "zh": "去偏有害"},
    "bnd.cross": {
        "en": "**Measured boundary: ESS/N ≈ {cross:.4f}** — stable across an 8× scale-up (0.3576 → 0.3579), "
              "so it is a property of the weighting scheme, not of this sample size.",
        "zh": "**实测边界：ESS/N ≈ {cross:.4f}**——8 倍扩规模后仍稳定（0.3576 → 0.3579），"
              "说明它是加权方案的性质，不是这次抽样规模的偶然。",
    },
    "bnd.chart_title": {"en": "SNIPS error and gain vs ESS/N", "zh": "SNIPS 误差与增益随 ESS/N 的变化"},
    "bnd.chart_x": {"en": "ESS/N", "zh": "ESS/N"},
    "bnd.chart_y": {"en": "Error / gain", "zh": "误差 / 增益"},
    "bnd.series_snips": {"en": "SNIPS error", "zh": "SNIPS 误差"},
    "bnd.series_naive": {"en": "Naive error", "zh": "naive 误差"},
    "bnd.series_gain": {"en": "Gain (naive − SNIPS)", "zh": "增益（naive − SNIPS）"},
    "bnd.chart_caption": {
        "en": "Read it this way: the naive error line is flat (the estimator is not broken); what collapses "
              "is the *gain* — once ESS/N falls under the boundary, correcting costs more variance than it buys in bias.",
        "zh": "读法：naive 误差那条线是平的（估计量本身没坏）；塌下去的是*增益*——"
              "ESS/N 一旦跌到边界以下，纠偏付出的方差就超过它换回的偏差。",
    },
    "bnd.takeaway": {
        "en": "**Why this is the deployable part**: `ESS/N` requires only the propensity and the sample size — "
              "**no ground truth**. In a real system \"how large is the bias\" is exactly what you cannot "
              "measure, so a warning signal that avoids it is the difference between a research result and "
              "something you can actually ship: `ess_warning_from_log(log, threshold=0.358)`.",
        "zh": "**为什么这条最能落地**：`ESS/N` 只需要 propensity 与样本量，**不需要 ground-truth**。"
              "在真实系统里，「偏差到底有多大」恰恰是测不出来的——"
              "所以一个绕开它的预警信号，才是「研究结论」与「能上线的东西」之间的差别："
              "`ess_warning_from_log(log, threshold=0.358)`。",
    },
    "bnd.robust": {
        "en": "**Is it robust to a real system's weakness?** Yes, at this scale. Replacing the closed-form π "
              "with a **log-estimated π̂** leaves SNIPS error unchanged (Δ ≈ −0.002, **p = 0.87**), and the "
              "ESS/N warning stays **fail-safe**: false alarms possible, **zero misses**. Caveat: a real "
              "system's estimation error can be larger than modelled here, so the practical boundary would be "
              "*stricter*, not looser.",
        "zh": "**它对真实系统的软肋稳健吗？** 在本规模下是的。把闭式 π 换成**日志估计的 π̂** 后，"
              "SNIPS 误差变化不显著（Δ ≈ −0.002，**p = 0.87**），且 ESS/N 预警保持 **fail-safe**："
              "可能误报，**零漏报**。但需说明：真实系统的估计误差可能大于本实验建模的量，"
              "因此实际边界只会*更严苛*，不会更宽松。",
    },
    "bnd.caveat": {
        "en": "**Do not port the number, port the method.** 0.358 was measured at this project's scale and "
              "prior; the transferable asset is *using ESS/N as a warning*, not this constant.",
        "zh": "**可迁移的是方法，不是这个数字。** 0.358 是在本项目的规模与先验下测出的；"
              "真正可迁移的是*用 ESS/N 作预警*这件事，而不是这个常数。",
    },

    # -------------------------------------------------------- page: model
    "model.header": {"en": "Model Architecture & Real-Data Validation", "zh": "模型架构与真实验证"},
    "model.caption": {
        "en": "The CTR ranker is a standard DeepFM (Guo et al. 2017). Below we show that the *same* "
              "implementation reaches industry-grade AUC on **real data**, and that this Demo's online "
              "inference is numerically equivalent to it.",
        "zh": "本项目 CTR 排序器是标准 DeepFM（Guo et al. 2017）。下方证明其同一套实现"
              "能在**真实数据**上达到业界水平 AUC，且本 Demo 在线推理与它数值等价。",
    },
    "model.sub_online": {"en": "Online model architecture (running live in this Demo)",
                         "zh": "在线模型架构（本 Demo 实时运行）"},
    "model.a_model": {"en": "Model", "zh": "模型"},
    "model.a_embed": {"en": "Embedding dim", "zh": "嵌入维度"},
    "model.a_hidden": {"en": "DNN hidden", "zh": "DNN 隐层"},
    "model.a_samples": {"en": "Training samples", "zh": "训练样本"},
    "model.online_caption": {
        "en": "Trained offline with torch; weights are exported as numpy (~29 KB) and loaded by a pure-numpy "
              "forward pass online — PyTorch vs numpy max deviation 1.19e-07 (per-sample check at build "
              "time). The online side does **not import torch**; peak memory ~297 MB, fitting the free "
              "instance's 690 MB floor.",
        "zh": "离线用 torch 训练，权重导出为 numpy（约 29KB）后由在线侧纯 numpy 前向加载，"
              "PyTorch vs numpy 最大偏差 1.19e-07（构建期逐样本校验）。"
              "在线侧**不 import torch**，峰值内存约 297MB，适配免费实例 690MB 下限。",
    },
    "model.sub_real": {"en": "Real-data validation (Criteo Display Advertising Challenge)",
                       "zh": "真实数据验证（Criteo Display Advertising Challenge）"},
    "model.real_info": {
        "en": "Criteo is the **standard real benchmark** for CTR / DeepFM papers and its task is fully "
              "isomorphic to this project (13 dense + 26 sparse → p(click)). The same DeepFM "
              "implementation trains directly on it with **zero model changes**.",
        "zh": "Criteo 是 CTR / DeepFM 论文的**标准真实基准**，任务与本项目完全同构"
              "（13 稠密 + 26 稀疏 → p(click)）。同一套 DeepFM 实现直接换数据即可训练，"
              "**模型零改动**。",
    },
    "model.chart_x": {"en": "Model / Data", "zh": "模型/数据"},
    "model.chart_y": {"en": "Validation AUC", "zh": "验证 AUC"},
    "model.bar.deepfm_1m": {"en": "DeepFM·1M", "zh": "DeepFM·1M"},
    "model.bar.lr_1m": {"en": "LR baseline·1M", "zh": "LR 基线·1M"},
    "model.bar.deepfm_200k": {"en": "DeepFM·200k", "zh": "DeepFM·200k"},
    "model.bar.constant": {"en": "Constant baseline", "zh": "常数基线"},
    "model.real_success": {
        "en": "**Conclusion**: the same DeepFM implementation reaches validation AUC = **{auc1m:.4f}** on "
              "real Criteo (1M rows) / **{auc200k:.4f}** (200k rows), significantly above the constant "
              "baseline 0.5000 and the LR baseline {lr:.4f}.",
        "zh": "**结论**：同一 DeepFM 实现，在真实 Criteo 数据上验证 AUC = "
              "**{auc1m:.4f}**（100 万行）/ **{auc200k:.4f}**（20 万行），"
              "显著高于常数基线 0.5000 与 LR 基线 {lr:.4f}。",
    },
    "model.exp_title": {"en": "⚠️ Honest boundary: why the travel model and the Criteo model are two models",
                        "zh": "⚠️ 诚实边界：旅游模型与 Criteo 模型为何是两个模型"},
    "model.exp_body": {
        "en": "This Demo's travel recommendation uses a DeepFM **trained on synthetic travel data** "
              "(feature schema: 13 dense + 3 sparse); the Criteo validation above uses a DeepFM "
              "**trained on real Criteo data** (13 dense + 26 sparse).\n\n"
              "The two have **different feature schemas and cannot be interchanged to score the same "
              "items** — so the point of the Criteo validation is to **prove the same implementation "
              "transfers to real data** (generalization), not to take Criteo weights and rank travel "
              "items directly.\n\n"
              "This is precisely the role of synthetic data here: it makes the *latent true CTR* a known "
              "quantity, enabling rigorous bias diagnosis and Oracle-bound evaluation; the real-data "
              "validation then covers the other half — “does the method still hold on a real distribution?”",
        "zh": "本 Demo 的旅游推荐，使用**在合成旅游数据上训练**的 DeepFM"
              "（特征维度：13 稠密 + 3 稀疏）；上面的 Criteo 验证使用"
              "**在真实 Criteo 数据上训练**的 DeepFM（13 稠密 + 26 稀疏）。\n\n"
              "两者**特征 schema 不同、不能互换用于同一份物品打分**，"
              "因此 Criteo 验证的意义是**证明同一套实现能迁移到真实数据**（泛化性），"
              "而非把 Criteo 权重直接拿来给旅游物品排序。\n\n"
              "这也正是合成数据在这个项目里的定位：它让「潜在真实 CTR」成为已知量，"
              "从而能做严谨的偏差诊断与 Oracle 上界评估；而真实数据验证则补上了"
              "「方法在真实分布上是否仍然成立」这一环。",
    },
    "model.sub_live": {"en": "This Demo's model is indeed running live", "zh": "本 Demo 的模型确实在实时推理"},
    "model.uid": {"en": "User ID", "zh": "用户 ID"},
    "model.run_live": {"en": "Run live inference", "zh": "运行实时推理"},
    "model.live_col_item": {"en": "Item", "zh": "物品"},
    "model.live_col_cat": {"en": "Category", "zh": "类目"},
    "model.live_col_ctr": {"en": "Pred. CTR", "zh": "预估 CTR"},
    "model.live_caption": {
        "en": "The CTRs above are computed by a **live forward pass** of the loaded numpy weights "
              "(not hard-coded); different users/items give different scores, proving the model truly runs.",
        "zh": "上述 CTR 由加载的 numpy 权重**实时前向计算**得出（非硬编码）；"
              "不同用户 / 物品分数不同，证明模型是真实在跑。",
    },

    # 第二个真实数据集：单靠 Criteo 会被质疑「只是碰巧它适合稀疏特征」，
    # 补一份结构完全不同（稠密为主、二分类表格数据）的数据集才构成迁移性证据。
    "model.sub_adult": {"en": "Second real dataset: UCI Adult / Census Income",
                        "zh": "第二个真实数据集：UCI Adult / Census Income"},
    "model.adult_info": {
        "en": "Deliberately a **different shape** from Criteo: 6 dense + 8 sparse (low cardinality), 48,842 "
              "rows, binary income classification — a task that has nothing to do with advertising. The same "
              "`(dense, sparse) → binary` interface and the same DeepFM code train on it unchanged.",
        "zh": "刻意选一份与 Criteo **形状不同**的数据：6 稠密 + 8 稀疏（低基数）、48,842 行、"
              "二分类收入预测——与广告毫无关系的任务。同一套 `(dense, sparse) → 二元` 接口、"
              "同一份 DeepFM 代码，零改动直接训练。",
    },
    "model.adult_success": {
        "en": "**Result**: validation AUC **{adult:.4f}** on {n_val:,} held-out rows (LR baseline "
              "{adult_lr:.4f}, constant baseline 0.5000). Combined with Criteo's 0.7733, the evidence is "
              "that *the implementation* transfers — while the travel-domain numbers remain synthetic by design.",
        "zh": "**结果**：{n_val:,} 行留出集上验证 AUC **{adult:.4f}**（LR 基线 {adult_lr:.4f}，"
              "常数基线 0.5000）。与 Criteo 的 0.7733 合起来，证明的是*实现本身*可迁移——"
              "而出行域的数值仍按设计为合成。",
    },
    "model.sub_repro": {"en": "Reproducibility is enforced, not promised", "zh": "复现性是被强制的，不是被承诺的"},
    "model.repro_info": {
        "en": "- **133 tests** (`pytest`) in CI on every push: data / recall / ranking / mixing / online "
              "engine / FastAPI error codes / bias estimators / boundary / loaders.\n"
              "- **Byte-for-byte drift gate**: CI re-runs all 7 bias experiments with **numpy only** and fails "
              "if `data/bias/*.json` differ by a single byte.\n"
              "- **Zero-heavy-dependency assertion**: the bias research path may not import "
              "torch / sklearn / pandas — enforced in CI, not by convention.\n"
              "- **Memory discipline assertion**: the online benchmark asserts `torch_loaded = false`.\n"
              "- Seeding is explicit (`src/rank/trainer.py`): DeepFM weights are initialised *after* "
              "`torch.manual_seed`, otherwise AUC drifts ~0.004 across environments.",
        "zh": "- **133 项测试**（`pytest`）每次 push 在 CI 跑：数据 / 召回 / 排序 / 混排 / 在线引擎 / "
              "FastAPI 错误码 / 偏差估计 / 边界 / 数据加载。\n"
              "- **逐字节零漂移门禁**：CI 只用 **numpy** 重跑全部 7 个偏差实验，"
              "`data/bias/*.json` 差一个字节即失败。\n"
              "- **零重依赖断言**：偏差研究路径不得 import torch / sklearn / pandas——由 CI 强制，不靠自觉。\n"
              "- **内存纪律断言**：在线基准断言 `torch_loaded = false`。\n"
              "- 播种显式化（`src/rank/trainer.py`）：DeepFM 权重必须在 `torch.manual_seed` **之后**初始化，"
              "否则同一份代码跨环境 AUC 会漂移约 0.004。",
    },

    # ------------------------------------------------------------ footer
    "foot.body": {
        "en": "<b>AeroRec</b> · Travel Recommendation Platform<br>"
              "This instance runs the micro tier (10k users × 10k items × 500k impressions); data is "
              "script-generated and reproducible; the model is offline pre-trained + numpy inference, "
              "with no torch / sklearn on the online side.<br>"
              "Preprint: <a href=\"https://doi.org/10.20944/preprints202610.2582.v2\">DOI 10.20944/preprints202610.2582.v2</a> "
              "(Preprints 2026, 202610.2582)<br>"
              "<a href=\"https://github.com/Elijah-Lin-Dancer/AeroRec\">Repository</a> · "
              "<a href=\"https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/docs/bias_report.md\">Bias-diagnosis tech report</a> · "
              "<a href=\"https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/data/bias/report.html\">Visual report</a>",
        "zh": "<b>AeroRec</b> · 出行智能推荐中台<br>"
              "本实例运行 micro 档（1 万用户 × 1 万物品 × 50 万曝光），数据由脚本合成、可复现；"
              "模型为离线预训练 + numpy 推理，在线侧不依赖 torch / sklearn。<br>"
              "论文预印本：<a href=\"https://doi.org/10.20944/preprints202610.2582.v2\">DOI 10.20944/preprints202610.2582.v2</a>"
              "（Preprints 2026, 202610.2582）<br>"
              "<a href=\"https://github.com/Elijah-Lin-Dancer/AeroRec\">完整仓库</a> · "
              "<a href=\"https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/docs/bias_report.md\">偏差诊断 tech report</a> · "
              "<a href=\"https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/data/bias/report.html\">可视化报告</a>",
    },

    # --------------------------------------------------- loading spinners
    "spin.engine": {"en": "First load generates data and loads the model (a few seconds)...",
                    "zh": "首次加载需生成数据并加载模型（约数秒）..."},
    "spin.bias": {"en": "Computing bias-propagation sweep (~1 s)...",
                  "zh": "计算偏差传播扫描（约 1 秒）..."},
}


def make_t(lang: str):
    """返回一个绑定到 `lang` 的翻译函数 t(key, **fmt)。"""
    if lang not in LANGS:
        lang = DEFAULT_LANG

    def t(key: str, **fmt) -> str:
        entry = TEXT.get(key)
        if entry is None:
            return key
        s = entry.get(lang, entry.get(DEFAULT_LANG, key))
        if fmt:
            try:
                return s.format(**fmt)
            except (KeyError, IndexError):
                return s
        return s

    return t


CATEGORY_EN = {"航线": "Flight", "酒店": "Hotel", "商城": "Shopping",
               "营销": "Deals", "直播": "Live"}


def localize_category(lang: str, cat: str) -> str:
    """把数据里的中文类目名映射到当前语言（数据本身仍是中文，仅在展示时本地化）。"""
    if lang == "en":
        return CATEGORY_EN.get(cat, cat)
    return cat


def localize_item_name(lang: str, name: str) -> str:
    """物品名形如「<类目>-<5位ID>」；英文界面下只把类目前缀本地化，保留 ID 可追溯。"""
    if lang == "en" and "-" in name:
        pre, _, tail = name.partition("-")
        if pre in CATEGORY_EN:
            return f"{CATEGORY_EN[pre]}-{tail}"
    return name


def method_label(lang: str, key: str) -> str:
    """把内部方法名（中文）映射到当前语言的显示名，供图表/表格复用。"""
    mapping = {
        "随机": "name.random",
        "热门": "name.popularity",
        "DeepFM+混排": "name.deepfm_mix",
        "Oracle": "name.oracle",
    }
    k = mapping.get(key)
    if not k:
        return key
    return TEXT[k].get(lang, key)
