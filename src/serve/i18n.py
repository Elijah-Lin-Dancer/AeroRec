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
        "en": "Undergrad AI portfolio · an engineering-grade recommendation pipeline "
              "(controlled experiment on synthetic data)",
        "zh": "本科 AI 申请作品 · 工程化复现工业级推荐链路（合成数据受控实验）",
    },
    "chip.recall": {"en": "4-way Recall", "zh": "四路召回"},
    "chip.rank": {"en": "DeepFM CTR Ranking", "zh": "DeepFM CTR 排序"},
    "chip.mix": {"en": "5-step Mixing", "zh": "5 步混排"},
    "chip.degrade": {"en": "Graceful Fallback", "zh": "降级兜底"},
    "chip.true_ctr": {"en": "True-CTR Evaluation", "zh": "真实 CTR 评估"},

    # ------------------------------------------------------------- sidebar
    "brand.sub": {"en": "Travel Recommender · Demo", "zh": "出行智能推荐中台 · Demo"},
    "sidebar.language": {"en": "Language", "zh": "语言"},
    "nav.label": {"en": "Navigation", "zh": "导航"},
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
              "从而能构造 **Oracle 上界**、给出**不依赖平滑估计**的评估口径——"
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
              "权重导出为 numpy 数组后加载，在线侧不依赖 torch/sklearn —— "
              "这是在「免费实例内存受限」条件下的工程取舍，详见仓库 `src/build_online_artifacts.py`。"
              "\n\n"
              "（平台说明：Streamlit Community Cloud 免费实例的资源为区间——CPU 0.078～2 核、"
              "内存 690MB～2.7GB、无 GPU；**无流量 12 小时后休眠**，访问即自动唤醒，"
              "唤醒时可能需等待约十秒。）",
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
        "zh": "拆开看每一步做了什么、中间态长什么样。这是「可解释」的落点——"
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
              "默认 `w_base=0`——候选池已由热门 + 规则召回构成，热度信息已体现在候选集里，"
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
              "因此它**严格 ≥ 所有被比较的方法**——这是一个有意义的天花板。",
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
              "增益低于完整档——这是规模差异，不是实现差异。）",
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
              "而日志为随机曝光——曝光少的物品偶然被点击就会推高平滑 CTR，形成**稀疏性偏差**，"
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
        "zh": "拖动「曝光偏差强度 τ」滑块，实时观察有偏曝光如何把推荐排序**搞反**——"
              "这是本项目研究主线（偏差传播诊断 + 去偏有效边界）的交互版。",
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
              "成为**已知量**，曝光策略的 propensity 又是策略参数的**闭式函数**——"
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
        "en": "Current τ={tau:.1f} → ranking-level spearman={sp:+.3f} (below the zero line = the “ranking inverted” region).",
        "zh": "当前 τ={tau:.1f} → 排序级 spearman={sp:+.3f}（零线以下即「排序被反转」区）。",
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
        "zh": "离线用 torch 训练，权重导出为 numpy（约 29KB）后由在线侧纯 numpy 前向加载——"
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
              "两者**特征 schema 不同、不能互换用于同一份物品打分**——"
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

    # ------------------------------------------------------------ footer
    "foot.body": {
        "en": "<b>AeroRec</b> · Travel Recommendation Platform · Undergrad AI portfolio<br>"
              "This instance runs the micro tier (10k users × 10k items × 500k impressions); data is "
              "script-generated and reproducible; the model is offline pre-trained + numpy inference, "
              "with no torch / sklearn on the online side.<br>"
              "<a href=\"https://github.com/Elijah-Lin-Dancer/AeroRec\">Repository</a> · "
              "<a href=\"https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/docs/bias_report.md\">Bias-diagnosis tech report</a> · "
              "<a href=\"https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/data/bias/report.html\">Visual report</a>",
        "zh": "<b>AeroRec</b> · 出行智能推荐中台 · 本科 AI 申请作品<br>"
              "本实例运行 micro 档（1 万用户 × 1 万物品 × 50 万曝光），数据由脚本合成、可复现；"
              "模型为离线预训练 + numpy 推理，在线侧不依赖 torch / sklearn。<br>"
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
