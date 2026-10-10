"""AeroRec 出行推荐中台 · 交互式 Demo（Streamlit）。

运行：streamlit run app.py

三种模式（通过环境变量 AEROREC_MODE 切换）
------------------------------------------
  micro（默认）：1 万用户 × 1 万物品 × 50 万曝光，纯 numpy 推理、**不加载 torch**，
                  实测峰值内存约 130MB，适配 1GB 内存的免费在线实例
                  （Streamlit Community Cloud）。模型权重离线预训练后加载，
                  故启动仅需约 1 秒。
  lite         ：2 万用户 × 2 万物品 × 100 万曝光，内存约 960MB（在线训练），
                  适配 2GB 内存的实例。
  full         ：10 万用户 × 10 万物品 × 500 万曝光，需较大内存与较长加载时间，
                  用于本地完整复现。

为什么默认是 micro（工程取舍，值得说明）
------------------------------------------
在线免费托管的**内存上限**是硬约束，而实测框架的固有导入开销为：

    裸 Python 9MB → +numpy 25MB → +sklearn 176MB → +torch 643MB → +streamlit 676MB

torch 一项就固定花掉 643MB，**与数据规模无关**：把数据从 100 万曝光压到
50 万曝光，峰值内存只从 847MB 降到 840MB。瓶颈是框架运行时，不是数据。

因此本项目把训练与推理**解耦**：离线（本机）用 torch 训练 DeepFM / LR，
把权重导出为 numpy（`python -m src.build_online_artifacts micro`），
在线侧只加载权重、用纯 numpy 复现前向计算。

    解耦前：峰值 840MB、启动 90s，紧贴 1GB 上限
    解耦后：峰值 130MB、启动 ~1s，留出 8 倍余量

数学等价性已在构建期逐样本校验（PyTorch vs numpy 最大偏差 1.19e-07）。

评估口径（重要）
----------------
本 Demo 有两个评估页面，口径可信度不同：

  ① 「真实 CTR 对比」—— 直接用已知的潜在真实 CTR 计算各方法 Top-K 的平均点击率，
     并给出逐用户候选池 Oracle 上界。**不受平滑偏差影响，结论可引用。**
  ② 「模拟 A/B」    —— 接近工业 A/B 的形式（曝光/点击/显著性检验），
     但对照的热门基线按「平滑历史 CTR」排序，存在稀疏性偏差，**增益被高估，仅供参考**。

注：以上两个页面的**可引用数值以完整档（full，10万×10万×500万）离线复现为准**；
在线任何档位（micro/lite）的数值仅用于交互演示。
"""
import os

import streamlit as st

# ⚠️ 这里**刻意不在模块顶层 `import pandas`**：
# 在线 micro 档走纯 numpy 路径，pandas 的固有导入开销约 150MB，
# 而免费实例内存上限仅 1GB（详见本文件顶部「工程取舍」说明）。
# `st.dataframe` / `st.bar_chart` 均可直接吃 dict 与 list[dict]，无需 pandas。
# 仅当需要 pandas 专有能力时，才在函数内部惰性导入。

MODE = os.environ.get("AEROREC_MODE", "micro").lower()

st.set_page_config(page_title="AeroRec · 出行智能推荐中台",
                   page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------
# 视觉层：统一注入一段 CSS，把默认的「未排版 Markdown」观感收敛成有层次的版式。
#
# 为什么用 CSS 注入而不是换组件库：
#   1. 在线实例内存上限 1GB，引入第三方 UI 组件（如 streamlit-extras 等）
#      会增加依赖与内存开销，与「训练/推理解耦」的内存纪律相悖；
#   2. 本项目其余可视化（docs/index.html、data/bias/report.html）已是
#      自包含、零外链的风格，此处沿用同一思路，不引入运行时外部资源。
#
# 配色沿用 src/bias/report.py 的 C 字典，保证全项目视觉一致：
#   主色 #2563eb（project blue）· 好 #16a34a（绿）· 差 #dc2626（红）
#   ⚠️ 绿/红在本项目恒为**质量语义**（绿=好、红=差），不套用股市涨跌配色。
# --------------------------------------------------------------------------
_CSS = """
<style>
/* ---- 版式基础：收窄行宽、提高标题与正文的层次差 ---- */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }

/* 主标题 + 副标题做成一个「页头」区块 */
.hero { margin: 0 0 1.1rem 0; }
.hero h1 {
  font-size: 2.05rem; font-weight: 800; letter-spacing: -0.02em;
  margin: 0 0 .35rem 0; color: #1a1d23; line-height: 1.2;
}
.hero .sub { color: #5f6672; font-size: .95rem; margin: 0; }
.hero .chain { margin-top: .6rem; display: flex; flex-wrap: wrap; gap: .35rem; }
.hero .chain span {
  background: #eef1f5; color: #334155; border-radius: 999px;
  padding: .18rem .62rem; font-size: .78rem; font-weight: 600;
  border: 1px solid #e2e8f0;
}

/* ---- 内容区标题：左侧加一条主色竖条，与正文拉开层次 ---- */
h2, h3 { letter-spacing: -0.01em; }
h2 { padding-left: .6rem; border-left: 4px solid #2563eb; line-height: 1.25; }

/* ---- 指标卡：白底 + 细边 + 悬停微交互 ---- */
div[data-testid="stMetric"] {
  background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
  padding: .85rem 1rem; box-shadow: 0 1px 2px rgba(15,23,42,.04);
}
div[data-testid="stMetricLabel"] p { font-size: .8rem !important; color: #5f6672; }
div[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700; }

/* ---- 按钮：主色实心、圆角 ---- */
div[data-testid="stButton"] > button[kind="primary"] {
  background: #2563eb; border: none; border-radius: 8px;
  padding: .5rem 1.15rem; font-weight: 600;
  box-shadow: 0 1px 2px rgba(37,99,235,.25);
}
div[data-testid="stButton"] > button[kind="primary"]:hover { background: #1d4ed8; }

/* ---- 数据表：去掉默认的粗重边框感 ---- */
div[data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 10px; }

/* ---- 推荐结果卡片（替代原先一行行刷 Markdown 的观感）---- */
.rec-card {
  display: flex; align-items: center; gap: .85rem;
  background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
  padding: .6rem .9rem; margin-bottom: .45rem;
  box-shadow: 0 1px 2px rgba(15,23,42,.03);
  transition: border-color .15s, box-shadow .15s;
}
.rec-card:hover { border-color: #c7d2fe; box-shadow: 0 2px 8px rgba(37,99,235,.08); }
.rec-rank {
  flex: 0 0 2rem; height: 2rem; border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: .85rem; color: #ffffff; background: #2563eb;
}
.rec-rank.top { background: #16a34a; }        /* 前三名用「好」色突出 */
.rec-main { flex: 1 1 auto; min-width: 0; }
.rec-name { font-weight: 650; color: #1a1d23; font-size: .95rem; }
.rec-meta { color: #5f6672; font-size: .8rem; margin-top: .12rem; }
.rec-tag {
  background: #eef1f5; color: #334155; border-radius: 6px;
  padding: .1rem .4rem; font-size: .74rem; margin-left: .35rem;
  border: 1px solid #e2e8f0;
}
.rec-cvr {
  flex: 0 0 4.6rem; text-align: right; font-size: .72rem;
  color: #5f6672; line-height: 1.25; letter-spacing: .02em;
  border-left: 1px solid #eef1f5; padding-left: .7rem;
}
.rec-cvr b { display: block; font-size: 1rem; color: #1a1d23; font-weight: 700; }

/* ---- 流程链（混排 5 步的可视化）---- */
.step-flow { display: flex; flex-wrap: wrap; align-items: center; gap: .4rem; margin: .3rem 0 .9rem 0; }
.step-chip {
  background: #ffffff; border: 1px solid #e2e8f0; border-radius: 999px;
  padding: .28rem .7rem; font-size: .82rem; color: #334155; font-weight: 600;
}
.step-chip.on { background: #2563eb; border-color: #2563eb; color: #fff; }
.step-arrow { color: #94a3b8; font-size: .85rem; }

/* ---- 页脚 ---- */
.foot {
  margin-top: 2.4rem; padding-top: 1rem; border-top: 1px solid #e2e8f0;
  color: #5f6672; font-size: .8rem; line-height: 1.7;
}
.foot a { color: #2563eb; text-decoration: none; font-weight: 600; }

/* 侧边栏：品牌块 */
.brand { padding: .2rem 0 .9rem 0; }
.brand .t { font-weight: 800; font-size: 1.02rem; color: #1a1d23; letter-spacing: -.01em; }
.brand .d { color: #5f6672; font-size: .76rem; margin-top: .15rem; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="首次加载需生成数据并加载模型（约数秒）...")
def get_engine():
    if MODE == "full":
        from src.serve.engine import RecommendationEngine
        return RecommendationEngine(), "full"
    if MODE == "lite":
        from src.serve.engine_lite import LiteRecommendationEngine
        return LiteRecommendationEngine(), "lite"
    from src.serve.engine_micro import MicroRecommendationEngine
    return MicroRecommendationEngine(), "micro"


engine, mode = get_engine()

st.markdown(
    '<div class="hero">'
    '<h1>AeroRec · 出行智能推荐中台</h1>'
    '<p class="sub">本科 AI 申请作品 · 工程化复现工业级推荐链路（合成数据受控实验）</p>'
    '<div class="chain">'
    '<span>四路召回</span><span>DeepFM CTR 排序</span><span>5 步混排</span>'
    '<span>降级兜底</span><span>真实 CTR 评估</span>'
    '</div></div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="brand">'
    '<div class="t">🧭 AeroRec</div>'
    '<div class="d">出行智能推荐中台 · Demo</div>'
    '</div>',
    unsafe_allow_html=True,
)
page = st.sidebar.radio("导航", ["推荐演示", "排序流程", "真实 CTR 对比", "模拟 A/B"])
st.sidebar.caption(f"当前模式：**{mode}**")

# 研究主线的入口。
# 这里只放链接、不嵌内容：`data/bias/report.html` 是带独立版式的完整文档，
# 用 st.components.v1.html 嵌进本页会出现「iframe 高度写死 → 只能在小框里滚」
# 与「两套视觉语言打架」两个问题，且流式沙箱下叠 iframe 有额外风险。
# 报告本质是「读」的成果，GitHub 上的渲染体验更好，故以链接形式给出。
st.sidebar.divider()
st.sidebar.markdown(
    "**研究主线**：偏差传播诊断 + 去偏有效边界\n\n"
    "- [tech report](https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/docs/bias_report.md)\n"
    "- [可视化报告（HTML）](https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/data/bias/report.html)\n"
    "- [完整仓库](https://github.com/Elijah-Lin-Dancer/AeroRec)"
)
n_users_max = {"micro": 9_999, "lite": 19_999}.get(mode, 99_999)

# 口径声明放在导航之后、且默认折叠：
# 此前三块横幅无条件铺在每页顶部，占掉约 600px，把各页真正的控件推到首屏之外。
# 声明本身必须保留（诚实边界是这个项目的核心），但应以「随时可查」而非「每页挡路」的形式存在。
# 其中「数据说明」这一条优先级最高——下面的规模警示只讲「数据量小」，
# 若读者先看到它，容易误解为「只是规模小、数据本身是真的」，故把它排在最前。
with st.expander("🧪 口径与边界声明（点开查看 · 影响所有页面的结论可信度）", expanded=False):
    st.info(
        "**数据说明**：本应用的全部数据由 `src/data/synthetic_generator.py` **合成生成**"
        "（固定随机种子，可一键复现），**不使用、也不声称使用任何真实业务数据**。\n\n"
        "选用合成数据并非取巧：它让「潜在真实 CTR」成为**已知量**，"
        "从而能构造 **Oracle 上界**、给出**不依赖平滑估计**的评估口径——"
        "这是本项目量化结论得以成立的前提。代价是：本页展示的是"
        "**方法在受控条件下的有效性**，不代表**线上可迁移性**。",
        icon="🧪",
    )

    _SCALE_INFO = {
        "micro": ("极轻档", "1 万用户 × 1 万物品 × 50 万曝光", "约 130MB", "1GB 内存的免费实例"),
        "lite":  ("轻量档", "2 万用户 × 2 万物品 × 100 万曝光", "约 960MB", "2GB 内存的实例"),
    }

    if mode in _SCALE_INFO:
        _name, _scale, _mem, _tier = _SCALE_INFO[mode]
        st.warning(
            f"**本实例为{_name}**：{_scale}（完整档为 10 万 × 10 万 × 500 万），"
            f"实测峰值内存 {_mem}，适配{_tier}，仅用于**交互体验**。"
            "因数据规模远小于完整档，**本页数值不可作为论文级结论**；"
            "可引用结论请以完整档离线复现为准（见 GitHub 仓库 README 与展示页）。",
            icon="⚠️",
        )
        st.caption(
            "ℹ️ 本实例的模型为**离线预训练 + numpy 推理**：训练阶段在本机完成，"
            "权重导出为 numpy 数组后加载，在线侧不依赖 torch/sklearn —— "
            "这是在「免费实例内存受限」条件下的工程取舍，详见仓库 `src/build_online_artifacts.py`。"
        )

# --------------------------------------------------------------------------
if page == "推荐演示":
    st.header("推荐演示")
    st.caption("选择用户与 Top-K，观察完整链路（召回 → 排序 → 混排）的最终推荐结果。")
    c1, c2 = st.columns([1, 1])
    user_id = c1.number_input("用户 ID", 0, n_users_max, 0, 1,
                              help=f"取值 0–{n_users_max}，对应合成数据中的一位用户")
    top_k = c2.slider("Top K", 5, 20, 10)
    if st.button("生成推荐", type="primary"):
        items = engine.recommend(int(user_id), top_k)
        if not items:
            st.warning("该用户无推荐结果")
        else:
            st.subheader(f"用户 {user_id} 的 Top-{top_k} 推荐")

            # 汇总指标：一眼看到结果的整体形态
            _ctr = [it["ctr"] for it in items]
            m1, m2, m3 = st.columns(3)
            m1.metric("平均预估 CTR", f"{sum(_ctr) / len(_ctr) * 100:.2f}%")
            m2.metric("最高预估 CTR", f"{max(_ctr) * 100:.2f}%")
            m3.metric("覆盖品类数", f"{len({it['category'] for it in items})}")

            # 卡片式结果：前三名用「好」色标号，其余用主色
            _cards = []
            for it in items:
                _top = " top" if it["rank"] <= 3 else ""
                _cards.append(
                    f'<div class="rec-card">'
                    f'<div class="rec-rank{_top}">{it["rank"]}</div>'
                    f'<div class="rec-main">'
                    f'<div class="rec-name">{it["name"]}'
                    f'<span class="rec-tag">{it["category"]}</span></div>'
                    f'<div class="rec-meta">预估 CTR {it["ctr"] * 100:.2f}%'
                    f' · 基础分 {it["base_score"]:.3f}</div>'
                    f'</div>'
                    f'<div class="rec-cvr"><b>{it["cvr"] * 100:.2f}%</b>CVR</div>'
                    f'</div>'
                )
            st.markdown("".join(_cards), unsafe_allow_html=True)

            with st.expander("查看完整字段（含内部打分）"):
                st.dataframe(items, use_container_width=True, hide_index=True)
            st.caption(
                "CTR = DeepFM 预估点击率；CVR = LR 预估转化率；基础分 = 召回阶段得分。"
                "结果由 `engine.recommend()` 一次产出，可复现（固定随机种子）。"
            )

# --------------------------------------------------------------------------
elif page == "排序流程":
    st.header("5 步混排流程（可解释）")
    st.caption(
        "拆开看每一步做了什么、中间态长什么样。这是「可解释」的落点——"
        "不只给结果，还给到达结果的路径。"
    )
    _STEPS = [
        ("① 归一化", "把各分数压到同一量纲，避免量纲大的指标主导加权"),
        ("② 多目标加权", "w_base·基础分 + w_ctr·(0.7×CTR + 0.3×CVR)，默认 w_base=0"),
        ("③ 分层排序", "按分数分层，让不同档位交替出现"),
        ("④ 同品类打散", "同品类不连排，避免结果同质化"),
        ("⑤ 去重截断", "剔除重复项，截断到 Top-K"),
    ]
    st.markdown(
        '<div class="step-flow">'
        + '<span class="step-arrow">→</span>'.join(
            f'<span class="step-chip on">{s}</span>' for s, _ in _STEPS)
        + '</div>',
        unsafe_allow_html=True,
    )
    user_id = st.number_input("用户 ID", 0, n_users_max, 0, 1)
    if st.button("展示流程", type="primary"):
        tr = engine.explain_steps(int(user_id), show=10)
        st.caption(
            f"候选池共 **{tr['candidate_count']}** 个物品；下表逐步展示中间态的变化。"
            "默认 `w_base=0`——候选池已由热门 + 规则召回构成，热度信息已体现在候选集里，"
            "混排阶段不再二次注入热门偏差（见 `src/rank/mixer.py` 实测说明）。"
        )

        _stage_keys = [
            ("candidates", "候选池（召回后）"),
            ("step1_normalized", "① 归一化后"),
            ("step2_weighted", "② 多目标加权后"),
            ("step3_stratified", "③ 分层排序后"),
            ("step4_diversified", "④ 同品类打散后"),
            ("final", "⑤ 去重截断 → 最终结果"),
        ]
        _stage_tabs = st.tabs([lbl for _, lbl in _stage_keys])
        for _tab, (key, _lbl) in zip(_stage_tabs, _stage_keys):
            with _tab:
                rows = tr.get(key) or []
                if not rows:
                    st.write("（该阶段无数据）")
                    continue
                st.dataframe({
                    "物品": [r["name"] for r in rows],
                    "类目": [r["category"] for r in rows],
                    "基础分": [r["base_score"] for r in rows],
                    "预估 CTR": [r["ctr"] for r in rows],
                    "预估 CVR": [r["cvr"] for r in rows],
                }, use_container_width=True, hide_index=True)

        st.subheader("最终结果的排序（按顺序）")
        _fin = tr.get("final") or []
        if _fin:
            st.markdown(
                '<div class="step-flow">'
                + '<span class="step-arrow">→</span>'.join(
                    f'<span class="step-chip">{r["name"]}</span>' for r in _fin)
                + '</div>',
                unsafe_allow_html=True,
            )
            st.caption("经打散后同品类不再连排；尾部重复项已在第 ⑤ 步截断。")

# --------------------------------------------------------------------------
elif page == "真实 CTR 对比":
    st.header("真实 CTR 口径对比（结论可引用）")
    st.info(
        "**为什么这一页最可信**：直接用**已知的潜在真实 CTR** 计算各方法 Top-K 的平均点击率，"
        "不依赖任何「平滑历史 CTR」估计，因此**不受稀疏性偏差影响**。\n\n"
        "**Oracle 上界** = 逐用户的候选池（热门 ∪ 规则 ∪ 随机）内按真实 CTR 取 Top-K，"
        "因此它**严格 ≥ 所有被比较的方法**——这是一个有意义的天花板。",
        icon="✅",
    )
    c1, c2 = st.columns(2)
    n_eval = c1.slider("样本用户数", 20, 200, 100, 20,
                       help="越大越稳，但耗时线性增长")
    top_k_eval = c2.slider("Top K", 5, 20, 10, 1, key="ctr_topk")
    if st.button("运行真实 CTR 对比", type="primary"):
        with st.spinner("评估中 ..."):
            r = engine.benchmark_rankers(n_users=n_eval, top_k=top_k_eval)
        pm = r["per_method"]
        base = r["base_method"]
        dfm = pm.get("DeepFM+混排", {})

        # 结论先行：三个关键数字放最上面
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("DeepFM 相对热门", f"{dfm.get('rel_to_base', 0):+.1%}",
                  help=f"以「{base}」为基线")
        m2.metric("占 Oracle 上界", f"{dfm.get('pct_of_oracle', 0):.1%}",
                  help="1.0 = 达到理论上界")
        m3.metric("Oracle 上界真实 CTR", f"{pm.get('Oracle', {}).get('true_ctr', 0):.4f}")
        m4.metric("样本量", f"{r['n_users']} 用户 × Top-{r['top_k']}")

        _order = ["随机", base, "DeepFM+混排", "Oracle"]
        rows = []
        for name in _order:
            if name not in pm:
                continue
            d = pm[name]
            rows.append({
                "方法": name,
                "平均真实 CTR": round(d["true_ctr"], 4),
                f"相对{base}": ("—" if name == base else f"{d['rel_to_base']:+.1%}"),
                "占 Oracle 上界": f"{d['pct_of_oracle']:.1%}",
            })
        st.subheader("各方法对比")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        _names = [n for n in _order if n in pm]
        st.subheader("平均真实 CTR（越高越好）")
        st.bar_chart({"方法": _names, "真实 CTR": [pm[n]["true_ctr"] for n in _names]},
                     x="方法", y="真实 CTR", color="#2563eb", height=320)

        st.success(
            f"**结论**：{r['n_users']} 用户、Top-{r['top_k']} 下，DeepFM+混排 相对 {base} "
            f"提升 **{dfm.get('rel_to_base', 0):+.1%}**，达到理论上界的 "
            f"**{dfm.get('pct_of_oracle', 0):.1%}**。"
            "（完整档 10万×10万×500万 的离线结果为 +28.5% / 88.4%，在线 micro 档因候选池更稀疏，"
            "增益低于完整档——这是规模差异，不是实现差异。）"
        )

# --------------------------------------------------------------------------
elif page == "模拟 A/B":
    st.header("模拟 A/B 报告")
    st.caption("对照组 = 热门排序；实验组 = DeepFM+混排。各自独立曝光并观测点击 / 转化（离线模拟）。")
    st.warning(
        "**本页口径受限，数值仅供参考。** 对照的热门基线按「历史日志平滑 CTR」排序，"
        "而日志为随机曝光——曝光少的物品偶然被点击就会推高平滑 CTR，形成**稀疏性偏差**，"
        "使实验组增益被**高估**。请以「真实 CTR 对比」页的结论为准。",
        icon="⚠️",
    )
    if mode == "lite":
        st.error(
            "**轻量档的 A/B 数值尤其不可引用。** 轻量档物品更少、更稀疏，"
            "该偏差被进一步放大，甚至可能出现负提升。",
            icon="🚫",
        )
    n_users = st.slider("模拟用户数", 200, 2000, 800, 100,
                        help="越大越稳，但耗时线性增长")
    if st.button("运行 A/B 模拟", type="primary"):
        with st.spinner("模拟中 ..."):
            r = engine.run_ab(n_users=n_users)

        st.subheader("实验结果")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("实验组 CTR", f"{r['trt_ctr'] * 100:.2f}%",
                  delta=f"{r['ctr_lift'] * 100:+.1f}% vs 对照")
        c2.metric("对照组 CTR", f"{r['ctrl_ctr'] * 100:.2f}%")
        c3.metric("实验组 CVR", f"{r['trt_cvr'] * 100:.2f}%",
                  delta=f"{r['cvr_lift'] * 100:+.1f}% vs 对照")
        c4.metric("CTR 显著性 p", f"{r['ctr_p']:.4f}",
                  help="两比例 z 检验；p < 0.05 通常视为显著")

        st.subheader("实验组 vs 对照组")
        # 配色说明：这里用主色（实验组）与中性灰（对照组）区分，而非红/绿。
        # 原因：本项目已把绿/红固定为**质量语义**（绿=好、红=差，见 data/bias/report.html）。
        # A/B 的「实验组更高」不等于「质量更好」（本页数值本身已被声明为不可引用），
        # 若在此处用绿色标实验组，会与偏差报告里的语义约定冲突。
        st.bar_chart({
            "指标": ["点击率 CTR", "转化率 CVR"],
            "对照组（热门）": [round(r["ctrl_ctr"], 4), round(r["ctrl_cvr"], 4)],
            "实验组（DeepFM+混排）": [round(r["trt_ctr"], 4), round(r["trt_cvr"], 4)],
        }, x="指标", color=["#94a3b8", "#2563eb"], height=320)
        st.caption(
            f"CVR 显著性 p={r['cvr_p']:.4f}；点击后转化率 对照 {r['ctrl_pcvr']:.2%} / "
            f"实验 {r['trt_pcvr']:.2%}（合成数据，非真实线上指标）"
        )

        with st.expander("📌 关于本页数值的可信度说明（为什么可能出现负提升）"):
            st.markdown(
                "**为什么这里可能出现负提升？**\n\n"
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
                "**结论**：本页数字请勿引用；请以「真实 CTR 对比」页为准。"
            )

# --------------------------------------------------------------------------
# 页脚：把「这个 Demo 是什么、还能看什么」收在一处，便于访客继续深入。
st.markdown(
    '<div class="foot">'
    '<b>AeroRec</b> · 出行智能推荐中台 · 本科 AI 申请作品<br>'
    '本实例运行 micro 档（1 万用户 × 1 万物品 × 50 万曝光），数据由脚本合成、可复现；'
    '模型为离线预训练 + numpy 推理，在线侧不依赖 torch / sklearn。<br>'
    '<a href="https://github.com/Elijah-Lin-Dancer/AeroRec">完整仓库</a> · '
    '<a href="https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/docs/bias_report.md">偏差诊断 tech report</a> · '
    '<a href="https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/data/bias/report.html">可视化报告</a>'
    '</div>',
    unsafe_allow_html=True,
)
