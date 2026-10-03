"""AeroRec 出行推荐中台 · 交互式 Demo（Streamlit）。

运行：streamlit run app.py

三种模式（通过环境变量 AEROREC_MODE 切换）
------------------------------------------
  micro（默认）：1 万用户 × 1 万物品 × 50 万曝光，纯 numpy 推理、**不加载 torch**，
                  实测峰值内存约 130MB，适配 1GB 内存的免费在线实例
                  （Streamlit Community Cloud）。模型权重离线预训练后加载，
                  故启动仅需约 1 秒。
  lite         ：2 万用户 × 2 万物品 × 100 万曝光，内存约 930MB（在线训练），
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

st.set_page_config(page_title="AeroRec 出行推荐中台", layout="wide")


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

st.title("AeroRec · 出行智能推荐中台")
st.caption("召回 → DeepFM CTR 排序 → 5 步混排 → 降级兜底 → 真实 CTR 评估（合成数据演示）")

_SCALE_INFO = {
    "micro": ("极轻档", "1 万用户 × 1 万物品 × 50 万曝光", "约 130MB", "1GB 内存的免费实例"),
    "lite":  ("轻量档", "2 万用户 × 2 万物品 × 100 万曝光", "约 930MB", "2GB 内存的实例"),
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

page = st.sidebar.radio("导航", ["推荐演示", "排序流程", "真实 CTR 对比", "模拟 A/B"])
st.sidebar.caption(f"当前模式：**{mode}**")
n_users_max = {"micro": 9_999, "lite": 19_999}.get(mode, 99_999)

# --------------------------------------------------------------------------
if page == "推荐演示":
    st.header("推荐演示")
    c1, c2 = st.columns(2)
    user_id = c1.number_input("用户 ID", 0, n_users_max, 0, 1)
    top_k = c2.slider("Top K", 5, 20, 10)
    if st.button("生成推荐", type="primary"):
        items = engine.recommend(int(user_id), top_k)
        if not items:
            st.warning("该用户无推荐结果")
        else:
            st.subheader(f"用户 {user_id} 的 Top{top_k} 推荐")
            st.dataframe(items, use_container_width=True)
            for it in items:
                st.markdown(
                    f"**#{it['rank']}  {it['name']}** · {it['category']} · "
                    f"预估 CTR {it['ctr'] * 100:.1f}% · 预估 CVR {it['cvr'] * 100:.1f}% · "
                    f"基础分 {it['base_score']:.3f}"
                )

# --------------------------------------------------------------------------
elif page == "排序流程":
    st.header("5 步混排流程（可解释）")
    user_id = st.number_input("用户 ID", 0, n_users_max, 0, 1)
    if st.button("展示流程", type="primary"):
        tr = engine.explain_steps(int(user_id), show=15)
        st.subheader("① 归一化 ② 多目标加权（w_base·基础分 + w_ctr·(0.7×CTR + 0.3×CVR)，默认 w_base=0）")
        st.caption(
            "默认 `w_base=0`：候选池已由热门+规则召回构成，热度信息已体现在候选集里，"
            "混排阶段不再二次注入热门偏差（见 `src/rank/mixer.py` 实测说明）。"
        )
        st.dataframe({
            "物品": tr["names"],
            "类目": tr["categories"],
            "基础分": tr["base"],
            "预估CTR": tr["ctr"],
            "预估CVR": tr["cvr"],
            "加权分": tr["combined"],
        }, use_container_width=True)
        st.subheader("③ 分层排序 → ④ 同品类打散 → ⑤ 去重截断")
        st.write("  →  ".join(tr["names"][:15]))

# --------------------------------------------------------------------------
elif page == "真实 CTR 对比":
    st.header("真实 CTR 口径对比（可引用）")
    st.caption(
        "直接用已知的潜在真实 CTR，计算每个方法 Top-K 列表的平均点击率。"
        "不依赖任何「平滑历史 CTR」估计，因此**不受稀疏性偏差影响**。"
        "Oracle 上界 = 逐用户候选池（热门∪规则∪随机）内按真实 CTR 取 Top-K。"
    )
    n_eval = st.slider("样本用户数", 20, 200, 100, 20)
    top_k_eval = st.slider("Top K", 5, 20, 10, 1, key="ctr_topk")
    if st.button("运行真实 CTR 对比", type="primary"):
        with st.spinner("评估中 ..."):
            r = engine.benchmark_rankers(n_users=n_eval, top_k=top_k_eval)
        pm = r["per_method"]
        base = r["base_method"]

        rows = []
        for name in ["随机", base, "DeepFM+混排", "Oracle"]:
            if name not in pm:
                continue
            d = pm[name]
            rows.append({
                "方法": name,
                "平均真实CTR": round(d["true_ctr"], 4),
                f"相对{base}": ("—" if name == base
                              else f"{d['rel_to_base']:+.1%}"),
                "占 Oracle 上界": f"{d['pct_of_oracle']:.1%}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        dfm = pm.get("DeepFM+混排", {})
        c1.metric("DeepFM 相对热门", f"{dfm.get('rel_to_base', 0):+.1%}")
        c2.metric("DeepFM 占上界", f"{dfm.get('pct_of_oracle', 0):.1%}")
        c3.metric("Oracle 上界真实CTR", f"{pm.get('Oracle', {}).get('true_ctr', 0):.4f}")

        chart = {
            "方法": [n for n in ["随机", base, "DeepFM+混排", "Oracle"] if n in pm],
            "真实CTR": [pm[n]["true_ctr"] for n in ["随机", base, "DeepFM+混排", "Oracle"]
                       if n in pm],
        }
        st.bar_chart(chart, x="方法", y="真实CTR")

        st.info(
            f"**这是本 Demo 中最可信的一组数字。** 样本 {r['n_users']} 用户、Top-{r['top_k']}；"
            f"DeepFM+混排 相对 {base} 基线提升 **{dfm.get('rel_to_base', 0):+.1%}**，"
            f"达到理论上界的 **{dfm.get('pct_of_oracle', 0):.1%}**。"
        )

# --------------------------------------------------------------------------
elif page == "模拟 A/B":
    st.header("模拟 A/B 报告")
    st.caption("对照组 = 热门排序；实验组 = DeepFM+混排。各自独立曝光并观测点击/转化（离线模拟）")
    st.warning(
        "**本页口径受限，数值仅供参考。** 对照的热门基线按「历史日志平滑 CTR」排序，"
        "而日志为随机曝光——曝光少的物品偶然被点击就会推高平滑 CTR，形成稀疏性偏差，"
        "使实验组增益被**高估**。请以「真实 CTR 对比」页的结论为准。",
        icon="⚠️",
    )
    if mode == "lite":
        st.error(
            "**轻量档的 A/B 数值尤其不可引用。** 轻量档物品更少、更稀疏，"
            "该偏差被进一步放大，甚至可能出现负提升。",
            icon="🚫",
        )
    n_users = st.slider("模拟用户数", 200, 2000, 800, 100)
    if st.button("运行 A/B 模拟", type="primary"):
        with st.spinner("模拟中 ..."):
            r = engine.run_ab(n_users=n_users)
        c1, c2 = st.columns(2)
        c1.metric("点击率 CTR", f"{r['trt_ctr'] * 100:.2f}%",
                  delta=f"{r['ctr_lift'] * 100:+.1f}% vs 对照")
        c2.metric("转化率（转化/曝光）", f"{r['trt_cvr'] * 100:.2f}%",
                  delta=f"{r['cvr_lift'] * 100:+.1f}% vs 对照")
        chart = {
            "指标": ["点击率 CTR", "转化率 CVR"],
            "对照组(热门)": [round(r["ctrl_ctr"], 4), round(r["ctrl_cvr"], 4)],
            "实验组(DeepFM+混排)": [round(r["trt_ctr"], 4), round(r["trt_cvr"], 4)],
        }
        st.bar_chart(chart, x="指标")
        st.caption(f"CTR 显著性 p={r['ctr_p']:.4f}，CVR 显著性 p={r['cvr_p']:.4f}"
                   f"；点击后转化率 对照 {r['ctrl_pcvr']:.2%} / 实验 {r['trt_pcvr']:.2%}"
                   "（合成数据，非真实线上指标）")

        with st.expander("📌 关于本页数值的可信度说明"):
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
