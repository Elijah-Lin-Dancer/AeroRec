"""AeroRec 出行推荐中台 · 交互式 Demo（Streamlit）。

运行：streamlit run app.py

两种模式（通过环境变量 AEROREC_MODE 切换）
------------------------------------------
  lite  （默认）：2 万用户 × 2 万物品 × 100 万曝光，峰值约 900MB，
                  适配 2GB 内存的在线实例（如 Hugging Face Spaces）。
  full         ：10 万用户 × 10 万物品 × 500 万曝光，需较大内存与较长加载时间，
                  用于本地完整复现。

在线实例默认 lite，并在页面上**明确标注规模与 A/B 口径差异**，
避免读者把 Demo 的交互数值误当成论文级结论。
"""
import os

import pandas as pd
import streamlit as st

MODE = os.environ.get("AEROREC_MODE", "lite").lower()

st.set_page_config(page_title="AeroRec 出行推荐中台", layout="wide")


@st.cache_resource(show_spinner="首次加载需生成数据并训练模型（约 10 秒）...")
def get_engine():
    if MODE == "full":
        from src.serve.engine import RecommendationEngine
        return RecommendationEngine(), "full"
    from src.serve.engine_lite import LiteRecommendationEngine
    return LiteRecommendationEngine(), "lite"


engine, mode = get_engine()

st.title("AeroRec · 出行智能推荐中台")
st.caption("召回 → DeepFM CTR 排序 → 5 步混排 → 降级兜底 → 模拟 A/B（合成数据演示）")

if mode == "lite":
    st.warning(
        "**本实例为轻量档**：2 万用户 × 2 万物品 × 100 万曝光（完整档为 10 万 × 10 万 × 500 万），"
        "仅用于交互体验。因数据更稀疏，**此处 A/B 数值不可引用**；"
        "量化结论请以完整档离线复现为准（见 GitHub 仓库 README 与展示页）。",
        icon="⚠️",
    )

page = st.sidebar.radio("导航", ["推荐演示", "排序流程", "A/B 报告"])
st.sidebar.caption(f"当前模式：**{mode}**")
n_users_max = 19_999 if mode == "lite" else 99_999

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
            st.dataframe(pd.DataFrame(items), use_container_width=True)
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
        st.subheader("① 归一化 ② 多目标加权（0.4×基础分 + 0.6×(0.7×CTR + 0.3×CVR)）")
        st.dataframe(pd.DataFrame({
            "物品": tr["names"],
            "类目": tr["categories"],
            "基础分": tr["base"],
            "预估CTR": tr["ctr"],
            "预估CVR": tr["cvr"],
            "加权分": tr["combined"],
        }), use_container_width=True)
        st.subheader("③ 分层排序 → ④ 同品类打散 → ⑤ 去重截断")
        st.write("  →  ".join(tr["names"][:15]))

# --------------------------------------------------------------------------
elif page == "A/B 报告":
    st.header("模拟 A/B 报告")
    st.caption("对照组 = 热门排序；实验组 = DeepFM+混排。各自独立曝光并观测点击/转化（离线模拟）")
    if mode == "lite":
        st.error(
            "**轻量档的 A/B 数值不可引用。** 热门基线用「历史日志平滑 CTR」排序，"
            "而日志为随机曝光——曝光次数少的物品偶然被点击就会推高平滑 CTR，"
            "形成稀疏性偏差（完整档实测该偏差约 +0.37）。轻量档物品更少、更稀疏，"
            "该偏差被放大，甚至出现负提升。请以完整档离线复现结论为准。",
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
        chart = pd.DataFrame({
            "指标": ["点击率 CTR", "转化率 CVR"],
            "对照组(热门)": [round(r["ctrl_ctr"], 4), round(r["ctrl_cvr"], 4)],
            "实验组(DeepFM+混排)": [round(r["trt_ctr"], 4), round(r["trt_cvr"], 4)],
        }).set_index("指标")
        st.bar_chart(chart)
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
                "**这意味着**：用「热门基线」作为对照，会**高估**推荐系统的相对收益。"
                "严谨的做法是同时报告一条**上界基线**（按真实 CTR 排序的 Oracle），"
                "给出「本方法逼近上界多少」的信息。本项目正在补齐这一项。\n\n"
                "**结论**：请勿引用轻量档或未加偏差说明的 A/B 数值。"
            )
