"""AeroRec 出行推荐中台 · 交互式 Demo（Streamlit）。

运行：streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src.serve.engine import RecommendationEngine


@st.cache_resource(show_spinner="首次加载需训练模型（约 30 秒）...")
def get_engine():
    return RecommendationEngine()


st.set_page_config(page_title="AeroRec 出行推荐中台", layout="wide")
st.title("AeroRec · 出行智能推荐中台")
st.caption("召回 → DeepFM CTR 排序 → 5 步混排 → 降级兜底 → 模拟 A/B（合成数据演示）")

engine = get_engine()
page = st.sidebar.radio("导航", ["推荐演示", "排序流程", "A/B 报告"])

if page == "推荐演示":
    st.header("推荐演示")
    c1, c2 = st.columns(2)
    user_id = c1.number_input("用户 ID", 0, 99999, 0, 1)
    top_k = c2.slider("Top K", 5, 20, 10)
    if st.button("生成推荐"):
        items = engine.recommend(int(user_id), top_k)
        if not items:
            st.warning("该用户无推荐结果")
        else:
            st.subheader(f"用户 {user_id} 的 Top{top_k} 推荐")
            st.dataframe(pd.DataFrame(items), use_container_width=True)
            for it in items:
                st.markdown(
                    f"**#{it['rank']}  {it['name']}** · {it['category']} · "
                    f"预估 CTR {it['ctr'] * 100:.1f}% · 基础分 {it['base_score']:.3f}"
                )

elif page == "排序流程":
    st.header("5 步混排流程（可解释）")
    user_id = st.number_input("用户 ID", 0, 99999, 0, 1)
    if st.button("展示流程"):
        tr = engine.explain_steps(int(user_id), show=15)
        st.subheader("① 归一化 ② 多目标加权（0.4×基础分 + 0.6×(0.7×CTR+0.3×CVR)）")
        st.dataframe(pd.DataFrame({
            "物品": tr["names"],
            "类目": tr["categories"],
            "基础分": tr["base"],
            "预估CTR": tr["ctr"],
            "预估CVR": tr["cvr"],
            "加权分": tr["combined"],
        }), use_container_width=True)
        st.subheader("③ 分层排序 → ④ 同品类打散 → ⑤ 去重截断")
        final_names = [engine.name_map[i] for i in tr["final_ids"]]
        st.write("  →  ".join(final_names))

elif page == "A/B 报告":
    st.header("模拟 A/B 报告")
    st.caption("对照组=热门排序，实验组=DeepFM+多目标混排；各自独立曝光并观测点击/转化（离线模拟）")
    n_users = st.slider("模拟用户数", 200, 5000, 2000, 200)
    if st.button("运行 A/B 模拟"):
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
