"""AeroRec 出行推荐中台 · 交互式 Demo（Streamlit，中英双语 / bilingual）。

运行：streamlit run app.py

语言
----
侧边栏顶部可切换 **English / 中文**（默认 English，面向海外 reviewer）。
所有面向用户的文案集中在 `src/serve/i18n.py`；语言存于 `st.session_state`，
翻译函数由 `make_t(lang)` 生成，**不使用模块级可变状态**（避免多会话串语言）。

三种模式（通过环境变量 AEROREC_MODE 切换）
------------------------------------------
  micro（默认）：1 万用户 × 1 万物品 × 50 万曝光，纯 numpy 推理、**不加载 torch**，
                  实测进程峰值内存约 297MB，适配免费在线实例
                  （Streamlit Community Cloud，内存下限 690MB）。模型权重离线
                  预训练后加载，故启动仅需约 1 秒。
  lite         ：2 万用户 × 2 万物品 × 100 万曝光，内存约 960MB（在线训练），
                  适配 2GB 内存的实例。
  full         ：10 万用户 × 10 万物品 × 500 万曝光，需较大内存与较长加载时间，
                  用于本地完整复现。

为什么默认是 micro（工程取舍，值得说明）
------------------------------------------
免费实例的资源是**区间不是固定值**（Streamlit Community Cloud 官方，2024-02 起）：
CPU 0.078～2 核、内存 690MB～2.7GB、存储 ≤50GB、无 GPU、无流量 12 小时休眠。
本项目取**下限 690MB** 作设计基准（最坏情况）。

在该约束下量两种运行方式的**框架基线**（同进程加载，未含数据）：

    torch + sklearn + numpy + streamlit（解耦前）= 673 MB
    numpy + streamlit                （解耦后）=  73 MB

673MB 的基线在 690MB 下限上只剩 16.7MB —— **连数据都装不下**。
即 `import torch` 一项就吃掉约 500–650MB（随 wheel 与版本浮动），与数据规模无关。

因此本项目把训练与推理**解耦**：离线（本机）用 torch 训练 DeepFM / LR，
把权重导出为 numpy（`python -m src.build_online_artifacts micro`），
在线侧只加载权重、用纯 numpy 复现前向计算。

    解耦前：框架基线 673MB，超出 690MB 下限 → 必然 OOM
    解耦后：实测峰值 297MB（79MB 空转 → 187MB 加载完 → 297MB 跑完最重操作），
            启动 ~1s，相对 690MB 下限余量 2.3×

数学等价性已在构建期逐样本校验（PyTorch vs numpy 最大偏差 1.19e-07）。
"""
import json
import os

import numpy as np
import streamlit as st

# ⚠️ 这里**刻意不在模块顶层 `import pandas`**：
# 在线 micro 档走纯 numpy 路径，pandas 的固有导入开销约 150MB，
# 而免费实例内存下限为 690MB（详见本文件顶部「工程取舍」说明）。
# `st.dataframe` / `st.bar_chart` 均可直接吃 dict 与 list[dict]，无需 pandas。
# 仅当需要 pandas 专有能力时，才在函数内部惰性导入。
from src.serve.i18n import (
    DEFAULT_LANG,
    LANG_LABELS,
    LANGS,
    localize_category,
    localize_item_name,
    make_t,
    method_label,
)

MODE = os.environ.get("AEROREC_MODE", "micro").lower()

st.set_page_config(page_title="AeroRec · Travel Recommender Platform",
                   page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------
# 视觉层：统一注入一段 CSS，把默认的「未排版 Markdown」观感收敛成有层次的版式。
#
# 为什么用 CSS 注入而不是换组件库：
#   1. 在线实例内存下限 690MB，引入第三方 UI 组件（如 streamlit-extras 等）
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


@st.cache_resource(show_spinner="Loading model & data (~a few s) · 首次加载需约数秒 ...")
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

# --------------------------------------------------------------------------
# 研究主线交互页所需的纯 numpy 辅助（不引入 torch / sklearn / pandas）
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Computing bias-propagation sweep · 计算偏差传播扫描 ...")
def _bias_full_sweep(n_items: int = 60, n_requests: int = 600, slots: int = 5,
                     alpha: float = 5.0, seed: int = 42):
    """对 τ∈[0,3] 逐步算「偏差传播」，缓存为一条交互扫描。

    与 `scripts/run_bias_experiments.py` 同口径：真实 CTR 由对数正态合成
    （seed=0，与可视化报告一致），曝光策略 π ∝ 热度^τ，平滑 CTR α=5。
    返回每 τ 的 L1/L2/L3 spearman、rel_err、曝光 Gini、最少曝光，
    以及当前 τ 的平滑 CTR 向量与曝光频率（供散点 / 柱状图用）。
    """
    from src.bias.metrics import exposure_gini
    from src.bias.propagation import run_stage_estimates, stage_error

    def _synth_true_ctr(n=60, s=0):
        r = np.random.default_rng(s)
        base = r.lognormal(mean=np.log(0.22), sigma=0.5, size=n)
        return np.clip(base, 0.01, 0.9)

    tctr = _synth_true_ctr(n_items, 0)                 # (C,)
    taus = [round(float(x), 1) for x in np.arange(0.0, 3.0001, 0.1)]
    rows = []
    for tau in taus:
        # CRN（与 `propagation_matrix` 同策略）：每 τ 都以 default_rng(seed) 重置，
        # 保证 demo 与报告 propagation.json 逐点一致，且与扫描顺序无关。
        rng = np.random.default_rng(seed)
        r = run_stage_estimates(tctr[None, :], n_items, float(tau), n_requests,
                                slots, alpha, 0.0, rng)

        def _stage(est):
            """逐级误差；对**常数估计**把 spearman 约定为 0。

            为什么需要这条约定：τ=0 时 L1 的估计 `tmean·C·π` 是常数向量，
            而 `_rank` 用稳定排序，会把「全等值」排成 0..N−1（即按物品下标），
            与真值秩相关后得到一个**纯伪相关**（60 物品、单种子下约 +0.18；
            落库的 500 物品/30 种子版只有 −0.0197，所以报告与论文不受影响）。
            常数序列本就不携带排序信息，约定为 0 才符合语义。
            注意：只改交互扫描的展示，不动 `src/bias/propagation.py` 的统计口径，
            因此 `data/bias/*.json` 与论文数字保持逐字节不变。
            """
            d = stage_error(est, r["true_ctr"])
            if np.asarray(est, dtype=np.float64).std() < 1e-12:
                d["spearman"] = 0.0
            return d

        e1 = _stage(r["L1_recall"])
        e2 = _stage(r["L2_rank"])
        e3 = _stage(r["L3_mix"])
        rows.append({
            "tau": float(tau),
            "L1_spearman": e1["spearman"], "L2_spearman": e2["spearman"],
            "L3_spearman": e3["spearman"],
            "L1_rel_err": e1["rel_err"], "L2_rel_err": e2["rel_err"],
            "L3_rel_err": e3["rel_err"],
            "exposure_gini": float(exposure_gini(r["exposure_freq"])),
            "impr_min": int(r["impressions"].min()),
            "exposure_freq": np.asarray(r["exposure_freq"], dtype=np.float64).copy(),
            "L2_est": np.asarray(r["L2_rank"], dtype=np.float64).copy(),
        })
    return {"taus": taus, "rows": rows, "true_ctr": np.asarray(tctr, dtype=np.float64)}


def _bias_row_at(sweep: dict, tau: float) -> dict:
    return min(sweep["rows"], key=lambda r: abs(r["tau"] - tau))


@st.cache_data
def _load_json(rel_path: str):
    from pathlib import Path
    return json.loads((Path(__file__).resolve().parent / rel_path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 语言选择（放在侧边栏最上方；切换会触发 rerun，后续文案随之重算）
# --------------------------------------------------------------------------
# 语言来源优先级：URL ?lang= > 环境变量 AEROREC_LANG > 默认（en）。
# 回写 URL 后可直接分享「英文界面」链接，落地即英文，无需再手动切一次。
_env_lang = os.environ.get("AEROREC_LANG", DEFAULT_LANG).lower()
if _env_lang not in LANGS:
    _env_lang = DEFAULT_LANG

# 读取 URL query（query API 在 streamlit>=1.30 才有；此处容错处理，缺失时退化为环境变量）
try:
    _qp_lang = st.query_params.get("lang", "")
    if isinstance(_qp_lang, list):          # 兼容旧版把重复参数读成 list 的行为
        _qp_lang = _qp_lang[0] if _qp_lang else ""
    _qp_lang = str(_qp_lang or "").lower()
    _has_query_api = True
except Exception:
    _qp_lang, _has_query_api = "", False

# 只在「URL 与我们上次写回的值不同」时才采纳 URL —— 否则会把用户刚在
# 侧边栏选好的语言回滚成上一轮的 URL 旧值（URL 回写比交互晚一轮）。
_synced = st.session_state.get("_lang_synced")
if _qp_lang in LANGS and _qp_lang != _synced:
    st.session_state["lang"] = _qp_lang          # URL 显式指定 → 采纳
elif "lang" not in st.session_state:
    st.session_state["lang"] = _env_lang         # 首帧无 URL → 用环境变量/默认

lang = st.sidebar.selectbox(
    "🌐 Language / 语言", list(LANGS),
    format_func=lambda c: LANG_LABELS[c],
    key="lang",
)

# 把当前语言回写 URL（设置 query_params 不会触发 rerun），便于分享/锚定语言。
if _has_query_api and st.query_params.get("lang") != lang:
    try:
        st.query_params["lang"] = lang
    except Exception:
        pass
st.session_state["_lang_synced"] = lang
t = make_t(lang)

st.markdown(
    '<div class="hero">'
    f'<h1>{t("hero.title")}</h1>'
    f'<p class="sub">{t("hero.sub")}</p>'
    '<div class="chain">'
    f'<span>{t("chip.recall")}</span><span>{t("chip.rank")}</span><span>{t("chip.mix")}</span>'
    f'<span>{t("chip.degrade")}</span><span>{t("chip.true_ctr")}</span>'
    '</div></div>',
    unsafe_allow_html=True,
)
# 预印本条目放在页头正下方而不是页脚：这是全站可信度最高的一条信号，
# 首屏可见决定访客是否继续往下看。
st.markdown(t("hero.paper"))

st.sidebar.markdown(
    '<div class="brand">'
    '<div class="t">🧭 AeroRec</div>'
    f'<div class="d">{t("brand.sub")}</div>'
    '</div>',
    unsafe_allow_html=True,
)
# Overview 排第一并作为默认页：多数访客不读论文、也不翻仓库，
# 需要先给「是什么 / 做了什么 / 关键结论 / 有何意义」，再让他自己决定深入哪一块。
page = st.sidebar.radio(
    t("nav.label"),
    ["overview", "demo", "bias", "pipeline", "true_ctr", "ab", "model"],
    format_func=lambda k: t(f"nav.{k}"),
)
st.sidebar.caption(t("sidebar.mode", mode=mode))

# 研究主线的入口。
# 这里只放链接、不嵌内容：`data/bias/report.html` 是带独立版式的完整文档，
# 用 st.components.v1.html 嵌进本页会出现「iframe 高度写死 → 只能在小框里滚」
# 与「两套视觉语言打架」两个问题，且 Streamlit 流式渲染的 iframe 沙箱机制下再叠嵌 iframe 有额外风险。
# 报告本质是「读」的成果，GitHub 上的渲染体验更好，故以链接形式给出。
st.sidebar.divider()
st.sidebar.markdown(
    t("sidebar.paper_title") + "\n\n"
    f"- [{t('sidebar.link.paper_doi')}](https://doi.org/10.20944/preprints202610.2582.v1)\n"
    f"- [{t('sidebar.link.paper_pdf')}]"
    "(https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/docs/paper/AeroRec-preprints.org-v1.pdf)"
)
st.sidebar.markdown(
    t("sidebar.research_title") + "\n\n"
    f"- [{t('sidebar.link.tech_report')}](https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/docs/bias_report.md)\n"
    f"- [{t('sidebar.link.html_report')}](https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/data/bias/report.html)\n"
    f"- [{t('sidebar.link.highlight')}](https://github.com/Elijah-Lin-Dancer/AeroRec/blob/main/docs/research_highlight.md)\n"
    f"- [{t('sidebar.link.repo')}](https://github.com/Elijah-Lin-Dancer/AeroRec)"
)
n_users_max = {"micro": 9_999, "lite": 19_999}.get(mode, 99_999)

# 口径声明放在导航之后、且默认折叠：
# 三块横幅若无条件铺在每页顶部，会占掉约 600px、把各页真正的控件推到首屏之外。
# 声明本身必须保留（诚实边界是这个项目的核心），但应以「随时可查」而非「每页挡路」的形式存在。
# 其中「数据说明」这一条优先级最高——下面的规模警示只讲「数据量小」，
# 若读者先看到它，容易误解为「只是规模小、数据本身是真的」，故把它排在最前。
with st.expander(t("scope.title"), expanded=False):
    st.info(t("scope.data_info"), icon="🧪")

    if mode in ("micro", "lite"):
        st.warning(t("scope.scale_micro") if mode == "micro" else t("scope.scale_lite"),
                   icon="⚠️")
        st.caption(t("scope.inference_note"))

# --------------------------------------------------------------------------
if page == "overview":
    st.header(t("ov.header"))
    st.markdown(t("ov.lead"))

    # 四个数字全部取自仓库内可核验的产物，不在页面里手打：
    #   · ESS/N 边界  ← data/bias/boundary.json
    #   · Criteo AUC  ← data/real_validation/result.json
    #   · +28.5% / 88.4% 来自完整档 `python -m src.eval.evaluate` 的落库结论（README 同源），
    #     微实例无法重算，只能引用——故在 help 里写明口径与档位。
    _bnd = _load_json("data/bias/boundary.json")
    _crit = _load_json("data/real_validation/result.json")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t("ov.m_gain"), "+28.5%", help=t("ov.m_gain_help"))
    m2.metric(t("ov.m_oracle"), "88.4%", help=t("ov.m_oracle_help"))
    m3.metric(t("ov.m_ess"), f"{_bnd['cross_ess_ratio']:.3f}", help=t("ov.m_ess_help"))
    m4.metric(t("ov.m_real"), f"{_crit['val_auc_deepfm']:.4f}", help=t("ov.m_real_help"))

    # ---- 做了什么：2×2 卡片 ------------------------------------------------
    st.subheader(t("ov.what_title"))
    # 变量名加 ov_ 前缀：app.py 是模块级脚本，所有页面共享一个作用域，
    # 与 demo 页的 `_cards`（list[str]）同名会让 mypy 沿用首个推断类型而报错。
    _ov_cards = [
        ("ov.c1_title", "ov.c1_body"),
        ("ov.c2_title", "ov.c2_body"),
        ("ov.c3_title", "ov.c3_body"),
        ("ov.c4_title", "ov.c4_body"),
    ]
    for _i in range(0, len(_ov_cards), 2):
        _c1, _c2 = st.columns(2)
        for _col, (_tk, _bk) in zip((_c1, _c2), _ov_cards[_i:_i + 2]):
            with _col, st.container(border=True):
                st.markdown(f"**{t(_tk)}**")
                st.markdown(t(_bk))

    # ---- 研究主线：三个发现 -----------------------------------------------
    st.subheader(t("ov.k_title"))
    for _tk, _bk in (("ov.k1_title", "ov.k1_body"),
                     ("ov.k2_title", "ov.k2_body"),
                     ("ov.k3_title", "ov.k3_body")):
        with st.container(border=True):
            st.markdown(f"**{t(_tk)}**")
            st.markdown(t(_bk))

    # ---- 为什么重要 --------------------------------------------------------
    st.subheader(t("ov.why_title"))
    st.markdown(t("ov.why_body"))

    # ---- 一眼看全（事实表）------------------------------------------------
    # 用 markdown 表格而不是 st.dataframe：dataframe 会**截断**超长单元格
    # （qualities gates 那行实测被切掉尾句），而表格会换行，信息不丢。
    st.subheader(t("ov.facts_title"))
    _fact_pairs = [
        ("paper_k", "paper_v"), ("chain_k", "chain_v"), ("tiers_k", "tiers_v"),
        ("result_k", "result_v"), ("bias_k", "bias_v"), ("real_k", "real_v"),
        ("gate_k", "gate_v"),
    ]
    _fmd = [f"| {t('ov.fact.item')} | {t('ov.fact.value')} |", "|---|---|"]
    _fmd += [f"| {t('ov.fact.' + k)} | {t('ov.fact.' + v)} |" for k, v in _fact_pairs]
    st.markdown("\n".join(_fmd))

    # ---- 导航地图 ----------------------------------------------------------
    st.subheader(t("ov.map_title"))
    _order = ["demo", "bias", "pipeline", "true_ctr", "ab", "model"]
    _mmd = [f"| {t('ov.map.page')} | {t('ov.map.what')} |", "|---|---|"]
    _mmd += [f"| **{t(f'nav.{k}')}** | {t(f'ov.map.{k}')} |" for k in _order]
    st.markdown("\n".join(_mmd))

    # ---- 引用 --------------------------------------------------------------
    st.subheader(t("ov.cite_title"))
    st.markdown(t("ov.cite_body"))
    # 引用串是语言无关的（学术引用不翻译），故写成模块内常量而非 i18n 条目。
    st.code(
        "Zhao, Y. Bias Propagation and Effective Boundaries of Debiasing in Recommender\n"
        "Systems: An Oracle-Based Diagnosis Study. Preprints 2026, 202610.2582.\n"
        "https://doi.org/10.20944/preprints202610.2582.v1",
        language="text",
    )
    with st.expander(t("ov.cite_bibtex")):
        st.code(
            "@article{zhao2026aerorec,\n"
            "  title   = {Bias Propagation and Effective Boundaries of Debiasing in Recommender Systems:\n"
            "             An Oracle-Based Diagnosis Study},\n"
            "  author  = {Zhao, Yulin},\n"
            "  journal = {Preprints},\n"
            "  year    = {2026},\n"
            "  number  = {202610.2582},\n"
            "  doi     = {10.20944/preprints202610.2582.v1},\n"
            "  url     = {https://doi.org/10.20944/preprints202610.2582.v1}\n"
            "}",
            language="bibtex",
        )

    st.divider()
    st.caption(t("ov.boundary"))

# --------------------------------------------------------------------------
if page == "demo":
    st.header(t("demo.header"))
    st.caption(t("demo.caption"))
    c1, c2 = st.columns([1, 1])
    user_id = c1.number_input(t("demo.user_id"), 0, n_users_max, 0, 1,
                              help=t("demo.user_id_help", n=n_users_max))
    top_k = c2.slider(t("demo.topk"), 5, 20, 10)
    if st.button(t("demo.run"), type="primary"):
        items = engine.recommend(int(user_id), top_k)
        if not items:
            st.warning(t("demo.empty"))
        else:
            st.subheader(t("demo.subheader", user_id=user_id, top_k=top_k))

            # 汇总指标：一眼看到结果的整体形态
            _ctr = [it["ctr"] for it in items]
            m1, m2, m3 = st.columns(3)
            m1.metric(t("demo.m_avg_ctr"), f"{sum(_ctr) / len(_ctr) * 100:.2f}%")
            m2.metric(t("demo.m_max_ctr"), f"{max(_ctr) * 100:.2f}%")
            m3.metric(t("demo.m_categories"), f"{len({it['category'] for it in items})}")

            # 卡片式结果：前三名用「好」色标号，其余用主色
            _cards = []
            for it in items:
                _top = " top" if it["rank"] <= 3 else ""
                _cards.append(
                    f'<div class="rec-card">'
                    f'<div class="rec-rank{_top}">{it["rank"]}</div>'
                    f'<div class="rec-main">'
                    f'<div class="rec-name">{localize_item_name(lang, it["name"])}'
                    f'<span class="rec-tag">{localize_category(lang, it["category"])}</span></div>'
                    f'<div class="rec-meta">{t("demo.card_ctr")} {it["ctr"] * 100:.2f}%'
                    f' · {t("demo.card_base")} {it["base_score"]:.3f}</div>'
                    f'</div>'
                    f'<div class="rec-cvr"><b>{it["cvr"] * 100:.2f}%</b>CVR</div>'
                    f'</div>'
                )
            st.markdown("".join(_cards), unsafe_allow_html=True)

            with st.expander(t("demo.expand_fields")):
                st.dataframe(items, use_container_width=True, hide_index=True)
            st.caption(t("demo.footer_caption"))

# --------------------------------------------------------------------------
elif page == "pipeline":
    st.header(t("pipe.header"))
    st.caption(t("pipe.caption"))
    _STEPS = [
        (t("pipe.s1"), t("pipe.s1d")),
        (t("pipe.s2"), t("pipe.s2d")),
        (t("pipe.s3"), t("pipe.s3d")),
        (t("pipe.s4"), t("pipe.s4d")),
        (t("pipe.s5"), t("pipe.s5d")),
    ]
    st.markdown(
        '<div class="step-flow">'
        + '<span class="step-arrow">→</span>'.join(
            f'<span class="step-chip on">{s}</span>' for s, _ in _STEPS)
        + '</div>',
        unsafe_allow_html=True,
    )
    user_id = st.number_input(t("pipe.user_id"), 0, n_users_max, 0, 1)
    if st.button(t("pipe.run"), type="primary"):
        tr = engine.explain_steps(int(user_id), show=10)
        st.caption(t("pipe.trace_caption", n=tr["candidate_count"]))

        _stage_keys = [
            ("candidates", t("pipe.stage.candidates")),
            ("step1_normalized", t("pipe.stage.s1")),
            ("step2_weighted", t("pipe.stage.s2")),
            ("step3_stratified", t("pipe.stage.s3")),
            ("step4_diversified", t("pipe.stage.s4")),
            ("final", t("pipe.stage.final")),
        ]
        _stage_tabs = st.tabs([lbl for _, lbl in _stage_keys])
        for _tab, (key, _lbl) in zip(_stage_tabs, _stage_keys):
            with _tab:
                rows = tr.get(key) or []
                if not rows:
                    st.write(t("pipe.no_data"))
                    continue
                st.dataframe({
                    t("pipe.col.item"): [localize_item_name(lang, r["name"]) for r in rows],
                    t("pipe.col.category"): [localize_category(lang, r["category"]) for r in rows],
                    t("pipe.col.base"): [r["base_score"] for r in rows],
                    t("pipe.col.ctr"): [r["ctr"] for r in rows],
                    t("pipe.col.cvr"): [r["cvr"] for r in rows],
                }, use_container_width=True, hide_index=True)

        st.subheader(t("pipe.final_header"))
        _fin = tr.get("final") or []
        if _fin:
            st.markdown(
                '<div class="step-flow">'
                + '<span class="step-arrow">→</span>'.join(
                    f'<span class="step-chip">{localize_item_name(lang, r["name"])}</span>' for r in _fin)
                + '</div>',
                unsafe_allow_html=True,
            )
            st.caption(t("pipe.final_caption"))

# --------------------------------------------------------------------------
elif page == "true_ctr":
    st.header(t("tctr.header"))
    st.info(t("tctr.info"), icon="✅")
    c1, c2 = st.columns(2)
    n_eval = c1.slider(t("tctr.n_users"), 20, 200, 100, 20,
                       help=t("tctr.n_users_help"))
    top_k_eval = c2.slider(t("demo.topk"), 5, 20, 10, 1, key="ctr_topk")
    if st.button(t("tctr.run"), type="primary"):
        with st.spinner(t("tctr.spinner")):
            r = engine.benchmark_rankers(n_users=n_eval, top_k=top_k_eval)
        pm = r["per_method"]
        base = r["base_method"]
        base_disp = method_label(lang, base)
        dfm = pm.get("DeepFM+混排", {})

        # 结论先行：三个关键数字放最上面
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t("tctr.m_rel"), f"{dfm.get('rel_to_base', 0):+.1%}",
                  help=t("tctr.m_rel_help", base=base_disp))
        m2.metric(t("tctr.m_oracle_pct"), f"{dfm.get('pct_of_oracle', 0):.1%}",
                  help=t("tctr.m_oracle_pct_help"))
        m3.metric(t("tctr.m_oracle_ctr"), f"{pm.get('Oracle', {}).get('true_ctr', 0):.4f}")
        m4.metric(t("tctr.m_sample"), t("tctr.m_sample_val", n=r["n_users"], k=r["top_k"]))

        _order = ["随机", base, "DeepFM+混排", "Oracle"]
        rows = []
        for name in _order:
            if name not in pm:
                continue
            d = pm[name]
            rows.append({
                t("tctr.col.method"): method_label(lang, name),
                t("tctr.col.true_ctr"): round(d["true_ctr"], 4),
                t("tctr.col.rel", base=base_disp): ("—" if name == base else f"{d['rel_to_base']:+.1%}"),
                t("tctr.col.pct"): f"{d['pct_of_oracle']:.1%}",
            })
        st.subheader(t("tctr.sub_table"))
        st.dataframe(rows, use_container_width=True, hide_index=True)

        _names = [n for n in _order if n in pm]
        st.subheader(t("tctr.sub_chart"))
        st.bar_chart({t("tctr.chart_x"): [method_label(lang, n) for n in _names],
                      t("tctr.chart_y"): [pm[n]["true_ctr"] for n in _names]},
                     x=t("tctr.chart_x"), y=t("tctr.chart_y"), color="#2563eb", height=320)

        st.success(t("tctr.success", n=r["n_users"], k=r["top_k"],
                     rel=dfm.get("rel_to_base", 0), pct=dfm.get("pct_of_oracle", 0),
                     base=base_disp))

# --------------------------------------------------------------------------
elif page == "ab":
    st.header(t("ab.header"))
    st.caption(t("ab.caption"))
    st.warning(t("ab.warning"), icon="⚠️")
    if mode == "lite":
        st.error(t("ab.error_lite"), icon="🚫")
    n_users = st.slider(t("ab.n_users"), 200, 2000, 800, 100,
                        help=t("ab.n_users_help"))
    if st.button(t("ab.run"), type="primary"):
        with st.spinner(t("ab.spinner")):
            r = engine.run_ab(n_users=n_users)

        st.subheader(t("ab.sub_result"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("ab.m_trt_ctr"), f"{r['trt_ctr'] * 100:.2f}%",
                  delta=t("ab.delta_vs_ctrl", v=r["ctr_lift"] * 100))
        c2.metric(t("ab.m_ctrl_ctr"), f"{r['ctrl_ctr'] * 100:.2f}%")
        c3.metric(t("ab.m_trt_cvr"), f"{r['trt_cvr'] * 100:.2f}%",
                  delta=t("ab.delta_vs_ctrl", v=r["cvr_lift"] * 100))
        c4.metric(t("ab.m_ctr_p"), f"{r['ctr_p']:.4f}",
                  help=t("ab.m_ctr_p_help"))

        st.subheader(t("ab.sub_chart"))
        # 配色说明：这里用主色（实验组）与中性灰（对照组）区分，而非红/绿。
        # 原因：本项目已把绿/红固定为**质量语义**（绿=好、红=差，见 data/bias/report.html）。
        # A/B 的「实验组更高」不等于「质量更好」（本页数值本身已被声明为不可引用），
        # 若在此处用绿色标实验组，会与偏差报告里的语义约定冲突。
        st.bar_chart({
            t("ab.chart_metric"): [t("ab.chart_ctr"), t("ab.chart_cvr")],
            t("ab.series_ctrl"): [round(r["ctrl_ctr"], 4), round(r["ctrl_cvr"], 4)],
            t("ab.series_trt"): [round(r["trt_ctr"], 4), round(r["trt_cvr"], 4)],
        }, x=t("ab.chart_metric"), color=["#94a3b8", "#2563eb"], height=320)
        st.caption(t("ab.caption2", cvr_p=r["cvr_p"], ctrl=r["ctrl_pcvr"], trt=r["trt_pcvr"]))

        with st.expander(t("ab.exp_title")):
            st.markdown(t("ab.exp_body"))

# --------------------------------------------------------------------------
elif page == "bias":
    st.header(t("bias.header"))
    st.caption(t("bias.caption"))
    # 明确本页与静态研究报告的对应关系，让审阅者一眼把「交互 demo」和「报告里的表」
    # 对上号：同设置、同种子，只是 τ 从报告的 5 个固定点变成可连续拖动。
    st.caption(t("bias.map_note"))
    with st.expander(t("bias.exp_title"), expanded=False):
        st.info(t("bias.exp_body"), icon="🧪")

    # 两个标签对应研究主线的两半：① 偏差确实毁排序（可交互感受）；
    # ② 去偏何时不划算（结论落地点）。只放 ① 会让访客以为「知道坏了」就完了。
    _tab_sweep, _tab_bnd = st.tabs([t("bias.tab_sweep"), t("bias.tab_boundary")])

    with _tab_sweep:
        tau = st.slider(t("bias.slider"), 0.0, 3.0, 2.0, 0.1,
                        help=t("bias.slider_help"))
        sweep = _bias_full_sweep()
        row = _bias_row_at(sweep, tau)
        sp = row["L2_spearman"]

        m1, m2, m3 = st.columns(3)
        m1.metric(t("bias.m_sp"), f"{sp:+.3f}", help=t("bias.m_sp_help"))
        m2.metric(t("bias.m_gini"), f"{row['exposure_gini']:.3f}", help=t("bias.m_gini_help"))
        m3.metric(t("bias.m_impr"), t("bias.m_impr_val", v=int(row["impr_min"])),
                  help=t("bias.m_impr_help"))

        if sp < -0.1:
            st.error(t("bias.status_bad", tau=tau, sp=sp), icon="🔴")
        elif sp > 0.3:
            st.success(t("bias.status_good", tau=tau, sp=sp), icon="🟢")
        else:
            st.warning(t("bias.status_warn", tau=tau, sp=sp), icon="🟡")

        st.subheader(t("bias.sub_line"))
        st.line_chart(
            {
                "τ": [round(x, 1) for x in sweep["taus"]],
                t("bias.series_l2"): [r["L2_spearman"] for r in sweep["rows"]],
                t("bias.series_l1"): [r["L1_spearman"] for r in sweep["rows"]],
                t("bias.series_l3"): [r["L3_spearman"] for r in sweep["rows"]],
            },
            x="τ", y=[t("bias.series_l2"), t("bias.series_l1"), t("bias.series_l3")], height=340,
        )
        st.caption(t("bias.line_caption", tau=tau, sp=sp))

        c1, c2 = st.columns(2)
        with c1:
            st.subheader(t("bias.sub_scatter"))
            tcr = sweep["true_ctr"]
            l2 = row["L2_est"]
            over = [t("bias.over") if a > b else t("bias.under") for a, b in zip(l2, tcr)]
            st.scatter_chart(
                {t("bias.scatter_x"): tcr, t("bias.scatter_y"): l2,
                 t("bias.scatter_status"): over},
                x=t("bias.scatter_x"), y=t("bias.scatter_y"),
                color=t("bias.scatter_status"), height=300,
            )
            st.caption(t("bias.scatter_caption"))
        with c2:
            st.subheader(t("bias.sub_bar"))
            ef = np.sort(row["exposure_freq"])[::-1]
            st.bar_chart(
                {t("bias.bar_x"): list(range(1, len(ef) + 1)), t("bias.bar_y"): ef},
                x=t("bias.bar_x"), y=t("bias.bar_y"), color="#f59e0b", height=300,
            )
            st.caption(t("bias.bar_caption"))

    with _tab_bnd:
        # 数字全部由 data/bias/boundary.json 驱动（CI 对该文件做逐字节零漂移门禁），
        # 页面里不手打任何边界值。
        bnd = _load_json("data/bias/boundary.json")
        cross = bnd["cross_ess_ratio"]
        st.info(t("bnd.info"), icon="🎯")
        st.success(t("bnd.cross", cross=cross), icon="📐")

        # 判定带容差：gain 在 ±tolerance 内属于噪声，不能报成「有害 / 有益」。
        # 否则会出现「ESS/N=0.91 却显示去偏有害（gain=−0.0001）」这类自相矛盾的行。
        # 注意这是**展示层**的三态读法，不改 JSON 里的任何结论字段。
        _tol = float(bnd.get("tolerance_used", 0.0))
        _brows = []
        for _r in bnd["rows"]:
            _ess = _r["ess_ratio"]
            _gain = _r["gain"]
            if abs(_ess - cross) < 1e-9:
                _verdict = t("bnd.verdict_edge")
            elif _gain < -_tol:
                _verdict = t("bnd.verdict_bad")
            elif _gain > _tol:
                _verdict = t("bnd.verdict_ok")
            else:
                _verdict = t("bnd.verdict_flat")
            _brows.append({
                t("bnd.col.range"): f"[{_r['pi_min']:g}, {_r['pi_max']:g}]",
                t("bnd.col.ratio"): f"{_r['pi_ratio']:g}",
                t("bnd.col.ess"): f"{_ess:.4f}",
                t("bnd.col.naive"): f"{_r['naive_err']:.4f}",
                t("bnd.col.snips"): f"{_r['snips_err']:.4f}",
                t("bnd.col.gain"): f"{_r['gain']:+.4f}",
                t("bnd.col.verdict"): _verdict,
            })
        st.subheader(t("bnd.sub_table"))
        st.dataframe(_brows, use_container_width=True, hide_index=True)

        st.subheader(t("bnd.chart_title"))
        # 按 ESS/N 升序绘图，曲线才是单调可读的（JSON 里是按 π 范围从宽到窄排列的）。
        _sorted = sorted(bnd["rows"], key=lambda r: r["ess_ratio"])
        st.line_chart({
            t("bnd.chart_x"): [r["ess_ratio"] for r in _sorted],
            t("bnd.series_snips"): [r["snips_err"] for r in _sorted],
            t("bnd.series_naive"): [r["naive_err"] for r in _sorted],
            t("bnd.series_gain"): [r["gain"] for r in _sorted],
        }, x=t("bnd.chart_x"),
            y=[t("bnd.series_snips"), t("bnd.series_naive"), t("bnd.series_gain")], height=320)
        st.caption(t("bnd.chart_caption"))

        st.markdown(t("bnd.takeaway"))
        st.markdown(t("bnd.robust"))
        st.warning(t("bnd.caveat"), icon="⚠️")

# --------------------------------------------------------------------------
elif page == "model":
    st.header(t("model.header"))
    st.caption(t("model.caption"))
    meta = _load_json("artifacts/micro/meta.json")
    criteo1m = _load_json("data/real_validation/result.json")
    criteo200k = _load_json("data/real_validation/result_criteo_200k.json")

    st.subheader(t("model.sub_online"))
    a1, a2, a3, a4 = st.columns(4)
    a1.metric(t("model.a_model"), "DeepFM")
    a2.metric(t("model.a_embed"), str(meta["embed_dim"]))
    a3.metric(t("model.a_hidden"), "→".join(str(h) for h in meta["dnn_hidden"]))
    a4.metric(t("model.a_samples"), f"{meta['train_sample']:,}")

    st.caption(t("model.online_caption"))

    st.subheader(t("model.sub_real"))
    st.info(t("model.real_info"), icon="✅")
    st.bar_chart(
        {
            t("model.chart_x"): [t("model.bar.deepfm_1m"), t("model.bar.lr_1m"),
                                 t("model.bar.deepfm_200k"), t("model.bar.constant")],
            t("model.chart_y"): [
                criteo1m["val_auc_deepfm"], criteo1m["val_auc_logistic_baseline"],
                criteo200k["val_auc_deepfm"], criteo1m["val_auc_naive_constant"],
            ],
        },
        x=t("model.chart_x"), y=t("model.chart_y"), color="#2563eb", height=320,
    )
    st.success(t("model.real_success",
                 auc1m=criteo1m["val_auc_deepfm"], auc200k=criteo200k["val_auc_deepfm"],
                 lr=criteo1m["val_auc_logistic_baseline"]))

    # 第二个真实数据集：单靠 Criteo 会被质疑「只是碰巧适合稀疏特征」，
    # 补一份结构完全不同（稠密为主、表格二分类）的数据集才构成迁移性证据。
    adult = _load_json("data/real_validation/result_adult.json")
    st.subheader(t("model.sub_adult"))
    st.info(t("model.adult_info"), icon="📊")
    st.success(t("model.adult_success",
                 adult=adult["val_auc_deepfm"], n_val=adult["n_val"],
                 adult_lr=adult["val_auc_logistic_baseline"]), icon="✅")

    with st.expander(t("model.exp_title"), expanded=False):
        st.markdown(t("model.exp_body"))

    # 复现性放在「模型与验证」页而非单独页：访客看到 AUC 后的第一个疑问
    # 往往是「这数是真的吗」，此处正好接住。
    st.subheader(t("model.sub_repro"))
    st.info(t("model.repro_info"), icon="🔒")

    st.subheader(t("model.sub_live"))
    uid = st.number_input(t("model.uid"), 0, n_users_max, 0, 1, key="model_uid")
    if st.button(t("model.run_live"), type="primary"):
        items = engine.recommend(int(uid), 8)
        rows = [{t("model.live_col_item"): localize_item_name(lang, it["name"]),
                 t("model.live_col_cat"): localize_category(lang, it["category"]),
                 t("model.live_col_ctr"): it["ctr"]} for it in items]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(t("model.live_caption"))

# --------------------------------------------------------------------------
# 页脚：把「这个 Demo 是什么、还能看什么」收在一处，便于访客继续深入。
st.markdown(f'<div class="foot">{t("foot.body")}</div>', unsafe_allow_html=True)
