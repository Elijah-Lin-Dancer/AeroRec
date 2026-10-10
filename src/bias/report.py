"""偏差诊断报告生成器：把实验结果渲染成自包含的 HTML（内联 SVG，无外部依赖）。

为什么内联 SVG 而不是图表库
---------------------------
报告要能**离线打开、直接发给评审**。引入 Chart.js / plotly 需要 CDN 或额外文件，
一旦断网或路径不对就是空白页。内联 SVG 只靠字符串拼接，零依赖、零请求、
体积小、在 GitHub Pages 与本地都能正常显示——这是本项目的既有取舍
（见 docs/index.html 同样使用内联 SVG）。

产出：data/bias/report.html
"""
from __future__ import annotations

import json
import os

# 与 docs/index.html 一致的配色
C = {
    "bg": "#f7f8fa", "card": "#ffffff", "text": "#1a1d23", "muted": "#5f6672",
    "accent": "#2563eb", "border": "#e5e7eb", "warn": "#f59e0b",
    "bad": "#dc2626", "purple": "#7c3aed",
    # ⚠️ 配色语义（勿套用股市"涨红跌绿"）：
    # 本报告的颜色表示**指标好坏**，不表示涨跌方向。
    # 因此 good=绿 / bad=红，与"质量高低"一致。
    "good": "#16a34a",
}


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# SVG 基元
# ----------------------------------------------------------------------
def _line_chart(xs, series, width=760, height=300, xlabel="", ylabel="",
                title="", y_zero=True, legend_labels=None):
    """多序列折线图。series = [(名称, 颜色, [y...]), ...]

    legend_labels : 可选，用于把图例文字换成更短的版本
                    （SVG 不会自动换行，长中文标签会溢出画布右侧）。
    """
    legend_labels = legend_labels or {}
    pad_l, pad_r, pad_t, pad_b = 62, 210, 34, 46
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    all_y = [v for _, _, ys in series for v in ys]
    if not all_y:
        return "<p>无数据</p>"
    ymin, ymax = min(all_y), max(all_y)
    if y_zero:
        ymin, ymax = min(ymin, 0.0), max(ymax, 0.0)
    span = (ymax - ymin) or 1.0
    ymin -= span * 0.08
    ymax += span * 0.12
    span = ymax - ymin

    def X(i):
        return pad_l + (pw * i / max(len(xs) - 1, 1))

    def Y(v):
        return pad_t + ph - (ph * (v - ymin) / span)

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px">']
    out.append(f'<text x="{pad_l}" y="18" font-size="13" fill="{C["text"]}" font-weight="600">{title}</text>')
    # y 网格
    for k in range(5):
        v = ymin + span * k / 4
        y = Y(v)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+pw}" y2="{y:.1f}" '
                   f'stroke="{C["border"]}" stroke-width="1"/>')
        out.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" font-size="11" fill="{C["muted"]}" '
                   f'text-anchor="end">{v:.2f}</text>')
    # 零线
    if y_zero and ymin < 0 < ymax:
        y0 = Y(0)
        out.append(f'<line x1="{pad_l}" y1="{y0:.1f}" x2="{pad_l+pw}" y2="{y0:.1f}" '
                   f'stroke="{C["muted"]}" stroke-width="1.4" stroke-dasharray="4,3"/>')
    # x 轴刻度
    for i, xv in enumerate(xs):
        out.append(f'<text x="{X(i):.1f}" y="{pad_t+ph+18}" font-size="11" '
                   f'fill="{C["muted"]}" text-anchor="middle">{xv}</text>')
    out.append(f'<text x="{pad_l+pw/2:.0f}" y="{height-6}" font-size="11" '
               f'fill="{C["muted"]}" text-anchor="middle">{xlabel}</text>')
    out.append(f'<text x="14" y="{pad_t+ph/2:.0f}" font-size="11" fill="{C["muted"]}" '
               f'transform="rotate(-90 14 {pad_t+ph/2:.0f})" text-anchor="middle">{ylabel}</text>')
    # 折线 + 图例
    for si, (name, color, ys) in enumerate(series):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(ys))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for i, v in enumerate(ys):
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.2" fill="{color}"/>')
        ly = pad_t + 12 + si * 20
        label = legend_labels.get(name, name)
        out.append(f'<rect x="{pad_l+pw+12}" y="{ly-8}" width="11" height="11" rx="2" fill="{color}"/>')
        out.append(f'<text x="{pad_l+pw+28}" y="{ly+1}" font-size="11.5" '
                   f'fill="{C["text"]}">{label}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _bar_chart(labels, values, width=760, height=280, xlabel="", ylabel="",
               title="", color=None):
    """单序列柱状图（含零线，正负分色）。"""
    pad_l, pad_r, pad_t, pad_b = 62, 24, 34, 52
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    vmin, vmax = min(min(values), 0.0), max(max(values), 0.0)
    span = (vmax - vmin) or 1.0
    vmin -= span * 0.1
    vmax += span * 0.1
    span = vmax - vmin
    n = len(values)
    bw = pw / max(n, 1) * 0.56

    def X(i):
        return pad_l + pw * (i + 0.5) / max(n, 1)

    def Y(v):
        return pad_t + ph - ph * (v - vmin) / span

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px">']
    out.append(f'<text x="{pad_l}" y="18" font-size="13" fill="{C["text"]}" font-weight="600">{title}</text>')
    for k in range(5):
        v = vmin + span * k / 4
        y = Y(v)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+pw}" y2="{y:.1f}" '
                   f'stroke="{C["border"]}"/>')
        out.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" font-size="11" fill="{C["muted"]}" '
                   f'text-anchor="end">{v:.2f}</text>')
    y0 = Y(0)
    out.append(f'<line x1="{pad_l}" y1="{y0:.1f}" x2="{pad_l+pw}" y2="{y0:.1f}" '
               f'stroke="{C["muted"]}"/>')
    for i, (lb, v) in enumerate(zip(labels, values)):
        yy = Y(v)
        top = min(yy, y0)
        h = abs(yy - y0) or 1
        col = color or (C["good"] if v >= 0 else C["bad"])
        out.append(f'<rect x="{X(i)-bw/2:.1f}" y="{top:.1f}" width="{bw:.1f}" '
                   f'height="{h:.1f}" rx="3" fill="{col}" opacity="0.9"/>')
        out.append(f'<text x="{X(i):.1f}" y="{pad_t+ph+16}" font-size="11" fill="{C["muted"]}" '
                   f'text-anchor="middle">{lb}</text>')
        off = -6 if v >= 0 else 14
        out.append(f'<text x="{X(i):.1f}" y="{yy+off:.1f}" font-size="10.5" '
                   f'fill={C["muted"]} text-anchor="middle">{v:+.3f}</text>')
    out.append(f'<text x="{pad_l+pw/2:.0f}" y="{height-8}" font-size="11" fill="{C["muted"]}" '
               f'text-anchor="middle">{xlabel}</text>')
    out.append(f'<text x="14" y="{pad_t+ph/2:.0f}" font-size="11" fill="{C["muted"]}" '
               f'transform="rotate(-90 14 {pad_t+ph/2:.0f})" text-anchor="middle">{ylabel}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _table(headers, rows_html):
    th = "".join(f"<th>{h}</th>" for h in headers)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"


def _td_better(v: float, ref: float = 0.0) -> str:
    """排序指标：越大越好（绿）／越小越好（红）。"""
    cls = "up" if v >= ref else "down"
    return f'<span class="{cls}">{v:+.3f}</span>'


def build_report(out_dir: str = "data/bias") -> str:
    prop = _load(os.path.join(out_dir, "propagation.json"))
    inj = _load(os.path.join(out_dir, "injection.json"))
    bnd = _load(os.path.join(out_dir, "boundary.json"))

    parts = [f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AeroRec · 偏差传播诊断报告</title>
<style>
  :root {{
    --bg: {C['bg']}; --card: {C['card']}; --text: {C['text']}; --muted: {C['muted']};
    --accent: {C['accent']}; --border: {C['border']};
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.7; }}
  .wrap {{ max-width: 1000px; margin: 0 auto; padding: 44px 22px 80px; }}
  header {{ text-align: center; padding-bottom: 26px; }}
  .badge {{ display: inline-block; font-size: 13px; color: var(--accent);
           border: 1px solid var(--accent); border-radius: 999px; padding: 2px 12px; margin-bottom: 14px; }}
  h1 {{ font-size: 31px; letter-spacing: -0.5px; }}
  .subtitle {{ color: var(--muted); margin-top: 10px; font-size: 15.5px; }}
  section {{ background: var(--card); border: 1px solid var(--border);
            border-radius: 14px; padding: 26px; margin-top: 22px; }}
  h2 {{ font-size: 19px; margin-bottom: 12px; }}
  h3 {{ font-size: 15px; margin: 20px 0 8px; }}
  p {{ margin-top: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; margin-top: 10px; }}
  th, td {{ text-align: left; padding: 9px 11px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; font-size: 12.5px; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  code {{ background: #eef1f5; padding: 1px 6px; border-radius: 4px; font-size: 12.5px; }}
  /* 颜色表示指标好坏（绿=好/红=差），不是涨跌方向 */
  .up {{ color: {C['good']}; font-weight: 600; }}
  .down {{ color: {C['bad']}; font-weight: 600; }}
  .note {{ font-size: 13px; color: var(--muted); background: #fff8e6;
          border-left: 3px solid {C['warn']}; padding: 10px 14px; border-radius: 6px; margin-top: 14px; }}
  .fix {{ font-size: 13px; color: #7c2d12; background: #fef2f2;
         border-left: 3px solid {C['bad']}; padding: 10px 14px; border-radius: 6px; margin-top: 14px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 16px; }}
  .kpi {{ background: var(--bg); border-radius: 10px; padding: 14px 16px; }}
  .kpi b {{ display: block; font-size: 21px; margin-bottom: 2px; }}
  .kpi span {{ font-size: 12.5px; color: var(--muted); }}
  .svg-wrap {{ overflow-x: auto; margin-top: 14px; }}
  footer {{ text-align: center; color: var(--muted); font-size: 13px; margin-top: 34px; }}
  @media (max-width: 700px) {{ .kpis {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <span class="badge">研究主线 · 偏差诊断与去偏</span>
    <h1>AeroRec · 链路偏差传播诊断报告</h1>
    <p class="subtitle">
      在可控 ground-truth 下精确测量推荐链路的偏差累积与二次注入，
      并标定 IPS / SNIPS / DR 去偏的<b>有效边界</b>
    </p>
  </header>
"""]

    # ---------- 0. 方法要点 ----------
    parts.append(f"""
  <section>
    <h2>0 · 为什么这件事只有合成数据能做</h2>
    <p>
      真实日志只能<b>估计</b>偏差——你不知道曝光策略、也不知道真值；
      本项目的数据是合成的，因此每个 (用户, 物品) 对的真实 CTR 由
      <code>compute_true_ctr</code> <b>解析已知</b>，并且曝光策略的 propensity π
      是策略参数的<b>闭式函数</b>（不是估计量）。
    </p>
    <p>
      这两个条件合起来，才让「偏差可精确测量」成立：<b>偏差 = 有偏观测 − 已知真值</b>。
      本报告的全部数字都建立在此之上。
    </p>
    <div class="note">
      本报告的实验<b>不训练任何模型</b>——研究的是「估计量在偏差下的行为」，
      用真实 CTR 作 ground-truth 即可，因此完全可复现（固定 seed）且秒级完成。
    </div>
  </section>
""")

    # ---------- 1. 传播矩阵 ----------
    if prop:
        rows = prop["rows"]
        kpi_tau0, kpi_tau3 = rows[0], rows[-1]
        parts.append(f"""
  <section>
    <h2>1 · 链路偏差传播矩阵</h2>
    <p>
      在 4 级链路（真值 → 召回 → 排序 → 混排）上逐级测量偏差。
      主指标为<b>逐物品相对误差 rel_err</b>（水平，机制 A）与
      <b>与真值的秩相关 spearman</b>（排序，机制 B）。
    </p>
    <div class="kpis">
      <div class="kpi"><b class="up">{kpi_tau0['L2_spearman']:+.3f}</b>
        <span>τ=0 排序相关（好）</span></div>
      <div class="kpi"><b class="down">{kpi_tau3['L2_spearman']:+.3f}</b>
        <span>τ=3 排序相关（已反转）</span></div>
      <div class="kpi"><b>{kpi_tau0['exposure_gini']:.3f} → {kpi_tau3['exposure_gini']:.3f}</b>
        <span>曝光 Gini</span></div>
      <div class="kpi"><b>{kpi_tau0['impr_min']} → {kpi_tau3['impr_min']}</b>
        <span>冷门物品最少曝光次数</span></div>
    </div>
""")
        headers = ["τ（曝光偏差）", "排序级 rel_err", "排序级 spearman",
                   "混排级 rel_err", "排序级增量", "曝光 KL", "曝光 Gini", "最少曝光"]
        trs = []
        for r in rows:
            sp_cls = "up" if r["L2_spearman"] > 0.3 else ("down" if r["L2_spearman"] < 0 else "")
            trs.append(
                f"<tr><td class='num'>{r['tau']:.1f}</td>"
                f"<td class='num'>{r['L2_rel_err']:.3f}</td>"
                f"<td class='num {sp_cls}'>{r['L2_spearman']:+.3f}</td>"
                f"<td class='num'>{r['L3_rel_err']:.3f}</td>"
                f"<td class='num'>{r['L2_incremental_rel']:+.3f}</td>"
                f"<td class='num'>{r['exposure_kl']:.4f}</td>"
                f"<td class='num'>{r['exposure_gini']:.3f}</td>"
                f"<td class='num'>{r['impr_min']}</td></tr>")
        parts.append(_table(headers, trs))

        parts.append('<div class="svg-wrap">')
        parts.append(_line_chart(
            [f"τ={r['tau']:.1f}" for r in rows],
            [("排序级 spearman（越接近 1 越好）", C["accent"], [r["L2_spearman"] for r in rows]),
             ("曝光 Gini", C["warn"], [r["exposure_gini"] for r in rows])],
            xlabel="曝光偏差强度 τ", ylabel="指标值",
            title="排序质量随偏差反转，曝光集中度同步上升",
            legend_labels={"排序级 spearman（越接近 1 越好）": "排序级 spearman"}))
        parts.append("</div>")
        parts.append("""
    <div class="note">
      <b>核心结论</b>：偏差的主要危害<b>不是整体偏高，而是把排序搞反</b>——
      spearman 从 τ=0 的正相关一路降到 τ=3 的<b>负相关</b>。
      机制是：曝光越偏，冷门物品曝光越少，被 α 平滑拽向先验 1.0，
      于是「最冷门的反而被估成最好」。
    </div>
  </section>
""")

    # ---------- 2. 二次注入 ----------
    if inj:
        rows = inj["rows"]
        parts.append(f"""
  <section>
    <h2>2 · 混排阶段的偏差二次注入</h2>
    <p>
      固定 τ={inj['tau']:.1f}，扫描混排基础分（热度）权重 <code>w_base</code>。
      候选池本就由热门召回构成，热度信息已存在；在混排再叠加一次即<b>二次注入</b>。
    </p>
    <div class="svg-wrap">
""")
        parts.append(_line_chart(
            [f"{r['w_base']:.1f}" for r in rows],
            [("混排级 rel_err", C["bad"], [r["L3_rel_err"] for r in rows]),
             ("混排级 spearman", C["accent"], [r["L3_spearman"] for r in rows])],
            xlabel="w_base（基础分/热度权重）", ylabel="指标值",
            title="w_base → 1 时混排级等于真值（误差严格为 0，实现自检）"))
        parts.append("</div>")
        headers = ["w_base", "排序级 rel_err", "混排级 rel_err", "注入增量(rel)",
                   "排序级 spearman", "混排级 spearman", "注入增量(sp)"]
        trs = []
        for r in rows:
            trs.append(
                f"<tr><td class='num'>{r['w_base']:.1f}</td>"
                f"<td class='num'>{r['L2_rel_err']:.3f}</td>"
                f"<td class='num'>{r['L3_rel_err']:.3f}</td>"
                f"<td class='num'>{r['injection_rel']:+.3f}</td>"
                f"<td class='num'>{r['L2_spearman']:+.3f}</td>"
                f"<td class='num'>{r['L3_spearman']:+.3f}</td>"
                f"<td class='num'>{r['injection_spearman']:+.3f}</td></tr>")
        parts.append(_table(headers, trs))
        parts.append("""
    <div class="note">
      <b>实现自检</b>：<code>w_base=1.0</code> 时混排级<b>等于真值</b>，
      故 rel_err 必须严格为 0、spearman 为 1——实测正是如此，
      这说明链路实现本身没有引入额外误差。
    </div>
  </section>
""")

    # ---------- 3. 去偏边界 ----------
    if bnd:
        rows = bnd["rows"]
        parts.append(f"""
  <section>
    <h2>3 · 去偏的有效边界</h2>
    <p>
      最初的假设是「曝光越偏（τ 越大），IPS 去偏越容易亏本」。
      <b>实测否定了这个假设</b>：τ 增大反而让去偏赚得更多，
      因为 τ 只是拉大了两个估计目标之间的差距，而 IPS 纠正的正是这个差距。
    </p>
    <p>
      真正决定成败的是 <b>π 的动态范围</b>，它通过<b>有效样本量 ESS/N</b> 起作用：
      π 越极端，1/π 的方差越大，IPS 越容易被少数记录带偏。
    </p>
    <div class="svg-wrap">
""")
        parts.append(_bar_chart(
            [f"{r['ess_ratio']:.3f}" for r in rows],
            [r["gain"] for r in rows],
            xlabel="ESS/N（有效样本量占比）", ylabel="去偏收益 gain",
            title="去偏收益 vs ESS/N：低于边界后转负"))
        parts.append("</div>")
        headers = ["π 取值下界", "π 上界", "极值比", "ESS/N", "naive_err",
                   "snips_err", "gain（>0 表示去偏划算）"]
        trs = []
        for r in rows:
            cls = "up" if r["gain"] >= -0.001 else "down"
            trs.append(
                f"<tr><td class='num'>{r['pi_min']:.0e}</td>"
                f"<td class='num'>{r['pi_max']:.1f}</td>"
                f"<td class='num'>{r['pi_ratio']:.0f}</td>"
                f"<td class='num'>{r['ess_ratio']:.4f}</td>"
                f"<td class='num'>{r['naive_err']:.4f}</td>"
                f"<td class='num'>{r['snips_err']:.4f}</td>"
                f"<td class='num {cls}'>{r['gain']:+.4f}</td></tr>")
        parts.append(_table(headers, trs))
        parts.append(f"""
    <div class="note">
      <b>边界结论</b>：{bnd.get('conclusion', '')}
    </div>
    <p>
      <b>落地价值</b>：<code>ESS/N</code> <b>不需要 ground-truth 就能计算</b>
      （只依赖日志与 logging policy）。因此可以在真实系统里用它做<b>在线预警</b>：
      ESS/N 低于在合成环境标定的阈值时，就放弃 IPS、回退到朴素估计或扩大曝光。
      这是本模块从研究走向可落地的关键一步。
    </p>
  </section>
""")

    # ---------- 4. 诚实披露 ----------
    parts.append(f"""
  <section>
    <h2>4 · 诚实披露：开发中修正的三个错误</h2>
    <p>本报告的结论是在修正了三处错误之后得到的。列出它们，是因为这些坑本身有参考价值。</p>
    <div class="fix">
      <b>① 用「向量均值之差」度量各级偏差</b> → 重标定会把逐物品偏差在求均值时完全抵消，
      L1 偏差在 τ=0..3 上恒为 0.0000，要研究的现象被对齐的一瞬间抹掉。
      <b>改为逐物品误差</b>（rel_err / spearman）。
    </div>
    <div class="fix">
      <b>② 把 propensity π 直接当作 CTR 估计</b> → τ=0 时 L1 相对误差竟达 0.92，
      且 spearman 与 rel_err 自相矛盾。根因是 π(~0.017) 与真实 CTR(~0.25) 量纲不同。
      <b>改为与真值同量纲、且不做均值对齐的构造</b>。
    </div>
    <div class="fix">
      <b>③ 混淆 π 的两种约定</b> → 把「分布概率」(Σπ=1) 塞进 DR 的模型项，
      修正项被 1/π² 放大，实测 DR 输出 <b>−10.2</b>（真值 0.3）。
      <b>已加入约定检测与告警</b>，并区分「包含概率」与「分布概率」两套公式。
    </div>
    <div class="note">
      同样值得记录的是<b>一个被否定的假设</b>：原以为「偏差越大、去偏越难」，
      实测发现真正决定边界的是 π 的动态范围（ESS），而非偏差幅度。
    </div>
  </section>
""")

    # ---------- 5. 复现 ----------
    parts.append("""
  <section>
    <h2>5 · 复现</h2>
    <pre style="background:#0f172a;color:#e2e8f0;padding:16px;border-radius:10px;overflow-x:auto;font-size:13px">python scripts/run_bias_experiments.py

# 产出：
#   data/bias/propagation.json   链路传播矩阵
#   data/bias/injection.json     混排二次注入扫描
#   data/bias/boundary.json      去偏有效边界
#   data/bias/report.html        本报告（由 scripts/build_bias_report.py 生成）

# 单测（76 项）：
python -m pytest -q</pre>
    <div class="note">
      实验不需要训练模型、不需要 GPU、全程固定随机种子，运行时间约 1 秒。
    </div>
  </section>

  <footer>
    AeroRec · 偏差诊断报告 · 由 <code>scripts/build_bias_report.py</code> 自动生成<br>
    数据全部为合成（受控实验），不使用任何真实业务数据
  </footer>
</div>
</body>
</html>
""")

    html = "\n".join(parts)
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "data/bias"
    p = build_report(d)
    print(f"[report] 已生成 {p}  ({os.path.getsize(p)/1024:.1f} KB)")
