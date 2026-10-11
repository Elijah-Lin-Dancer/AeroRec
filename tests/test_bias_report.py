"""HTML 报告生成器。

断言报告**结构完整**且**自包含**（可离线打开、可直接发送），
因为这是交付物，不能依赖 CDN 或外部文件。
"""
from __future__ import annotations

import json
import os

import pytest
from src.bias.report import build_report


@pytest.fixture()
def report_dir(tmp_path):
    """构造三份最小实验 JSON，避免依赖真实实验产物。"""
    d = tmp_path
    (d / "propagation.json").write_text(json.dumps({
        "taus": [0.0, 1.0, 3.0],
        "rows": [
            {"tau": 0.0, "L2_rel_err": 0.66, "L3_rel_err": 0.66,
             "L2_incremental_rel": 0.22, "L1_rel_err": 0.44, "L3_incremental_rel": 0.0,
             "L1_spearman": 0.18, "L2_spearman": 0.78, "L3_spearman": 0.78,
             "L2_rank_mae": 0.2, "L2_signed": 0.1, "impr_min": 13, "impr_max": 40,
             "exposure_kl": 0.02, "exposure_gini": 0.11},
            {"tau": 1.0, "L2_rel_err": 0.87, "L3_rel_err": 0.87,
             "L2_incremental_rel": 0.87, "L1_rel_err": 0.0, "L3_incremental_rel": 0.0,
             "L1_spearman": 1.0, "L2_spearman": 0.54, "L3_spearman": 0.54,
             "L2_rank_mae": 0.25, "L2_signed": 0.2, "impr_min": 6, "impr_max": 109,
             "exposure_kl": 0.11, "exposure_gini": 0.27},
            {"tau": 3.0, "L2_rel_err": 2.0, "L3_rel_err": 2.0,
             "L2_incremental_rel": 1.37, "L1_rel_err": 0.63, "L3_incremental_rel": 0.0,
             "L1_spearman": 1.0, "L2_spearman": -0.57, "L3_spearman": -0.57,
             "L2_rank_mae": 0.4, "L2_signed": 0.3, "impr_min": 0, "impr_max": 197,
             "exposure_kl": 0.70, "exposure_gini": 0.62},
        ],
        "metric": "rel_err + spearman",
    }), encoding="utf-8")
    (d / "injection.json").write_text(json.dumps({
        "tau": 2.0,
        "rows": [
            {"w_base": 0.0, "L2_rel_err": 1.43, "L3_rel_err": 1.43, "injection_rel": 0.0,
             "L2_spearman": -0.28, "L3_spearman": -0.28, "injection_spearman": 0.0},
            {"w_base": 1.0, "L2_rel_err": 1.44, "L3_rel_err": 0.0, "injection_rel": -1.44,
             "L2_spearman": -0.22, "L3_spearman": 1.0, "injection_spearman": 1.22},
        ],
        "metric": "rel_err + spearman",
    }), encoding="utf-8")
    (d / "boundary.json").write_text(json.dumps({
        "t_pop": 0.25, "n_records": 20000,
        "rows": [
            {"pi_min": 1.0, "pi_max": 1.0, "pi_ratio": 1.0, "ess_ratio": 1.0,
             "naive_err": 0.002, "snips_err": 0.002, "gain": 0.0},
            {"pi_min": 0.05, "pi_max": 2.0, "pi_ratio": 40.0, "ess_ratio": 0.358,
             "naive_err": 0.002, "snips_err": 0.004, "gain": -0.002},
            {"pi_min": 1e-4, "pi_max": 2.0, "pi_ratio": 20000.0, "ess_ratio": 0.006,
             "naive_err": 0.002, "snips_err": 0.033, "gain": -0.031},
        ],
        "cross_ess_ratio": 0.358,
        "conclusion": "ESS/N 降至约 0.358 以下时，去偏收益转负。",
    }), encoding="utf-8")
    return str(d)


def test_report_generated_and_has_sections(report_dir):
    path = build_report(report_dir)
    assert os.path.exists(path)
    html = open(path, encoding="utf-8").read()
    # 五个核心小节 + 三张图 + 三张表
    assert html.count("<section") >= 5
    assert html.count("<svg") == 3
    assert html.count("<table") == 3


def test_report_is_self_contained(report_dir):
    """自包含：无 CDN / 无外部脚本 / 无外链资源（wx.org 命名空间除外）。"""
    html = open(build_report(report_dir), encoding="utf-8").read()
    stripped = html.replace("http://www.w3.org", "").replace("https://www.w3.org", "")
    assert "http://" not in stripped
    assert "https://" not in stripped
    assert "<script" not in html.lower()
    assert 'src="' not in html          # 无外部资源引用


def test_report_contains_key_conclusions(report_dir):
    """关键数字与结论必须真的落在报告里（避免模板化空壳）。"""
    html = open(build_report(report_dir), encoding="utf-8").read()
    assert "偏差的主要危害" in html and "把排序搞反" in html      # 偏差传播结论
    assert "数值为 0" or "实现自检" in html                       # 二次注入自检
    assert "ESS/N" in html                                       # 去偏有效边界
    assert "0.358" in html                                       # 边界阈值
    assert "不需要" in html and "ground-truth" in html            # 落地价值


def test_report_color_semantics_are_quality_not_stock(report_dir):
    """配色必须是"绿=好/红=差"的质量语义，不能套用股市涨跌。"""
    from src.bias.report import C
    assert C["good"] == "#16a34a"
    assert C["bad"] == "#dc2626"
    html = open(build_report(report_dir), encoding="utf-8").read()
    assert "不是涨跌方向" in html


def test_report_handles_missing_files_gracefully(tmp_path):
    """缺 JSON 时不应崩溃，只渲染已有部分。"""
    (tmp_path / "propagation.json").write_text(json.dumps({
        "taus": [0.0], "rows": [{
            "tau": 0.0, "L2_rel_err": 0.6, "L3_rel_err": 0.6, "L2_incremental_rel": 0.2,
            "L1_rel_err": 0.4, "L3_incremental_rel": 0.0, "L1_spearman": 0.2,
            "L2_spearman": 0.8, "L3_spearman": 0.8, "L2_rank_mae": 0.2,
            "L2_signed": 0.1, "impr_min": 10, "impr_max": 30,
            "exposure_kl": 0.02, "exposure_gini": 0.1}],
    }), encoding="utf-8")
    path = build_report(str(tmp_path))
    html = open(path, encoding="utf-8").read()
    assert "<section" in html
    assert html.count("<svg") == 1        # 只有传播矩阵那张
