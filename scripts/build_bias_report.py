"""生成偏差诊断 HTML 报告（自包含、内联 SVG、零外部依赖）。

用法：
    python scripts/build_bias_report.py                 # 读 data/bias/*.json
    python scripts/build_bias_report.py --dir data/bias

前置：先跑 `python scripts/run_bias_experiments.py` 产出三个 JSON。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bias.report import build_report


def main():
    ap = argparse.ArgumentParser(description="生成偏差诊断 HTML 报告")
    ap.add_argument("--dir", default="data/bias", help="实验 JSON 所在目录")
    args = ap.parse_args()

    missing = [f for f in ("propagation.json", "injection.json", "boundary.json")
               if not os.path.exists(os.path.join(args.dir, f))]
    if missing:
        print(f"[report] 警告：缺少 {missing}，先运行 scripts/run_bias_experiments.py")

    path = build_report(args.dir)
    print(f"[report] 已生成：{path}  ({os.path.getsize(path) / 1024:.1f} KB)")
    print("[report] 自包含 HTML（内联 SVG），可直接用浏览器打开或发送。")


if __name__ == "__main__":
    main()
