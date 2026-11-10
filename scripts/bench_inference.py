"""在线推理性能基准：冷启动 / 单请求延迟分位 / 吞吐 QPS / 峰值内存。

用法：
    python scripts/bench_inference.py                 # 默认 micro 档、500 次测量
    python scripts/bench_inference.py --iters 2000
    python scripts/bench_inference.py --mode lite

产出：
    data/bench.json     结构化结果（供 README / 报告引用）
    stdout              人类可读表格

口径与诚实边界
--------------
- 测的是**在线推理路径**（加载 numpy 权重 → 召回 → DeepFM/LR 前向 → 混排 → 组装结果），
  **不含训练**。冷启动 = 导入模块 + 构造引擎（含即时数据生成与权重加载）。
- 计时为**单进程、单线程、串行**调用，反映的是「一次请求端到端」的服务成本，
  不模拟并发；QPS 即 1 / 平均延迟，仅作数量级参考。
- 结果对机器敏感：本脚本会自动记录 CPU 型号 / Python / numpy 版本；换机器数值会变，
  故所有结论均应连同环境一并引用。
- 运行前会在正式计时前做 warm-up 丢弃，降低冷缓存与首调 JIT/分支预测噪声。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Protocol

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


class _Engine(Protocol):
    """三档引擎共有的最小接口（本脚本只用到 `recommend`）。

    三档引擎是三个**互不继承**的类，但对外接口刻意保持一致
    （见 src/serve/engine_micro.py 的类注释）。用 Protocol 把这份契约
    写进类型系统，既避免把 `factory` 退化成 `Any` 而丢失检查，
    也让"接口漂移"在 mypy 阶段就暴露出来，而不是等到线上才炸。
    """

    def recommend(self, user_id: int, top_k: int | None = None) -> list: ...


class _EngineFactory(Protocol):
    """可调用、无参构造的引擎工厂。"""

    def __call__(self) -> _Engine: ...


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _measure_import_seconds(mode: str) -> float:
    """在**独立子进程**里量「导入在线引擎模块」的纯导入耗时（隔离掉主进程已导入的影响）。"""
    mod = {"micro": "src.serve.engine_micro",
           "lite": "src.serve.engine_lite",
           "full": "src.serve.engine"}[mode]
    code = (
        "import time, sys; "
        "t=time.perf_counter(); "
        f"__import__('{mod}'); "
        "print(time.perf_counter()-t)"
    )
    try:
        p = subprocess.run([sys.executable, "-c", code], cwd=str(_ROOT),
                           capture_output=True, text=True, timeout=300)
        return round(float(p.stdout.strip().splitlines()[-1]), 4)
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description="在线推理性能基准")
    ap.add_argument("--mode", default="micro", choices=["micro", "lite", "full"])
    ap.add_argument("--iters", type=int, default=500, help="正式计时调用次数")
    ap.add_argument("--warmup", type=int, default=50, help="warm-up 丢弃次数")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default="data/bench.json")
    args = ap.parse_args()

    os.environ["AEROREC_MODE"] = args.mode

    import numpy as np

    # ---- 1) 冷启动：模块导入（子进程）+ 引擎构造（本进程）----
    import_s = _measure_import_seconds(args.mode)

    # 只导入**当前档位**所需的引擎模块：micro / lite 走纯 numpy，
    # 绝不能因为导入 full 引擎（src.serve.engine）而把 torch 拖进内存。
    # 三个引擎类互不继承，故先分别导入，再统一收进一个工厂变量；
    # 类型由 _EngineFactory 协议约束（见文件头）。
    factory: _EngineFactory
    if args.mode == "micro":
        from src.serve.engine_micro import MicroRecommendationEngine
        factory = MicroRecommendationEngine
    elif args.mode == "lite":
        from src.serve.engine_lite import LiteRecommendationEngine
        factory = LiteRecommendationEngine
    else:
        from src.serve.engine import RecommendationEngine
        factory = RecommendationEngine

    t0 = time.perf_counter()
    engine = factory()
    init_s = time.perf_counter() - t0

    import resource
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    # ---- 2) 延迟分位 + 吞吐（串行、剔除 warm-up）----
    n_users = getattr(engine, "n_users", 10000)
    rng = np.random.default_rng(0)
    uids = rng.integers(0, n_users, size=args.warmup + args.iters)

    for i in range(args.warmup):
        engine.recommend(int(uids[i]), args.top_k)

    lat_ms = []
    t_wall0 = time.perf_counter()
    for i in range(args.warmup, args.warmup + args.iters):
        s = time.perf_counter()
        engine.recommend(int(uids[i]), args.top_k)
        lat_ms.append((time.perf_counter() - s) * 1000.0)
    wall_s = time.perf_counter() - t_wall0

    lat_ms.sort()

    def pct(p):
        if not lat_ms:
            return float("nan")
        k = min(len(lat_ms) - 1, max(0, int(round(p / 100.0 * (len(lat_ms) - 1)))))
        return round(lat_ms[k], 4)

    qps = round(args.iters / wall_s, 1) if wall_s > 0 else float("nan")
    peak_rss_mb = max(peak_rss_mb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)

    # 诚实性自检：
    # - micro 档是**唯一**的在线部署形态，其路径必须零 torch（权重已导出为 numpy）。
    # - lite / full 档的引擎会**现场训练** DeepFM（src/serve/engine_lite.py:47
    #   导入 src.rank.trainer，后者 import torch），因此必然加载 torch。
    #   它们属于「训练形态」而非「在线轻量形态」，冷启动里包含训练成本——
    #   这恰恰是「为什么在线要选 micro」的证据，不该被当成错误拒绝。
    torch_loaded = "torch" in sys.modules
    if args.mode == "micro" and torch_loaded:
        raise SystemExit("✗ micro 在线路径意外加载了 torch（应纯 numpy），基准无效")
    if args.mode != "micro" and not torch_loaded:
        raise SystemExit(
            f"✗ {args.mode} 档应包含 torch（引擎需现场训练 DeepFM），"
            "未加载说明引擎选择有误，基准无效")

    result = {
        "mode": args.mode,
        "torch_loaded": torch_loaded,
        "deployment_form": ("在线部署形态（权重预导出，零 torch）" if args.mode == "micro"
                            else "训练形态（冷启动含 DeepFM 现场训练，需 torch）"),
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu": _cpu_model(),
            "platform": platform.platform(),
        },
        "cold_start": {
            "import_s": import_s,
            "engine_init_s": round(init_s, 4),
            "note": ("engine_init = 数据即时生成 + numpy 权重加载；"
                     "import = 子进程纯导入在线引擎模块"
                     if args.mode == "micro" else
                     "engine_init 含 DeepFM 现场训练成本，故远高于 micro 档；"
                     "import = 子进程导入引擎模块（会连带加载 torch）"),
        },
        "latency_ms": {
            "n": args.iters,
            "top_k": args.top_k,
            "mean": round(statistics.fmean(lat_ms), 4) if lat_ms else None,
            "p50": pct(50), "p95": pct(95), "p99": pct(99),
            "min": round(lat_ms[0], 4) if lat_ms else None,
            "max": round(lat_ms[-1], 4) if lat_ms else None,
            "stdev": round(statistics.pstdev(lat_ms), 4) if lat_ms else None,
        },
        "throughput": {"qps_single_thread": qps, "wall_s": round(wall_s, 4)},
        "memory": {"peak_rss_mb": round(peak_rss_mb, 1)},
        "note": "单进程单线程串行；QPS = 1/平均延迟，仅作数量级参考；对机器敏感。",
    }

    out = _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 78)
    print(f"在线推理性能基准 · mode={args.mode}")
    print("-" * 78)
    print(f"CPU         : {result['env']['cpu']}")
    print(f"Python      : {result['env']['python']}   numpy {result['env']['numpy']}")
    print("-" * 78)
    print(f"冷启动  import(子进程)      : {import_s:8.3f} s")
    print(f"冷启动  engine 构造         : {init_s:8.3f} s  (数据生成 + 权重加载)")
    print(f"内存    峰值 RSS            : {peak_rss_mb:8.1f} MB")
    print("-" * 78)
    print(f"延迟({args.iters} 次, Top-{args.top_k})  "
          f"P50 {pct(50):.3f} ms · P95 {pct(95):.3f} ms · P99 {pct(99):.3f} ms · "
          f"mean {result['latency_ms']['mean']:.3f} ms")
    print(f"吞吐    单线程 QPS          : {qps:8.1f}  (=1/mean)")
    print("=" * 78)
    print(f"结果已写入：{out}")


if __name__ == "__main__":
    main()
