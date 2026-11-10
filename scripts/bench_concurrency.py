"""在线服务并发压测：多线程 QPS / 延迟分位 / 错误率。

为什么需要这个脚本
------------------
`scripts/bench_inference.py` 测的是**单线程串行**延迟——它回答「一次请求多贵」，
但不回答「服务能同时扛多少请求」。第 7 章「系统测试」若只有串行延迟，
评委会问「并发下会怎样」，而这是推荐服务最现实的运行条件。

口径与诚实边界
--------------
- 压测对象是**在线推理引擎**（不含 HTTP 层），因此数值反映引擎本身的并发能力，
  不含网络与 FastAPI 序列化开销。要测 HTTP 层请用 `--http`（未实现，见下）。
- Python 有 GIL：纯 numpy 推理的并发收益主要来自 numpy 内部释放 GIL 的部分
  （矩阵乘、元素级运算）与 I/O。本脚本如实报告**实测加速比**，不假设线性扩展。
- 结论对机器敏感：自动记录 CPU / Python / numpy 版本，换机器数值会变。
- 错误率必须为 0 才算通过；任何异常都会计数并记录首个堆栈。

用法
----
    python scripts/bench_concurrency.py                  # micro 档，默认 8 线程
    python scripts/bench_concurrency.py --threads 1 2 4 8 16
    python scripts/bench_concurrency.py --requests-per-thread 200

产出
----
    data/bench_concurrency.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def run_level(engine, n_threads: int, requests_per_thread: int,
              user_ids: np.ndarray, top_k: int) -> dict:
    """在指定线程数下压测，返回该档位的统计结果。"""
    latencies: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(n_threads)

    def worker(tid: int):
        # 每个线程用不同的用户偏移，避免命中同一批缓存行
        offset = tid * 7919
        local: list[float] = []
        try:
            barrier.wait()                      # 让所有线程尽量同时起跑
            for i in range(requests_per_thread):
                uid = int(user_ids[(offset + i) % len(user_ids)])
                t0 = time.perf_counter()
                engine.recommend(uid, top_k=top_k)
                local.append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            with lock:
                errors.append(traceback.format_exc(limit=3))
        with lock:
            latencies.extend(local)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start

    s = sorted(latencies)
    total = len(s)
    return {
        "n_threads": int(n_threads),
        "requests_total": total,
        "wall_s": round(wall, 4),
        "qps": round(total / wall, 1) if wall > 0 else 0.0,
        "error_count": len(errors),
        "error_rate": round(len(errors) / max(total + len(errors), 1), 6),
        "first_error": errors[0] if errors else None,
        "latency_ms": {
            "mean": round(float(np.mean(s)), 4) if s else 0.0,
            "p50": round(_percentile(s, 0.50), 4),
            "p95": round(_percentile(s, 0.95), 4),
            "p99": round(_percentile(s, 0.99), 4),
            "max": round(s[-1], 4) if s else 0.0,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="在线服务并发压测")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--requests-per-thread", type=int, default=150)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default="data/bench_concurrency.json")
    args = ap.parse_args()

    from src.serve.engine_micro import MicroRecommendationEngine

    print("[conc] 构造 micro 引擎 ...")
    t0 = time.time()
    engine = MicroRecommendationEngine()
    print(f"[conc] 引擎就绪 {time.time() - t0:.1f}s")

    # 用户选择必须与 `scripts/bench_inference.py` **同口径**（随机而非连续），
    # 否则两份基准不可对照：实测连续用户 0-499 的 mean 延迟是 1.9ms，
    # 而随机用户的串行基准是 0.73ms——差异来自「有无点击历史」（无历史者走更快的降级路径），
    # 不是并发本身造成的。
    rng = np.random.default_rng(0)
    user_pool = rng.integers(0, engine.n_users, size=2000)

    # warm-up：必须**覆盖压测会遍历到的全部用户**。
    # 早前版本只 warm 了前 20 个用户，导致 1 线程档 QPS 只有 59（串行基准是 1378），
    # 相差 23 倍——冷用户的首调开销（候选特征缓存未建立）污染了整档统计。
    n_warm = max(200, args.threads[-1] * args.requests_per_thread)
    warm_users = user_pool[:min(len(user_pool), n_warm)]
    for u in warm_users:
        engine.recommend(int(u), top_k=args.top_k)
    print(f"[conc] warm-up {len(warm_users)} 个用户（随机口径，与串行基准一致）")

    rows = []
    for nt in args.threads:
        r = run_level(engine, nt, args.requests_per_thread, user_pool, args.top_k)
        rows.append(r)
        print(f"[conc] {nt:>2} 线程  QPS={r['qps']:>8.1f}  "
              f"p50={r['latency_ms']['p50']:>7.3f}ms  "
              f"p99={r['latency_ms']['p99']:>7.3f}ms  "
              f"错误={r['error_count']}")

    base_qps = rows[0]["qps"] if rows else 0.0
    for r in rows:
        r["speedup_vs_1thread"] = (round(r["qps"] / base_qps, 2)
                                    if base_qps > 0 else None)

    result = {
        "mode": "micro",
        "torch_loaded": False,
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu": platform.processor() or "unknown",
            "platform": platform.platform(),
        },
        "params": {
            "threads": args.threads,
            "requests_per_thread": args.requests_per_thread,
            "top_k": args.top_k,
        },
        "levels": rows,
        "peak_qps": max((r["qps"] for r in rows), default=0.0),
        "peak_qps_threads": max(rows, key=lambda r: r["qps"])["n_threads"] if rows else None,
        "all_errors_zero": all(r["error_count"] == 0 for r in rows),
        "note": ("单进程多线程（threading），测引擎推理路径，不含 HTTP 层。"
                 "Python GIL 存在，加速比来自 numpy 释放 GIL 的部分，不假设线性扩展。"
                 "对机器敏感，结论须连同 env 一并引用。"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[conc] 峰值 QPS={result['peak_qps']:.1f} @ {result['peak_qps_threads']} 线程")
    print(f"[conc] 全部档位零错误：{result['all_errors_zero']}")
    print(f"[conc] 已写入 {out}")


if __name__ == "__main__":
    main()
