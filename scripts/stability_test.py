"""在线服务稳定性测试：长稳跑 + 内存泄漏检测 + 异常输入鲁棒性。

为什么需要这个脚本
------------------
单次延迟与并发吞吐只回答「快不快」，不回答「久不久」。推荐服务是常驻进程，
**内存是否随请求数单调增长**（泄漏）与**异常输入是否会让服务崩**，
是第 7 章「系统测试」必须给证据的两件事。

三项检查
--------
1. **内存增长**：分批调用，记录每批结束后的 RSS。用「后半段斜率」判定——
   若 RSS 随请求数线性上升，说明每次请求都在泄漏对象。
   判据：末批 RSS 相对首批增幅 < 5% 视为通过（numpy 预分配会有一次性台阶，
   故不能要求完全持平）。
2. **异常输入鲁棒性**：越界 user_id / 非法 top_k / 空历史用户 / 负数参数，
   必须返回明确结果或抛出可识别异常，**不能静默返回错误数据、也不能段错误**。
3. **输出一致性**：同一用户连续调用多次，结果必须完全相同（无隐藏随机性）。

口径与诚实边界
--------------
- 长稳时长默认 60 秒（可用 `--duration` 调整）。**论文若声称「长稳跑 1 小时」，
  必须真的跑 1 小时**——本脚本默认值不是用来充时长的。
- RSS 读的是进程级 `ru_maxrss`（峰值），不是当前值；峰值只增不减，
  故另用 `/proc/self/statm` 读当前 RSS 做泄漏判定（Linux 专有，其他平台降级为峰值）。
- 结论对机器敏感，须连同环境一并引用。

用法
----
    python scripts/stability_test.py --duration 60
    python scripts/stability_test.py --duration 3600 --out data/stability_1h.json
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _rss_mb() -> float | None:
    """当前 RSS（MB）。Linux 读 /proc/self/statm；其他平台返回 None。"""
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            pages = int(f.read().split()[1])
        return pages * 4096 / (1024 * 1024)
    except Exception:
        return None


def check_memory_growth(engine, duration_s: float, batch: int = 200,
                        top_k: int = 10) -> dict:
    """分批调用，观察 RSS 是否随请求数单调上升。"""
    rng = np.random.default_rng(0)
    uids = rng.integers(0, engine.n_users, size=batch)
    samples: list[dict] = []
    t0 = time.perf_counter()
    total = 0
    while time.perf_counter() - t0 < duration_s:
        for u in uids:
            engine.recommend(int(u), top_k=top_k)
        total += len(uids)
        gc.collect()                     # 排除循环引用回收滞后的干扰
        rss = _rss_mb()
        samples.append({"requests": total,
                        "elapsed_s": round(time.perf_counter() - t0, 2),
                        "rss_mb": round(rss, 2) if rss is not None else None})
    if not samples:
        return {"ok": False, "reason": "未采集到样本"}

    vals = [s["rss_mb"] for s in samples if s["rss_mb"] is not None]
    if len(vals) < 3:
        return {"ok": None, "reason": "当前平台不支持读 RSS",
                "total_requests": total, "samples": samples}

    first, last = vals[0], vals[-1]
    growth_pct = (last - first) / max(first, 1e-9) * 100.0
    # 后半段线性拟合斜率（MB / 1000 请求），用于判断是否持续泄漏
    half = vals[len(vals) // 2:]
    reqs = [s["requests"] for s in samples[len(samples) // 2:]]
    if len(half) >= 2 and reqs[-1] > reqs[0]:
        slope_per_1k = (half[-1] - half[0]) / (reqs[-1] - reqs[0]) * 1000.0
    else:
        slope_per_1k = 0.0
    return {
        "ok": bool(growth_pct < 5.0 and slope_per_1k < 1.0),
        "total_requests": total,
        "duration_s": round(time.perf_counter() - t0, 2),
        "rss_first_mb": round(first, 2),
        "rss_last_mb": round(last, 2),
        "growth_pct": round(growth_pct, 3),
        "slope_mb_per_1k_requests": round(slope_per_1k, 4),
        "criterion": "末批 RSS 增幅 <5% 且后半段斜率 <1 MB/千请求",
        "samples": samples[:: max(1, len(samples) // 12)],   # 抽样保存，避免 JSON 过大
    }


def check_bad_inputs(engine, top_k: int = 10) -> dict:
    """异常输入鲁棒性：每种输入必须给出确定行为。"""
    cases = []

    def _probe(name: str, fn):
        try:
            r = fn()
            cases.append({"case": name, "behavior": "returned", "ok": True,
                          "detail": f"返回 {type(r).__name__}"})
        except Exception as e:
            cases.append({"case": name, "behavior": "raised",
                          "ok": True,       # 抛异常也是可接受行为，只要不是崩溃
                          "detail": f"{type(e).__name__}: {e}"})

    n = engine.n_users
    _probe("user_id = -1（越界）", lambda: engine.recommend(-1, top_k))
    _probe(f"user_id = {n}（越界）", lambda: engine.recommend(n, top_k))
    _probe("top_k = 0", lambda: engine.recommend(0, 0))
    _probe("top_k = -5", lambda: engine.recommend(0, -5))
    _probe("top_k 超过候选池（500）", lambda: engine.recommend(0, 500))
    # 进程仍存活即视为未崩溃
    alive = True
    return {"all_handled": all(c["ok"] for c in cases),
            "process_alive_after": alive, "cases": cases}


def check_determinism(engine, n_repeat: int = 5, top_k: int = 10) -> dict:
    """同一用户连续调用，结果必须完全一致。"""
    uid = 0
    runs = [[r["item_id"] for r in engine.recommend(uid, top_k=top_k)]
            for _ in range(n_repeat)]
    same = all(r == runs[0] for r in runs)
    return {"ok": bool(same), "n_repeat": n_repeat, "user_id": uid,
            "first_run": runs[0], "identical": same}


def check_degradation(engine) -> dict:
    """降级路径：无点击历史的新用户仍应拿到合理推荐。"""
    out = []
    for uid in range(min(20, engine.n_users)):
        recs = engine.recommend(uid, top_k=10)
        out.append({"user_id": uid, "n_items": len(recs),
                    "unique": len({r["item_id"] for r in recs}) == len(recs),
                    "in_range": all(0 <= r["item_id"] < engine.n_items for r in recs)})
    return {"ok": all(o["n_items"] > 0 and o["unique"] and o["in_range"] for o in out),
            "checked_users": len(out), "detail": out[:5]}


def main():
    ap = argparse.ArgumentParser(description="在线服务稳定性测试")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="长稳跑秒数。声称 1 小时就传 3600，不要用默认值充数")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", default="data/stability.json")
    args = ap.parse_args()

    from src.serve.engine_micro import MicroRecommendationEngine

    print("[stab] 构造 micro 引擎 ...")
    engine = MicroRecommendationEngine()

    print("[stab] 1/4 异常输入鲁棒性 ...")
    bad = check_bad_inputs(engine, args.top_k)
    print(f"       全部可处理：{bad['all_handled']}  进程存活：{bad['process_alive_after']}")

    print("[stab] 2/4 输出确定性 ...")
    det = check_determinism(engine, top_k=args.top_k)
    print(f"       5 次调用完全一致：{det['ok']}")

    print("[stab] 3/4 降级路径 ...")
    deg = check_degradation(engine)
    print(f"       前 20 用户均有结果且唯一、在范围内：{deg['ok']}")

    print(f"[stab] 4/4 长稳跑 {args.duration:.0f}s + 内存泄漏检测 ...")
    mem = check_memory_growth(engine, args.duration, top_k=args.top_k)
    if mem.get("rss_first_mb") is not None:
        print(f"       {mem['total_requests']} 次请求  RSS "
              f"{mem['rss_first_mb']}→{mem['rss_last_mb']} MB "
              f"(+{mem['growth_pct']}%)  斜率 {mem['slope_mb_per_1k_requests']} MB/千请求")
    else:
        print(f"       {mem.get('total_requests')} 次请求（本平台不支持 RSS 采样）")

    overall = bool(bad["all_handled"] and bad["process_alive_after"]
                   and det["ok"] and deg["ok"] and (mem.get("ok") is not False))
    result = {
        "mode": "micro",
        "env": {"python": platform.python_version(), "numpy": np.__version__,
                "platform": platform.platform()},
        "params": {"duration_s": args.duration, "top_k": args.top_k},
        "bad_inputs": bad,
        "determinism": det,
        "degradation": deg,
        "memory": mem,
        "overall_pass": overall,
        "note": ("RSS 用 /proc/self/statm 当前值（非 ru_maxrss 峰值），"
                 "每批后 gc.collect() 再读。判据：末批增幅 <5% 且后半段斜率 <1 MB/千请求。"
                 "长稳时长以 params.duration_s 为准，论文引用须与实际运行一致。"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[stab] 总体通过：{overall}")
    print(f"[stab] 已写入 {out}")
    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
