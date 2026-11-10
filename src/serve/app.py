"""FastAPI 服务：推荐 / 排序流程 / A/B 报告。

运行：uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

错误处理约定
------------
- 400 ：调用方参数不合法（user_id 越界 / top_k 非法）；
- 503 ：引擎未就绪或冷启动初始化失败；
- 500 ：其余未预期异常——服务端记录完整栈（便于排查），
        客户端只拿到通用信息，**不泄露内部实现细节**。
"""
import traceback
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from src.serve.engine import RecommendationEngine

# 候选池上限：candidates() 默认召回 200，超出该值的 top_k 无更多物品可返回
MAX_TOP_K = 200


@lru_cache(maxsize=1)
def get_engine():
    return RecommendationEngine()


app = FastAPI(title="AeroRec 出行推荐中台", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


def _checked(user_id: int, top_k: int):
    """校验请求参数并返回就绪的引擎；不合法立即抛 4xx，冷启动失败抛 503。"""
    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(status_code=400,
                            detail=f"top_k 必须在 1..{MAX_TOP_K}，收到 {top_k}")
    try:
        engine = get_engine()
    except Exception:                       # 冷启动失败：不向调用方暴露内部栈
        traceback.print_exc()
        raise HTTPException(status_code=503, detail="引擎暂不可用，请稍后重试")
    n_users = len(engine.users)
    if not (0 <= user_id < n_users):
        raise HTTPException(status_code=400,
                            detail=f"user_id 必须在 0..{n_users - 1}，收到 {user_id}")
    return engine


@app.get("/recommend")
def recommend(user_id: int, top_k: int = 10):
    engine = _checked(user_id, top_k)
    try:
        return {"user_id": user_id, "items": engine.recommend(user_id, top_k)}
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部错误，请稍后重试")


@app.get("/explain")
def explain(user_id: int, top_k: int = 10):
    engine = _checked(user_id, top_k)
    try:
        return engine.explain_steps(user_id, top_k)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部错误，请稍后重试")


@app.get("/ab")
def ab(n_users: int = 2000, top_k: int = 10):
    if n_users < 1:
        raise HTTPException(status_code=400, detail=f"n_users 必须 ≥ 1，收到 {n_users}")
    engine = _checked(user_id=0, top_k=top_k)
    try:
        return engine.run_ab(n_users=n_users, top_k=top_k)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部错误，请稍后重试")


@app.get("/bias-guard")
def bias_guard(lam: float = 1.0):
    """去偏判据：只看日志，回答「当前该不该启用 IPS/SNIPS 去偏」。

    这是本项目判据的**在线落地形态**——判据不是离线报告，而是随服务可查的开关信号。
    返回 ESS/N（**记录级口径**）、三档建议与 `should_debias` 布尔开关。

    ⚠️ 口径说明：ESS/N 必须与口径一起看。同一份日志在候选池级口径下可能
    给出与记录级口径**相反**的结论（实测 0.9907 vs 0.1129）。
    本接口固定用记录级口径（与合成标定 `data/bias/boundary.json` 同口径），
    并在响应的 `caliber` 字段中显式声明。
    """
    if lam < 0:
        raise HTTPException(status_code=400, detail=f"lam 必须 ≥ 0，收到 {lam}")
    engine = _checked(user_id=0, top_k=10)
    try:
        return engine.bias_guard(lam=lam)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="内部错误，请稍后重试")
