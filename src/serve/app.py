"""FastAPI 服务：推荐 / 排序流程 / A/B 报告。

运行：uvicorn src.serve.app:app --host 0.0.0.0 --port 8000
"""
from functools import lru_cache

from fastapi import FastAPI

from src.serve.engine import RecommendationEngine


@lru_cache(maxsize=1)
def get_engine():
    return RecommendationEngine()


app = FastAPI(title="AeroRec 出行推荐中台", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend")
def recommend(user_id: int, top_k: int = 10):
    engine = get_engine()
    return {"user_id": user_id, "items": engine.recommend(user_id, top_k)}


@app.get("/explain")
def explain(user_id: int, top_k: int = 10):
    engine = get_engine()
    return engine.explain_steps(user_id, top_k)


@app.get("/ab")
def ab(n_users: int = 2000, top_k: int = 10):
    engine = get_engine()
    return engine.run_ab(n_users=n_users, top_k=top_k)
