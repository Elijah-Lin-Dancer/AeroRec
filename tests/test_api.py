"""FastAPI 端点测试：用桩引擎（stub）替代真实引擎，仅验证参数校验与错误处理。

真实引擎加载 torch / sklearn 且冷启动耗时长，不适合单测；
这里 monkeypatch `get_engine`，聚焦 app 层的错误处理约定：
  400 参数不合法 / 503 引擎不可用 / 500 内部错误（不泄露内部栈）。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from src.serve import app as app_module


class _StubEngine:
    """与 RecommendationEngine 对外接口同构的最小桩。"""

    users = list(range(5))          # _checked 用 len(engine.users) 校验 user_id 范围

    def recommend(self, user_id, top_k=None):
        return [{"rank": i + 1, "item_id": i, "name": f"x-{i}", "category": "c",
                 "base_score": 0.1, "ctr": 0.2, "cvr": 0.3} for i in range(top_k)]

    def explain_steps(self, user_id, top_k=None, show: int = 20):
        return {"user_id": user_id, "top_k": top_k}

    def run_ab(self, n_users=1000, top_k=None):
        return {"n_users": n_users, "top_k": top_k}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(app_module, "get_engine", lambda: _StubEngine())
    # raise_server_exceptions=False：让未处理异常返回 500 响应而非在测试端直接抛出
    return TestClient(app_module.app, raise_server_exceptions=False)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_recommend_ok(client):
    r = client.get("/recommend", params={"user_id": 0, "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == 0
    assert len(body["items"]) == 3


def test_recommend_rejects_bad_top_k(client):
    assert client.get("/recommend", params={"user_id": 0, "top_k": 0}).status_code == 400
    assert client.get("/recommend", params={"user_id": 0, "top_k": 999}).status_code == 400


def test_recommend_rejects_out_of_range_user(client):
    r = client.get("/recommend", params={"user_id": 99})
    assert r.status_code == 400
    assert "user_id" in r.json()["detail"]


def test_explain_and_ab_ok(client):
    assert client.get("/explain", params={"user_id": 1, "top_k": 2}).status_code == 200
    r = client.get("/ab", params={"n_users": 10, "top_k": 3})
    assert r.status_code == 200
    assert r.json() == {"n_users": 10, "top_k": 3}


def test_ab_rejects_non_positive_n_users(client):
    assert client.get("/ab", params={"n_users": 0}).status_code == 400


def test_engine_cold_start_failure_returns_503(client, monkeypatch):
    def _boom():
        raise RuntimeError("cold start failed")

    monkeypatch.setattr(app_module, "get_engine", _boom)
    assert client.get("/recommend", params={"user_id": 0}).status_code == 503


def test_internal_error_returns_500_without_leaking_stack(client, monkeypatch):
    eng = _StubEngine()

    def _boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    eng.recommend = _boom
    monkeypatch.setattr(app_module, "get_engine", lambda: eng)
    r = client.get("/recommend", params={"user_id": 0})
    assert r.status_code == 500
    assert "secret internal detail" not in r.text       # 不向调用方泄露内部栈
