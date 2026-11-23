"""Async concurrency, singleflight coalescing, RSI regression, and API contract tests."""

import asyncio
from datetime import datetime, timedelta
import time

import httpx
import numpy as np
import pandas as pd
import pytest

from backend.app import create_app
from backend.security.auth import current_user_id_ctx
from backend.security.key_vault import key_vault
from backend.services.ai_assistant_service import _compute_stock_dossier_quant_sync
from backend.services.alpha_radar_service import _RADAR_SINGLEFLIGHT, get_alpha_radar_async
from backend.services.feature_engineering import BASE_FEATURE_COLUMNS, add_cross_sectional_features
from backend.services.llm_client import async_chat_completion
from backend.services.runtime import AsyncSingleflightCache


@pytest.mark.asyncio
async def test_async_singleflight_coalesces_concurrent_calls():
    cache = AsyncSingleflightCache(maxsize=10, default_ttl=60.0)
    calls = 0

    async def expensive():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.08)
        return {"ok": calls}

    results = await asyncio.gather(*(cache.get_or_compute("k1", expensive) for _ in range(10)))
    assert calls == 1
    assert all(res == {"ok": 1} for res in results)


@pytest.mark.asyncio
async def test_async_server_processes_concurrent_requests_non_blocking(rsa_auth_env, monkeypatch):
    token = rsa_auth_env["sign_token"]()

    async def slow_tutor(q: str):
        await asyncio.sleep(0.25)
        return f"Answer for: {q}"

    monkeypatch.setattr("backend.routes.ai_routes.answer_general_knowledge_async", slow_tutor)
    app = create_app(load_models=False)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        t0 = time.perf_counter()
        tasks = [
            client.post(
                "/get-general-knowledge",
                json={"user_question": f"What is Sharpe ratio #{i}?"},
                headers={"Authorization": f"Bearer {token}"},
            )
            for i in range(4)
        ]
        tasks.append(client.get("/health"))
        responses = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    # Serial execution of 4x 0.25s sleeps would take >= 1.0s; parallel async takes ~0.25-0.45s
    assert elapsed < 0.75
    for resp in responses[:4]:
        assert resp.status_code == 200
        assert "Answer for:" in resp.json()["answer"]
    assert responses[4].status_code == 200


@pytest.mark.asyncio
async def test_alpha_radar_refresh_cooldown_prevents_cache_bust_dos(monkeypatch):
    _RADAR_SINGLEFLIGHT.clear()
    fetch_count = 0

    async def fake_panel(force_refresh=False):
        nonlocal fetch_count
        fetch_count += 1
        return pd.DataFrame(), {}, pd.DataFrame()

    monkeypatch.setattr("backend.services.alpha_radar_service.get_live_universe_panel_async", fake_panel)

    await get_alpha_radar_async(horizon=21, force_refresh=True)
    await get_alpha_radar_async(horizon=21, force_refresh=True)
    assert fetch_count == 1
    _RADAR_SINGLEFLIGHT.clear()


def test_all_api_endpoints_happy_path_contracts(auth_client, monkeypatch):
    client, env = auth_client
    token = env["sign_token"]()
    headers = {"Authorization": f"Bearer {token}"}
    tomorrow = (datetime.now().date() + timedelta(days=5)).strftime("%Y-%m-%d")

    async def fake_detail(ticker):
        return {"info": {"symbol": ticker, "currentPrice": 1500.0}, "chart": [{"time": "2026-11-20", "value": 1500.0}]}

    async def fake_radar(horizon=21, force_refresh=False):
        return {"requested_horizon": horizon, "active_horizon": horizon, "stocks": [], "sectors": []}

    async def fake_predict(stock_name, ticker, future_date_str, algorithm_name):
        if isinstance(algorithm_name, list):
            return {a: {"stock_name": stock_name, "algorithm_name": a, "predicted_price": "1550.00"} for a in algorithm_name}, None, 200
        return {"stock_name": stock_name, "ticker": ticker, "algorithm_name": algorithm_name, "predicted_price": "1550.00"}, None, 200

    async def fake_hist(ticker):
        return {"ohlc": [{"time": "2026-11-20", "open": 1, "high": 2, "low": 1, "close": 1.5}], "volume": [], "rsi": [], "sma": []}

    async def fake_backtest(**kwargs):
        return {"strategy": kwargs["strategy"], "metrics": {"final_capital": 120000}}, None

    async def fake_proj(stock_name, principal, future_date_str):
        return {"stock": stock_name, "final_value": principal * 1.1}, None

    async def fake_opt(stocks, principal, future_date_str, mode="ai_bl", custom_weights=None):
        return {"mode": mode, "final_value": principal * 1.12, "allocations": []}, None

    async def fake_engine(principal, withdrawal_date_str):
        return {"top_stocks": [], "ai_advice": "Hold cash."}

    async def fake_sent(display_name, ticker):
        return {"score": 72, "label": "Bullish", "summary": "Positive momentum.", "news": []}

    async def fake_insights(**kwargs):
        return "Sample insight", 200

    async def fake_gk(q):
        return "CAGR stands for Compound Annual Growth Rate."

    monkeypatch.setattr("backend.routes.market_routes.get_stock_detail_async", fake_detail)
    monkeypatch.setattr("backend.routes.market_routes.get_alpha_radar_async", fake_radar)
    monkeypatch.setattr("backend.routes.prediction_routes.predict_async", fake_predict)
    monkeypatch.setattr("backend.routes.prediction_routes.get_historical_chart_data_async", fake_hist)
    monkeypatch.setattr("backend.routes.prediction_routes.run_strategy_backtest_async", fake_backtest)
    monkeypatch.setattr("backend.routes.portfolio_routes.calculate_portfolio_projection_async", fake_proj)
    monkeypatch.setattr("backend.routes.portfolio_routes.optimize_multi_stock_portfolio_async", fake_opt)
    monkeypatch.setattr("backend.routes.portfolio_routes.run_investment_engine_async", fake_engine)
    monkeypatch.setattr("backend.routes.sentiment_routes.analyze_stock_sentiment_async", fake_sent)
    monkeypatch.setattr("backend.routes.ai_routes.generate_ai_insights_async", fake_insights)
    monkeypatch.setattr("backend.routes.ai_routes.answer_general_knowledge_async", fake_gk)

    assert client.get("/get-market-data?ticker=INFY.NS", headers=headers).status_code == 200
    assert client.get("/alpha-radar?horizon=21", headers=headers).status_code == 200
    assert client.post("/predict", json={"stock_name": "Infosys", "future_date": tomorrow, "algorithm_name": "Ensemble"}, headers=headers).status_code == 200
    assert client.post("/predict", json={"stock_name": "Infosys", "future_date": tomorrow, "algorithm_name": ["Ensemble", "LightGBM"]}, headers=headers).status_code == 200
    assert client.get("/historical-data?stock_name=Infosys", headers=headers).status_code == 200
    assert client.post("/run-backtest", json={"strategy": "top_n_long", "horizon": 21}, headers=headers).status_code == 200
    assert client.post("/calculate-portfolio", json={"stock_name": "Infosys", "principal": 50000, "future_date": tomorrow}, headers=headers).status_code == 200
    assert client.post("/optimize-portfolio", json={"stocks": ["Infosys", "TCS"], "principal": 50000, "future_date": tomorrow}, headers=headers).status_code == 200
    assert client.post("/investment-engine", json={"principal": 50000, "withdrawal_date": tomorrow}, headers=headers).status_code == 200
    assert client.post("/get-sentiment", json={"stock_name": "Infosys"}, headers=headers).status_code == 200
    assert client.post("/get-ai-insights", json={"user_question": "Analyze Infosys"}, headers=headers).status_code == 200
    assert client.post("/get-general-knowledge", json={"user_question": "What is CAGR?"}, headers=headers).status_code == 200


def test_rsi14_scaling_in_ai_dossier():
    dates = pd.bdate_range("2026-01-01", periods=65)
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["INFY.NS"] * 65,
            "Close": np.linspace(1500.0, 1600.0, 65),
            "ret_5d": np.full(65, 0.01),
            "ret_21d": np.full(65, 0.03),
            "vol_21d": np.full(65, 0.015),
            "rsi_14": np.full(65, 0.625),
            "cs_rank_ret_21d": np.full(65, 0.25),
            "sector_excess_ret_21d": np.full(65, 0.02),
            "vix_level": np.full(65, 0.70),
        }
    )
    dossier = _compute_stock_dossier_quant_sync("Infosys", "INFY.NS", df)
    assert dossier["rsi_14"] == 62.5


def test_vectorized_cross_sectional_features_consistency():
    dates = pd.bdate_range("2026-01-01", periods=5)
    rows = []
    for d_idx, dt in enumerate(dates):
        for s_idx, (t, sec) in enumerate([("INFY.NS", "IT"), ("TCS.NS", "IT"), ("HDFCBANK.NS", "Financials")]):
            row = {"date": dt, "ticker": t, "sector": sec}
            for col in BASE_FEATURE_COLUMNS:
                row[col] = 0.01 * (s_idx + 1) + 0.002 * d_idx
            rows.append(row)

    panel = pd.DataFrame(rows)
    enriched = add_cross_sectional_features(panel)
    assert not enriched.isna().any().any()
    assert enriched["cs_rank_ret_21d"].min() >= -0.5
    assert enriched["cs_rank_ret_21d"].max() <= 0.5


@pytest.mark.asyncio
async def test_llm_client_prefers_lm_studio_and_falls_back_to_nvidia(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_KEY_ENCRYPTION_SECRET", "async-failover-test-secret")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(key_vault, "_explicit_vault_path", str(tmp_path / "vault.json"))
    key_vault.save_user_api_key("user-alice-123", "nvapi-TestFailoverKey9876543210")

    calls = []

    class FakeResp:
        def __init__(self, text: str) -> None:
            self._text = text

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": self._text}}]}

    async def fake_post_lm_ok(url, payload, headers, timeout):
        calls.append((url, payload, headers))
        return FakeResp("Local LM Studio response")

    monkeypatch.setattr("backend.services.llm_client._post_json", fake_post_lm_ok)
    token = current_user_id_ctx.set("user-alice-123")
    try:
        res_local = await async_chat_completion([{"role": "user", "content": "Hello"}])
        assert res_local == "Local LM Studio response"
        assert len(calls) == 1
        assert calls[0][0].startswith("http://127.0.0.1:1234")

        calls.clear()

        async def fake_post_lm_down_nvidia_ok(url, payload, headers, timeout):
            calls.append((url, payload, headers))
            if "127.0.0.1:1234" in url:
                raise httpx.ConnectError("LM Studio offline")
            return FakeResp("NVIDIA Nemotron response")

        monkeypatch.setattr("backend.services.llm_client._post_json", fake_post_lm_down_nvidia_ok)
        res_nvidia = await async_chat_completion([{"role": "user", "content": "Hello"}])
        assert res_nvidia == "NVIDIA Nemotron response"
        assert len(calls) == 2
        assert "127.0.0.1:1234" in calls[0][0]
        assert calls[1][0] == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert calls[1][1]["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
        assert calls[1][2]["Authorization"] == "Bearer nvapi-TestFailoverKey9876543210"
    finally:
        current_user_id_ctx.reset(token)
