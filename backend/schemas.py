"""Strict Pydantic v2 request validation schemas for all NeuroStock API endpoints."""

from datetime import date, datetime, timedelta
import math
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.config import ALGORITHMS, STOCK_LIST
from backend.security.key_vault import validate_nvidia_api_key
from backend.security.sanitizers import sanitize_prompt_text

_ALLOWED_ALGO_SET = frozenset(ALGORITHMS)


def _ensure_finite_float(val: float, field_name: str) -> float:
    num = float(val)
    if not math.isfinite(num):
        raise ValueError(f"{field_name} must be a finite number.")
    return num


def _parse_iso_date(raw: Union[str, date], field_name: str) -> date:
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    if not isinstance(raw, str) or len(raw.strip()) != 10:
        raise ValueError(f"{field_name} must be formatted as YYYY-MM-DD.")
    try:
        return datetime.strptime(raw.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid YYYY-MM-DD date.") from exc


def _validate_stock_label(raw: str, field_name: str = "stock_name") -> str:
    cleaned = sanitize_prompt_text(raw, max_len=80)
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty.")
    return cleaned


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stock_name: str = Field(..., min_length=1, max_length=80)
    future_date: date
    algorithm_name: Union[str, list[str]]

    @field_validator("stock_name")
    @classmethod
    def check_stock_name(cls, v: str) -> str:
        cleaned = _validate_stock_label(v)
        if cleaned not in STOCK_LIST:
            raise ValueError("Invalid stock_name")
        return cleaned

    @field_validator("future_date", mode="before")
    @classmethod
    def check_future_date(cls, v: Union[str, date]) -> date:
        dt = _parse_iso_date(v, "future_date")
        today = datetime.now().date()
        if dt < today - timedelta(days=7):
            raise ValueError("future_date cannot be in the distant past.")
        if dt > today + timedelta(days=366 * 5):
            raise ValueError("future_date cannot exceed 5 years into the future.")
        return dt

    @field_validator("algorithm_name")
    @classmethod
    def check_algorithms(cls, v: Union[str, list[str]]) -> Union[str, list[str]]:
        if isinstance(v, str):
            algo = v.strip()
            if algo not in _ALLOWED_ALGO_SET:
                raise ValueError("Invalid algorithm_name")
            return algo
        if not isinstance(v, list) or len(v) == 0 or len(v) > len(ALGORITHMS):
            raise ValueError(f"algorithm_name list must contain 1 to {len(ALGORITHMS)} items.")
        deduped: list[str] = []
        for item in v:
            if not isinstance(item, str) or item.strip() not in _ALLOWED_ALGO_SET:
                raise ValueError(f"Invalid algorithm: {item}")
            algo = item.strip()
            if algo not in deduped:
                deduped.append(algo)
        return deduped


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["top_n_long", "long_short_quintile", "single_stock"] = "top_n_long"
    horizon: Literal[1, 5, 10, 21, 63] = 21
    algorithm: str = "Ensemble"
    top_n: int = Field(default=10, ge=2, le=30)
    cost_bps: float = Field(default=15.0, ge=0.0, le=200.0)
    initial_capital: float = Field(default=100000.0, ge=1000.0, le=1e11)
    lookback_months: int = Field(default=24, ge=0, le=120)
    stock_name: str | None = Field(default=None, max_length=80)

    @field_validator("algorithm")
    @classmethod
    def check_algo(cls, v: str) -> str:
        cleaned = v.strip()
        if cleaned not in _ALLOWED_ALGO_SET:
            raise ValueError("Invalid algorithm")
        return cleaned

    @field_validator("cost_bps", "initial_capital")
    @classmethod
    def check_finite(cls, v: float, info) -> float:
        return _ensure_finite_float(v, info.field_name)

    @field_validator("stock_name")
    @classmethod
    def check_optional_stock(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = sanitize_prompt_text(v, max_len=80)
        return cleaned or None


class PortfolioProjectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stock_name: str = Field(..., min_length=1, max_length=80)
    principal: float = Field(..., ge=100.0, le=1e11)
    future_date: date

    @field_validator("stock_name")
    @classmethod
    def check_stock(cls, v: str) -> str:
        return _validate_stock_label(v)

    @field_validator("principal")
    @classmethod
    def check_principal(cls, v: float) -> float:
        return _ensure_finite_float(v, "principal")

    @field_validator("future_date", mode="before")
    @classmethod
    def check_future_date(cls, v: Union[str, date]) -> date:
        dt = _parse_iso_date(v, "future_date")
        today = datetime.now().date()
        if dt <= today:
            raise ValueError("Future date must be in the future.")
        if dt > today + timedelta(days=366 * 5):
            raise ValueError("Future date cannot exceed 5 years.")
        return dt


class OptimizePortfolioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stocks: list[str] = Field(..., min_length=2, max_length=10)
    principal: float = Field(..., ge=100.0, le=1e11)
    future_date: date
    mode: Literal["ai_bl", "min_var", "risk_parity", "custom"] = "ai_bl"
    custom_weights: list[float] | None = None

    @field_validator("stocks")
    @classmethod
    def check_stocks(cls, v: list[str]) -> list[str]:
        return [_validate_stock_label(item, "stocks[]") for item in v]

    @field_validator("principal")
    @classmethod
    def check_principal(cls, v: float) -> float:
        return _ensure_finite_float(v, "principal")

    @field_validator("future_date", mode="before")
    @classmethod
    def check_future_date(cls, v: Union[str, date]) -> date:
        dt = _parse_iso_date(v, "future_date")
        today = datetime.now().date()
        if dt <= today:
            raise ValueError("Future date must be in the future.")
        if dt > today + timedelta(days=366 * 5):
            raise ValueError("Future date cannot exceed 5 years.")
        return dt

    @model_validator(mode="after")
    def check_custom_weights(self) -> "OptimizePortfolioRequest":
        if self.custom_weights is not None:
            if len(self.custom_weights) != len(self.stocks):
                raise ValueError("custom_weights length must match stocks length.")
            validated = []
            for w in self.custom_weights:
                fw = _ensure_finite_float(w, "custom_weights[]")
                if fw < 0.0 or fw > 1000.0:
                    raise ValueError("custom_weights entries must be between 0 and 1000.")
                validated.append(fw)
            self.custom_weights = validated
        return self


class InvestmentEngineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principal: float = Field(default=100000.0, ge=100.0, le=1e11)
    withdrawal_date: date

    @field_validator("principal")
    @classmethod
    def check_principal(cls, v: float) -> float:
        return _ensure_finite_float(v, "principal")

    @field_validator("withdrawal_date", mode="before")
    @classmethod
    def check_withdrawal_date(cls, v: Union[str, date]) -> date:
        dt = _parse_iso_date(v, "withdrawal_date")
        today = datetime.now().date()
        if dt <= today:
            raise ValueError("withdrawal_date must be in the future.")
        if dt > today + timedelta(days=366 * 5):
            raise ValueError("withdrawal_date cannot exceed 5 years.")
        return dt


class SentimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stock_name: str = Field(..., min_length=1, max_length=80)

    @field_validator("stock_name")
    @classmethod
    def check_stock(cls, v: str) -> str:
        return _validate_stock_label(v)


class GeneralKnowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_question: str = Field(..., min_length=1, max_length=1000)

    @field_validator("user_question")
    @classmethod
    def check_question(cls, v: str) -> str:
        cleaned = sanitize_prompt_text(v, max_len=1000)
        if not cleaned:
            raise ValueError("Missing user_question")
        return cleaned


class ActiveContextSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tab: str | None = Field(default=None, max_length=40)
    stock_name: str | None = Field(default=None, max_length=80)
    forecastSummary: str | None = Field(default=None, max_length=300)


class TrendDataSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dates: list[str] = Field(..., max_length=400)
    prices: list[float] = Field(..., max_length=400)
    history_cutoff: int = Field(default=100, ge=0, le=400)
    upper_band: list[float | None] | None = Field(default=None, max_length=400)
    lower_band: list[float | None] | None = Field(default=None, max_length=400)

    @field_validator("prices")
    @classmethod
    def check_prices(cls, v: list[float]) -> list[float]:
        return [_ensure_finite_float(p, "trend_data.prices[]") for p in v]


class AiInsightsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stock_name: str | None = Field(default=None, max_length=80)
    trend_data: TrendDataSchema | None = None
    user_question: str | None = Field(default=None, max_length=1000)
    active_context: ActiveContextSchema | None = None

    @field_validator("stock_name")
    @classmethod
    def check_stock_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = sanitize_prompt_text(v, max_len=80)
        return cleaned or None

    @field_validator("user_question")
    @classmethod
    def check_question(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = sanitize_prompt_text(v, max_len=1000)
        return cleaned or None

    @model_validator(mode="after")
    def require_any_context(self) -> "AiInsightsRequest":
        if not self.user_question and not self.stock_name and not self.active_context:
            raise ValueError("Missing user_question or stock context")
        return self


class LlmKeyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(..., min_length=20, max_length=256)

    @field_validator("api_key")
    @classmethod
    def check_api_key(cls, v: str) -> str:
        return validate_nvidia_api_key(v)
