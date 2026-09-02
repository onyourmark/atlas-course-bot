"""Supported AI providers and models for faculty-managed ATLAS courses."""

from __future__ import annotations

from typing import Dict, List


PROVIDER_LABELS = {
    "anthropic": "Anthropic (Claude models)",
    "openai": "OpenAI (ChatGPT models)",
}


MODEL_OPTIONS: Dict[str, List[Dict[str, str]]] = {
    "anthropic": [
        {
            "id": "claude-haiku-4-5-20251001",
            "name": "Claude Haiku 4.5",
            "cost": "Lowest Anthropic cost",
            "description": "Fast and suitable when keeping cost low is the priority.",
        },
        {
            "id": "claude-sonnet-5",
            "name": "Claude Sonnet 5",
            "cost": "Lower cost",
            "description": "A good balance of answer quality, speed, and cost.",
        },
        {
            "id": "claude-opus-5",
            "name": "Claude Opus 5",
            "cost": "Higher cost",
            "description": "For courses that need stronger reasoning and can accept a higher cost.",
        },
        {
            "id": "claude-fable-5-1",
            "name": "Claude Fable 5.1",
            "cost": "Highest Anthropic cost",
            "description": "For the most demanding questions and the highest cost tier.",
        },
        {
            "id": "claude-sonnet-4-6",
            "name": "Claude Sonnet 4.6",
            "cost": "Previous ATLAS choice",
            "description": "Kept available so existing ATLAS courses do not change automatically.",
        },
    ],
    "openai": [
        {
            "id": "gpt-5.6-luna",
            "name": "GPT-5.6 Luna",
            "cost": "Lowest OpenAI cost",
            "description": "Designed for cost-sensitive, high-volume use.",
        },
        {
            "id": "gpt-5.6-terra",
            "name": "GPT-5.6 Terra",
            "cost": "Medium cost",
            "description": "A good balance of answer quality and cost.",
        },
        {
            "id": "gpt-5.6-sol",
            "name": "GPT-5.6 Sol",
            "cost": "Highest OpenAI cost",
            "description": "For the strongest answers when higher cost is acceptable.",
        },
    ],
}


DEFAULT_MODEL_BY_PROVIDER = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-5.6-terra",
}


def normalize_provider(provider: str) -> str:
    value = (provider or "").strip().lower()
    if value not in PROVIDER_LABELS:
        raise ValueError("Choose Anthropic or OpenAI.")
    return value


def validate_provider_model(provider: str, model: str) -> tuple[str, str]:
    normalized_provider = normalize_provider(provider)
    normalized_model = (model or "").strip()
    supported = {
        option["id"] for option in MODEL_OPTIONS[normalized_provider]
    }
    if normalized_model not in supported:
        raise ValueError("Choose a supported model for that provider.")
    return normalized_provider, normalized_model


def model_catalog() -> List[Dict]:
    return [
        {
            "id": provider,
            "name": PROVIDER_LABELS[provider],
            "default_model": DEFAULT_MODEL_BY_PROVIDER[provider],
            "models": [dict(option) for option in MODEL_OPTIONS[provider]],
        }
        for provider in ("anthropic", "openai")
    ]


def model_name(provider: str, model: str) -> str:
    for option in MODEL_OPTIONS.get(provider, []):
        if option["id"] == model:
            return option["name"]
    return model
