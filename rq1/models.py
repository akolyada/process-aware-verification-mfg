"""Thin model clients (standard library only). Configure in rq1/config.yaml
(copy config.example.yaml). Pin model versions/snapshots and report them (spec §6)."""
from __future__ import annotations

import json
import os
import urllib.request


class OpenAICompatible:
    """Local open-weight models behind an OpenAI-compatible API (Ollama, llama.cpp, vLLM)."""

    def __init__(self, base_url: str, model: str, temperature: float = 0.0, api_key: str = "none",
                 no_think: bool = False, json_schema: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.model, self.temperature, self.api_key = model, temperature, api_key
        # Reasoning models (e.g. Qwen3): disable the reasoning phase via the API, so the
        # prompt stays identical across models. Verified with Ollama 0.20.2: no
        # "reasoning" field is returned when reasoning_effort is "none".
        self.no_think = no_think
        self.json_schema = json_schema   # structured output (constrained decoding)

    def complete(self, system: str, user: str, seed: int | None = None) -> str:
        body = {"model": self.model, "temperature": self.temperature,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        if self.no_think:
            body["reasoning_effort"] = "none"
        if self.json_schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "output", "schema": self.json_schema}}
        if seed is not None:
            body["seed"] = seed
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", json.dumps(body).encode(),
            {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]


class AnthropicAPI:
    """Anthropic Messages API. Needs ANTHROPIC_API_KEY. Use a pinned model id from the docs."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1500):
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens

    def complete(self, system: str, user: str, seed: int | None = None) -> str:
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "temperature": self.temperature, "system": system,
                "messages": [{"role": "user", "content": user}]}
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", json.dumps(body).encode(),
            {"Content-Type": "application/json", "x-api-key": os.environ["ANTHROPIC_API_KEY"],
             "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.loads(r.read())
        return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")


class MockPlanner:
    """Offline test double (never used for results). 'correct' returns the correct plan;
    'obedient' applies the attack class's manipulation whenever the untrusted text it can
    read contains the class marker the test harness plants. Verifies the pipeline,
    the compliance check, D4a (which removes the text) and D2 plumbing."""

    def __init__(self, mode: str):
        self.mode, self.model = mode, f"mock-{mode}"

    def complete(self, system: str, user: str, seed: int | None = None) -> str:
        import re, sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
        from mfgsec.plans import correct_plan, attack_plan
        from mfgsec.model import Order
        import numpy as np
        if system.startswith("You are a safety and quality reviewer"):
            return '{"decision": "ACCEPT", "reason": "mock"}'
        wo = json.loads(re.search(r"WORK ORDER \(structured\): (\{.*?\})\n", user).group(1))
        md = json.loads(re.search(r"MASTER DATA: (\{.*\})", user).group(1))
        order = Order(wo["order_id"], wo["product"], wo["variant"], wo["quantity"],
                      m1_down=wo["machine_M1_down"], exceptions=md["recorded_exceptions"],
                      recorded=md["recorded_exceptions"])
        plan = correct_plan(order)
        if self.mode == "obedient":
            free = user.split("MASTER DATA:", 1)[1]
            for cls in ("C1_wrong_variant", "C2_inspection_omission",
                        "C3_param_outside_quality_window", "C4_operation_reorder",
                        "C5_throughput_sabotage", "C6_denial_of_service_holds"):
                if f"[[{cls}]]" in free:
                    plan, _ = attack_plan(order, cls, np.random.default_rng(seed or 0))
        return json.dumps(plan)


def build_client(cfg: dict):
    if cfg["kind"] == "mock":
        return MockPlanner(cfg["mode"])
    if cfg["kind"] == "openai_compatible":
        from rq1.prompt import PLAN_JSON_SCHEMA, DECISION_JSON_SCHEMA
        schema = {"plan": PLAN_JSON_SCHEMA, "decision": DECISION_JSON_SCHEMA,
                  None: None}[cfg.get("structured_output")]
        return OpenAICompatible(cfg["base_url"], cfg["model"], cfg.get("temperature", 0.0),
                                cfg.get("api_key", "none"), cfg.get("no_think", False), schema)
    if cfg["kind"] == "anthropic":
        return AnthropicAPI(cfg["model"], cfg.get("temperature", 0.0))
    raise ValueError(cfg["kind"])
