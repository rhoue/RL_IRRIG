"""
Standalone local LLM adapter for offline use.

This module is intentionally self-contained and optional. It does not modify
existing app behavior unless imported and called explicitly.

Example:
    from src.llm_local import LocalLLM, LLMConfig
    llm = LocalLLM(LLMConfig(backend="ollama", model="llama3"))
    reply = llm.generate("Explain the training curve.", context={"reward": [1, 2, 3]})
    print(reply.text)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Union
import json
import os
import time
import urllib.request
import urllib.error

JsonDict = Dict[str, Any]


@dataclass
class LLMConfig:
    backend: str = "ollama"  # "ollama", "llamacpp", "transformers"
    model: str = "llama3"
    endpoint: Optional[str] = None
    timeout_s: int = 60
    temperature: float = 0.2
    max_tokens: int = 512
    top_p: float = 0.9
    stop: Optional[Iterable[str]] = None
    extra: JsonDict = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    raw: JsonDict
    model: str
    elapsed_s: float


class LocalLLM:
    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[JsonDict] = None,
        params: Optional[JsonDict] = None,
    ) -> LLMResponse:
        full_prompt = self._build_prompt(prompt, system=system, context=context)
        cfg = self._merge_params(params)
        t0 = time.time()
        if cfg.backend == "ollama":
            # Ollama expose un endpoint simple /api/generate, sans session.
            raw = self._generate_ollama(full_prompt, cfg)
            text = raw.get("response", "")
            model = raw.get("model", cfg.model)
        elif cfg.backend == "llamacpp":
            # llama.cpp peut exposer /completion ou /v1/completions selon la config.
            raw = self._generate_llamacpp(full_prompt, cfg)
            text = raw.get("content", "") or raw.get("choices", [{}])[0].get("text", "")
            model = raw.get("model", cfg.model)
        elif cfg.backend == "transformers":
            # Backend local Python, utile pour un modèle déjà téléchargé.
            raw = self._generate_transformers(full_prompt, cfg)
            text = raw.get("text", "")
            model = raw.get("model", cfg.model)
        else:
            raise ValueError(f"Unsupported backend: {cfg.backend}")
        return LLMResponse(text=text, raw=raw, model=model, elapsed_s=time.time() - t0)

    def _build_prompt(
        self,
        user_prompt: str,
        system: Optional[str] = None,
        context: Optional[JsonDict] = None,
    ) -> str:
        parts = []
        if system:
            parts.append(f"System:\n{system}\n")
        if context:
            # On sérialise en JSON pour un contexte compact et traçable.
            ctx = json.dumps(context, ensure_ascii=True, sort_keys=True, indent=2)
            parts.append(f"Context (JSON):\n{ctx}\n")
        parts.append(f"User:\n{user_prompt}\n")
        parts.append("Assistant:\n")
        return "\n".join(parts)

    def _merge_params(self, params: Optional[JsonDict]) -> LLMConfig:
        if not params:
            return self.config
        merged = LLMConfig(**{**self.config.__dict__, **params})
        return merged

    def _generate_ollama(self, prompt: str, cfg: LLMConfig) -> JsonDict:
        endpoint = cfg.endpoint or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        url = endpoint.rstrip("/") + "/api/generate"
        payload: JsonDict = {
            "model": cfg.model,
            "prompt": prompt,
            "stream": False,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "num_predict": cfg.max_tokens,
        }
        if cfg.stop:
            payload["stop"] = list(cfg.stop)
        payload.update(cfg.extra)
        return self._post_json(url, payload, timeout_s=cfg.timeout_s)

    def _generate_llamacpp(self, prompt: str, cfg: LLMConfig) -> JsonDict:
        endpoint = cfg.endpoint or os.getenv("LLAMACPP_ENDPOINT", "http://localhost:8080")
        if endpoint.endswith("/completion") or endpoint.endswith("/v1/completions"):
            url = endpoint
            use_v1 = endpoint.endswith("/v1/completions")
        else:
            url = endpoint.rstrip("/") + "/completion"
            use_v1 = False
        # Ajuste le schéma de payload selon l'API exposée.
        if use_v1:
            payload = {
                "model": cfg.model,
                "prompt": prompt,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "max_tokens": cfg.max_tokens,
            }
            if cfg.stop:
                payload["stop"] = list(cfg.stop)
        else:
            payload = {
                "prompt": prompt,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "n_predict": cfg.max_tokens,
            }
            if cfg.stop:
                payload["stop"] = list(cfg.stop)
        payload.update(cfg.extra)
        return self._post_json(url, payload, timeout_s=cfg.timeout_s)

    def _generate_transformers(self, prompt: str, cfg: LLMConfig) -> JsonDict:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "transformers backend requires 'transformers' and 'torch' installed."
            ) from exc
        # Chargement direct du modèle local; utile hors-ligne.
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        model = AutoModelForCausalLM.from_pretrained(cfg.model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_tokens,
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"text": text, "model": cfg.model}

    def _post_json(self, url: str, payload: JsonDict, timeout_s: int) -> JsonDict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"LLM HTTP error {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM connection error: {exc.reason}") from exc
