from __future__ import annotations

import json
import re

import requests

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma3:4b"

JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class OllamaError(RuntimeError):
    pass


class LocalLLM:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, timeout: int = 300, temperature: float = 0.2):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    def version(self) -> str | None:
        try:
            response = requests.get(f"{self.host}/api/version", timeout=5)
            response.raise_for_status()
            return response.json().get("version")
        except requests.RequestException:
            return None

    def models(self) -> list[str]:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(
                f"cannot reach a local Ollama server at {self.host} ({exc}). "
                "Start it with `ollama serve` and pull a model, for example `ollama pull gemma3:4b`."
            ) from exc
        return [m.get("name", "") for m in response.json().get("models", [])]

    def ensure_model(self) -> None:
        available = self.models()
        if not available:
            raise OllamaError(f"no local models found. Run `ollama pull {self.model}` first.")
        names = {name.split(":")[0] for name in available} | set(available)
        if self.model not in available and self.model.split(":")[0] not in names:
            raise OllamaError(
                f"model `{self.model}` is not available locally. Run `ollama pull {self.model}`. "
                f"Installed: {', '.join(available)}"
            )

    def chat_json(self, system: str, user: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_ctx": 8192},
        }
        try:
            response = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"ollama request failed: {exc}") from exc

        content = response.json().get("message", {}).get("content", "")
        return _parse_json(content)


def _parse_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content.strip())
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = JSON_BLOCK.search(content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise OllamaError(f"model did not return valid JSON: {content[:400]}")
