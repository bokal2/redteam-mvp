# backend/ola_client.py
import httpx
from typing import Any, Dict

OLLAMA_BASE = "http://localhost:11434"


async def generate(
    model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 1024
) -> Dict[str, Any]:
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Ollama request failed for model '{model}' ({exc.__class__.__name__}: {exc})"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Ollama returned HTTP {exc.response.status_code} for model '{model}': {exc.response.text}"
        ) from exc

    if isinstance(data, dict):
        for key in ("completion", "text", "response", "outputs"):
            if key in data:
                return {"raw": data, "text": data[key]}

    return {"raw": data, "text": str(data)}
