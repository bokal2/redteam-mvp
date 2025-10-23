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
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        # Data structure from Ollama may vary; normalize to access a text string
        # Many Ollama responses contain 'completion' or 'text' or similar fields.
        # We'll attempt to find joined text:
        if isinstance(data, dict):
            # try a few keys
            for k in ("completion", "text", "response", "outputs"):
                if k in data:
                    return {"raw": data, "text": data[k]}
        # fallback: stringified
        return {"raw": data, "text": str(data)}
