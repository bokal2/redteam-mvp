# backend/prompts.py
from typing import Dict, List, Sequence
import uuid

REFUSAL_TOKEN = "[REFUSE:UNSAFE]"


def _make_prompt_id() -> str:
    return str(uuid.uuid4())


def is_potentially_unsafe(text: str) -> bool:
    return False


def generate_prompts(
    instruction: str, *, user_prompts: Sequence[str]
) -> List[Dict[str, str]]:
    """
    Returns a list of prompt objects with structure:
    {
        "id": "<uuid>",
        "modifier_id": "USER_1",
        "modifier_category": "user_defined",
        "modifier_text": "...",
        "prompt_text": "..."
    }
    """
    if not user_prompts:
        raise ValueError("user_prompts must contain at least one prompt.")

    prompts: List[Dict[str, str]] = []
    base_instruction = instruction.strip()

    for idx, raw_text in enumerate(user_prompts, start=1):
        prompt_id = _make_prompt_id()
        modifier_text = raw_text.strip()
        if not modifier_text:
            raise ValueError(f"Prompt {idx} is empty after trimming whitespace.")
        prompt_text = (
            f"{base_instruction}\n\n{modifier_text}"
            if base_instruction
            else modifier_text
        )
        prompts.append(
            {
                "id": prompt_id,
                "modifier_id": f"USER_{idx}",
                "modifier_category": "user_defined",
                "modifier_text": modifier_text,
                "prompt_text": prompt_text,
            }
        )
    return prompts
