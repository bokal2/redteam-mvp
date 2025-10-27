# backend/schemas.py
from typing import Optional

from pydantic import BaseModel, conlist, constr

PromptText = constr(strip_whitespace=True, min_length=1)


class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    model: str


class RunCreate(BaseModel):
    instruction: str
    prompts: conlist(PromptText, min_length=5, max_length=5)
    judge_model: Optional[str] = None
    use_openai_judge: bool = True
