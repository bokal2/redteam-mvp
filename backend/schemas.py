# backend/schemas.py
from pydantic import BaseModel
from typing import Optional


class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    model: str


class RunCreate(BaseModel):
    instruction: str
    judge_model: Optional[str] = None
    use_openai_judge: bool = True
