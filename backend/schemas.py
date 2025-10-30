# backend/schemas.py
from typing import Optional

from pydantic import BaseModel, Field, conlist, constr

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


MetricText = constr(strip_whitespace=True, min_length=1)


class AgentRunCreate(BaseModel):
    instruction: str
    prompt_count: int = Field(default=5, ge=1, le=20)
    metrics: conlist(MetricText, min_length=1, max_length=6)
    guidance: Optional[str] = None
    generator_model: Optional[str] = None
    judge_model: Optional[str] = None
    use_openai_judge: bool = True
    target_model: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0.0, le=1.5)
    max_tokens: int = Field(default=1024, ge=64, le=4096)
    parallelism: int = Field(default=3, ge=1, le=10)
    model_timeout: int = Field(default=30, ge=5, le=120)
