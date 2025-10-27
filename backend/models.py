# backend/models.py
from typing import Optional
from sqlalchemy import Column, JSON
from sqlmodel import SQLModel, Field
from datetime import datetime
import uuid
from pydantic import ConfigDict


def gen_uuid():
    return str(uuid.uuid4())


class Experiment(SQLModel, table=True):
    id: str = Field(default_factory=gen_uuid, primary_key=True, index=True)
    name: str
    description: Optional[str] = None
    model: str  # e.g., "llama3.2" or "gemma"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Run(SQLModel, table=True):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=gen_uuid, primary_key=True, index=True)
    experiment_id: str = Field(foreign_key="experiment.id", index=True)
    base_instruction: str
    prompt: str
    response: Optional[str] = None
    blocked: bool = False
    error: Optional[str] = None
    modifier_id: Optional[str] = Field(default=None, index=True)
    modifier_category: Optional[str] = None
    detector_scores: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    judge_scores: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    run_metadata: Optional[dict] = Field(
        default=None,
        sa_column=Column("metadata", JSON),
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class JudgePrompt(SQLModel, table=True):
    id: str = Field(default_factory=gen_uuid, primary_key=True)
    name: str
    prompt_text: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
