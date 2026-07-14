"""Pydantic request/response contracts for the API."""
from typing import Optional, List
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["Why are people unhappy with Credit Cards?"])
    product_filter: Optional[str] = Field(
        default=None,
        description="Restrict retrieval to one product category, e.g. 'Credit card'.",
    )


class SourceChunk(BaseModel):
    text: str
    complaint_id: Optional[str] = None
    product_category: Optional[str] = None
    issue: Optional[str] = None
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    collection_count: int
