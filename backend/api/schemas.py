"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class SummarizeRequest(BaseModel):
    """Request body for summarization endpoints."""
    text: str = Field(..., min_length=50, description="Text to summarize (minimum 50 characters)")
    num_sentences: Optional[int] = Field(None, ge=1, description="Number of sentences for extractive summary")
    ratio: float = Field(0.3, ge=0.1, le=0.9, description="Compression ratio (0.1 = very short, 0.9 = minimal compression)")
    max_length: Optional[int] = Field(None, ge=20, description="Max length for abstractive summary")
    min_length: Optional[int] = Field(None, ge=10, description="Min length for abstractive summary")
    reference_summary: Optional[str] = Field(None, description="Optional reference summary for evaluation")


class SummaryResult(BaseModel):
    """Individual summary result."""
    summary: str
    method: str
    compression_ratio: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None


class ComparisonResult(BaseModel):
    """Comparison between methods."""
    rankings: Dict[str, Any]


class SummarizeResponse(BaseModel):
    """Response for the compare endpoint."""
    input_stats: Dict[str, int]
    extractive: Dict[str, Any]
    abstractive: Dict[str, Any]
    comparison: ComparisonResult


class ExtractiveResponse(BaseModel):
    """Response for extractive-only endpoint."""
    summary: str
    method: str
    num_sentences_selected: int
    num_sentences_total: int
    compression_ratio: Optional[float] = None
    metrics: Dict[str, Any]


class AbstractiveResponse(BaseModel):
    """Response for abstractive-only endpoint."""
    summary: str
    method: str
    model_id: str
    max_length: int
    min_length: int
    compression_ratio: Optional[float] = None
    metrics: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    available_methods: List[str]
