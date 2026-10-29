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
    cross_lingual: bool = Field(
        False,
        description="Run every model regardless of the language it was trained on, so English-only "
                    "models are also applied to Turkish text and vice versa.",
    )


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
    # Values are mixed types: counts plus the detected language code.
    input_stats: Dict[str, Any]
    extractive: Dict[str, Any]
    abstractive: Dict[str, Any]
    comparison: ComparisonResult


class ExperimentDocument(BaseModel):
    """A user-supplied document to include in an experiment run."""
    text: str = Field(..., min_length=50)
    title: Optional[str] = None
    key: Optional[str] = None
    language: Optional[str] = None
    reference_summary: Optional[str] = None


class ExperimentRequest(BaseModel):
    """Request body for the multi-document experiment runner."""
    document_keys: Optional[List[str]] = Field(
        None, description="Built-in sample keys to evaluate (None = all)"
    )
    model_keys: Optional[List[str]] = Field(
        None, description="Restrict to these model keys (None = all available)"
    )
    ratio: float = Field(0.3, ge=0.1, le=0.9)
    extra_documents: Optional[List[ExperimentDocument]] = Field(
        None, description="Additional documents supplied by the user"
    )


class ExtractiveResponse(BaseModel):
    """Response for extractive-only endpoint."""
    summary: str
    method: str
    num_sentences_selected: int
    num_sentences_total: int
    compression_ratio: Optional[float] = None
    language: Optional[str] = None
    reliability: Optional[str] = None
    warning: Optional[str] = None
    metrics: Dict[str, Any]


class AbstractiveResponse(BaseModel):
    """Response for abstractive-only endpoint."""
    summary: str
    method: str
    model_id: str
    max_length: int
    min_length: int
    compression_ratio: Optional[float] = None
    language: Optional[str] = None
    reliability: Optional[str] = None
    warning: Optional[str] = None
    metrics: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    available_methods: List[str]
