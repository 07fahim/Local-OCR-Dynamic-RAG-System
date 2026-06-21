"""Pydantic request/response schemas (API contract, PRD Section 8)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    status: str
    filename: str
    page_count: int
    chunk_count: int
    languages_detected: list[str]


class SearchFilters(BaseModel):
    document_type: str | None = None
    language: str | None = None  # "bn" | "en" | "mixed"
    date_from: str | None = None  # YYYY-MM-DD
    date_to: str | None = None  # YYYY-MM-DD


class SearchRequest(BaseModel):
    query: str
    filters: SearchFilters = Field(default_factory=SearchFilters)
    top_k: int = 5


class Source(BaseModel):
    filename: str
    page_number: int
    document_date: str | None = None
    document_type: str | None = None
    language: str | None = None
    chunk_text: str
    distance: float | None = None


class SearchResponse(BaseModel):
    answer: str
    answer_mode: str  # local | none
    sources: list[Source]
    filters_applied: dict
    chunks_retrieved: int
    warning: str | None = None


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    upload_timestamp: str
    page_count: int | None = None
    chunk_count: int | None = None
    languages_detected: list[str] = Field(default_factory=list)
    document_date: str | None = None
    document_type: str | None = None
    status: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int
