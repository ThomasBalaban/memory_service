# memory_service/models.py

from pydantic import BaseModel
from typing import Optional, List, Dict


class MemoryRecord(BaseModel):
    id: str
    timestamp: float
    source: str           # InputSource.name string, e.g. "MICROPHONE"
    text: str
    memory_text: Optional[str] = None
    interestingness: float = 0.5


class AddMemoryRequest(BaseModel):
    id: str
    timestamp: float
    source: str
    text: str
    memory_text: Optional[str] = None
    interestingness: float = 0.5


class RetrieveRequest(BaseModel):
    query: str
    limit: int = 5


class CompressRequest(BaseModel):
    """
    List of serialised events sent by the director for narrative compression.
    Each entry has at minimum 'source' (InputSource name) and 'text'.
    """
    events: List[Dict[str, str]]


class NarrativeAddRequest(BaseModel):
    text: str