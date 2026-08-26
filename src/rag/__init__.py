"""Public interfaces for ResaleReady document retrieval."""

from .models import RetrievedChunk
from .retrieval import retrieve

__all__ = ["RetrievedChunk", "retrieve"]
