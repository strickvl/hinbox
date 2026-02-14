"""Base classes and interfaces for embedding providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Valid device targets for local embedding models
EmbeddingDevice = Literal["auto", "cpu", "cuda", "mps"]


class EmbeddingConfig(BaseModel):
    """Configuration for embedding providers."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    batch_size: int = 32
    max_retries: int = 3
    timeout: int = 30
    device: EmbeddingDevice = "auto"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingResult(BaseModel):
    """Result from embedding operation."""

    embeddings: List[List[float]]
    model: str
    dimension: Optional[int] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    @abstractmethod
    async def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> EmbeddingResult:
        """Embed multiple texts in a batch."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        pass
