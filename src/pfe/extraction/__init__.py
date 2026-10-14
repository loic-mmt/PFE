"""Optional local financial text extraction; importing this package runs no model."""

from pfe.extraction.engine import ExtractionLimits, ExtractionResult, Extractor
from pfe.extraction.evaluation import evaluate_extractions
from pfe.extraction.ollama import ModelResponse, OllamaBackend
from pfe.extraction.schemas import (
    ENVELOPES,
    EVENT_TYPES,
    PROMPT_VERSION,
    RELATIONS,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    extraction_messages,
    extraction_schema,
)
from pfe.extraction.validation import adjudicate_fact, validate_fact

__all__ = [
    "ENVELOPES",
    "EVENT_TYPES",
    "PROMPT_VERSION",
    "RELATIONS",
    "SCHEMA_VERSION",
    "SYSTEM_PROMPT",
    "ExtractionLimits",
    "ExtractionResult",
    "Extractor",
    "ModelResponse",
    "OllamaBackend",
    "adjudicate_fact",
    "evaluate_extractions",
    "extraction_messages",
    "extraction_schema",
    "validate_fact",
]
