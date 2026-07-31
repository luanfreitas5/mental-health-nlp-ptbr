"""Exceções customizadas do projeto, organizadas por domínio.

Todas herdam de :class:`~exceptions.base.MentalHealthNLPError`, o que permite
capturar qualquer erro previsto com um único ``except`` no orquestrador.

Modules
-------
base
    :class:`MentalHealthNLPError` — raiz da hierarquia.
configuration
    Erros de YAML, validação Pydantic e segredos ausentes.
data
    Erros de coleta, leitura/escrita e contratos de dados (pandera).
model
    Erros de construção, treinamento, persistência e LLM (Ollama).
pipeline
    Erros de orquestração, dependência entre etapas e barreira ética.
"""

from exceptions.base import MentalHealthNLPError
from exceptions.configuration import (
    ConfigFileNotFoundError,
    ConfigParsingError,
    ConfigurationError,
    ConfigValidationError,
    MissingSecretError,
)
from exceptions.data import (
    ClassImbalanceError,
    CollectionError,
    DataError,
    DatasetNotFoundError,
    EmptyDatasetError,
    InsufficientDataError,
    RateLimitError,
    SchemaValidationError,
)
from exceptions.model import (
    LLMError,
    LLMResponseError,
    LLMUnavailableError,
    MissingDependencyError,
    ModelError,
    ModelNotFittedError,
    ModelPersistenceError,
    TrainingError,
    UnknownModelError,
)
from exceptions.pipeline import (
    EthicalGateError,
    PipelineError,
    StageDependencyError,
    StageExecutionError,
    UnknownStageError,
)

__all__ = [
    "ClassImbalanceError",
    "CollectionError",
    "ConfigFileNotFoundError",
    "ConfigParsingError",
    "ConfigValidationError",
    "ConfigurationError",
    "DataError",
    "DatasetNotFoundError",
    "EmptyDatasetError",
    "EthicalGateError",
    "InsufficientDataError",
    "LLMError",
    "LLMResponseError",
    "LLMUnavailableError",
    "MentalHealthNLPError",
    "MissingDependencyError",
    "MissingSecretError",
    "ModelError",
    "ModelNotFittedError",
    "ModelPersistenceError",
    "PipelineError",
    "RateLimitError",
    "SchemaValidationError",
    "StageDependencyError",
    "StageExecutionError",
    "TrainingError",
    "UnknownModelError",
    "UnknownStageError",
]
