"""Exceções de modelos: construção, treinamento, persistência e LLMs."""

from __future__ import annotations

from exceptions.base import MentalHealthNLPError


class ModelError(MentalHealthNLPError):
    """Erro genérico de modelo."""


class UnknownModelError(ModelError):
    """Nome de modelo/estimador não registrado na fábrica."""


class ModelNotFittedError(ModelError):
    """Tentativa de prever ou salvar um modelo que ainda não foi treinado."""


class ModelPersistenceError(ModelError):
    """Falha ao salvar ou carregar um modelo do disco."""


class TrainingError(ModelError):
    """Falha durante o laço de treinamento."""


class LLMError(ModelError):
    """Erro genérico na comunicação com o provedor de LLM (Ollama)."""


class LLMUnavailableError(LLMError):
    """Servidor Ollama inacessível ou modelo não baixado localmente.

    Notes
    -----
    Falhar aqui é intencional: baixar vários GB de modelo no meio de um
    pipeline longo, sem o usuário perceber, é pior do que interromper.
    """


class LLMResponseError(LLMError):
    """Resposta do LLM fora do formato JSON esperado, mesmo após reparos."""


class MissingDependencyError(ModelError):
    """Dependência opcional necessária ausente.

    Examples
    --------
    >>> raise MissingDependencyError(
    ...     "PyTorch não instalado: rode 'make install-llm' para os extras de LLM."
    ... )
    Traceback (most recent call last):
    ...
    exceptions.model.MissingDependencyError: PyTorch não instalado: ...
    """
