"""Exceções de orquestração do pipeline."""

from __future__ import annotations

from exceptions.base import MentalHealthNLPError


class PipelineError(MentalHealthNLPError):
    """Erro genérico de pipeline."""


class UnknownStageError(PipelineError):
    """Etapa solicitada não registrada no orquestrador."""


class StageDependencyError(PipelineError):
    """Etapa executada sem que sua dependência tenha produzido a saída esperada.

    Examples
    --------
    >>> raise StageDependencyError(
    ...     "A etapa 'features' exige a saída de 'label'.",
    ...     context={"stage": "features", "missing": "tweets_labeled.parquet"},
    ... )
    Traceback (most recent call last):
    ...
    exceptions.pipeline.StageDependencyError: A etapa 'features' exige ...
    """


class StageExecutionError(PipelineError):
    """Falha durante a execução de uma etapa."""


class EthicalGateError(PipelineError):
    """Barreira ética não satisfeita.

    A etapa de coleta só executa com a aprovação do CEP/CONEP registrada em
    ``.env`` (``ETHICS_APPROVAL_ID``). Ver docs/guides/ethics.md.
    """
