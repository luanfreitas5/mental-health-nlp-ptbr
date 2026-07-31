"""Exceções de dados: coleta, leitura, escrita e contratos (pandera)."""

from __future__ import annotations

from exceptions.base import MentalHealthNLPError


class DataError(MentalHealthNLPError):
    """Erro genérico de dados."""


class DatasetNotFoundError(DataError):
    """Artefato de dados esperado por uma etapa não existe.

    Normalmente significa que uma etapa anterior do pipeline não foi executada.
    """


class SchemaValidationError(DataError):
    """Violação de contrato de dados (schema pandera).

    Levantada na fronteira entre estágios (``raw -> interim -> processed``)
    para impedir que corrupção silenciosa se propague pelo pipeline.
    """


class EmptyDatasetError(DataError):
    """Dataset vazio após filtragem ou limpeza."""


class InsufficientDataError(DataError):
    """Volume de dados abaixo do mínimo exigido para a etapa.

    Exemplos: usuário com menos tweets que ``min_tweets_per_user``, ou classe
    com menos usuários que ``min_users_per_class``.
    """


class ClassImbalanceError(DataError):
    """Desbalanceamento entre classes acima do limite aceitável."""


class CollectionError(DataError):
    """Falha na coleta de dados do X/Twitter (twscrape)."""


class RateLimitError(CollectionError):
    """Limite de requisições atingido durante a coleta."""
