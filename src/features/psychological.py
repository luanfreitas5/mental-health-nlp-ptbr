"""Agregação do vetor psicológico extraído por LLM (prefixo ``psy_``).

Cobre a Seção 6 da proposta e sustenta a hipótese H3 — que atributos
psicológicos extraídos por LLM aumentam o desempenho dos classificadores.

A extração em si acontece na etapa ``psych`` (:mod:`labeling.llm`); aqui os
scores por lote são agregados no nível do usuário. O máximo é agregado ao
lado da média por um motivo clínico: um único período de risco elevado é
clinicamente relevante mesmo quando a média do ano é baixa, e a média sozinha
diluiria exatamente o sinal que mais importa.
"""

from __future__ import annotations

import polars as pl

from config.logging import get_logger
from config.settings import PsychologicalSection
from constants.columns import PSYCHOLOGICAL_PREFIX, USER_ID
from features.emotional import build_aggregations
from utils.validation import require_columns

logger = get_logger(__name__)


def build_psychological_features(
    scores: pl.DataFrame,
    config: PsychologicalSection,
) -> pl.DataFrame:
    """Agrega o vetor psicológico por usuário.

    Parameters
    ----------
    scores : pl.DataFrame
        Scores por lote (:class:`schemas.tweets.PsychologicalScoreSchema`).
    config : PsychologicalSection
        Seção ``psychological`` de ``configs/features.yaml``.

    Returns
    -------
    pl.DataFrame
        Uma linha por usuário, com as colunas de prefixo ``psy_``.

    Raises
    ------
    KeyError
        Se alguma dimensão configurada não existir nos scores.

    Examples
    --------
    >>> build_psychological_features(scores, config.features.psychological)  # doctest: +SKIP
    """
    if scores.is_empty():
        logger.warning(
            "Nenhum score psicológico disponível: o grupo 'psychological' ficará vazio. "
            "Execute a etapa 'psych' para habilitá-lo."
        )
        return pl.DataFrame({USER_ID: []}, schema={USER_ID: pl.Utf8})

    require_columns(scores, [USER_ID], context="features psicológicas")

    missing = [dimension for dimension in config.dimensions if dimension not in scores.columns]
    if missing:
        raise KeyError(
            f"Dimensões psicológicas ausentes nos scores: {missing}. "
            f"Disponíveis: {sorted(scores.columns)}"
        )

    expressions: list[pl.Expr] = []
    for dimension in config.dimensions:
        expressions.extend(build_aggregations(dimension, config.aggregations, PSYCHOLOGICAL_PREFIX))

    expressions.append(pl.len().alias(f"{PSYCHOLOGICAL_PREFIX}n_batches"))

    result = scores.group_by(USER_ID).agg(expressions).sort(USER_ID)

    # Índice composto de risco: combina as dimensões negativas e desconta a
    # esperança, que é a única dimensão positiva do vetor. Resume o vetor num
    # único número interpretável para figuras e para o model card.
    negative = [
        f"{PSYCHOLOGICAL_PREFIX}{dimension}_mean"
        for dimension in ("tristeza", "isolamento", "ansiedade", "risco_suicida")
        if f"{PSYCHOLOGICAL_PREFIX}{dimension}_mean" in result.columns
    ]
    hope = f"{PSYCHOLOGICAL_PREFIX}esperanca_mean"

    if negative:
        composite = pl.mean_horizontal([pl.col(column) for column in negative])
        if hope in result.columns:
            composite = composite - pl.col(hope)
        result = result.with_columns(composite.alias(f"{PSYCHOLOGICAL_PREFIX}risk_index"))

    logger.info(
        "Features psicológicas: %d colunas para %d usuários.", result.width - 1, result.height
    )
    return result


def compute_psychological_trend(scores: pl.DataFrame, dimension: str) -> pl.DataFrame:
    """Mede a evolução de uma dimensão psicológica ao longo dos lotes.

    Os lotes são construídos em ordem cronológica, então a diferença entre o
    último e o primeiro descreve a trajetória do usuário — informação que
    nenhuma agregação estática captura.

    Parameters
    ----------
    scores : pl.DataFrame
        Scores por lote.
    dimension : str
        Dimensão a analisar (ex.: ``"risco_suicida"``).

    Returns
    -------
    pl.DataFrame
        Delta entre o último e o primeiro lote, por usuário.

    Raises
    ------
    KeyError
        Se a dimensão não existir nos scores.

    Examples
    --------
    >>> compute_psychological_trend(scores, "risco_suicida")  # doctest: +SKIP
    """
    if dimension not in scores.columns:
        raise KeyError(f"Dimensão '{dimension}' ausente. Disponíveis: {sorted(scores.columns)}")

    return (
        scores.sort([USER_ID, "batch_index"])
        .group_by(USER_ID)
        .agg(
            (pl.col(dimension).last() - pl.col(dimension).first()).alias(
                f"{PSYCHOLOGICAL_PREFIX}{dimension}_delta"
            ),
            pl.col(dimension).max().alias(f"{PSYCHOLOGICAL_PREFIX}{dimension}_peak"),
        )
        .sort(USER_ID)
    )
