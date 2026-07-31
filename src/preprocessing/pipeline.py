"""Orquestração da etapa de pré-processamento.

Encadeia deduplicação, filtros de qualidade, normalização, limpeza e filtros
pós-limpeza, validando o contrato de dados na entrada e na saída.
"""

from __future__ import annotations

import polars as pl

from config.logging import get_logger
from config.settings import Config
from constants.columns import TEXT, TEXT_CLEAN, TEXT_NORMALIZED
from preprocessing.cleaning import (
    deduplicate,
    filter_after_cleaning,
    filter_automated_accounts,
    filter_by_quality,
    filter_users_by_activity,
)
from preprocessing.text import clean_text, normalize_text
from schemas.tweets import CleanTweetSchema, RawTweetSchema
from schemas.validation import validate_frame
from utils.lexicons import load_stopwords
from utils.timing import log_duration

logger = get_logger(__name__)


def apply_text_processing(frame: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Cria as colunas ``text_normalized`` e ``text_clean``.

    Parameters
    ----------
    frame : pl.DataFrame
        Tweets com a coluna ``text``.
    config : Config
        Configuração completa do projeto.

    Returns
    -------
    pl.DataFrame
        Tweets com as duas colunas de texto processado.

    Examples
    --------
    >>> apply_text_processing(tweets, config)  # doctest: +SKIP
    """
    normalization = config.preprocessing.normalization
    cleaning = config.preprocessing.cleaning
    stopwords = load_stopwords() if cleaning.remove_stopwords else frozenset()

    return frame.with_columns(
        pl.col(TEXT)
        .map_elements(
            lambda text: normalize_text(text, normalization),
            return_dtype=pl.Utf8,
        )
        .alias(TEXT_NORMALIZED)
    ).with_columns(
        pl.col(TEXT_NORMALIZED)
        .map_elements(
            lambda text: clean_text(text, cleaning, stopwords),
            return_dtype=pl.Utf8,
        )
        .alias(TEXT_CLEAN)
    )


def run_preprocessing(frame: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Executa a etapa completa de pré-processamento.

    A ordem é intencional: os filtros baratos (deduplicação, comprimento)
    rodam **antes** da normalização, que é a parte cara — não faz sentido
    normalizar milhões de tweets que serão descartados na linha seguinte.

    Parameters
    ----------
    frame : pl.DataFrame
        Tweets brutos, conforme :class:`schemas.tweets.RawTweetSchema`.
    config : Config
        Configuração completa do projeto.

    Returns
    -------
    pl.DataFrame
        Tweets limpos, conforme :class:`schemas.tweets.CleanTweetSchema`.

    Raises
    ------
    SchemaValidationError
        Se a entrada ou a saída violarem seus contratos.
    EmptyDatasetError
        Se todos os tweets forem descartados pelos filtros.

    Examples
    --------
    >>> limpos = run_preprocessing(brutos, config)  # doctest: +SKIP
    """
    from utils.validation import require_non_empty

    validate_frame(frame, RawTweetSchema, context="entrada do preprocess")

    with log_duration("Pré-processamento dos tweets"):
        result = deduplicate(frame, config.preprocessing.deduplication)
        result = filter_by_quality(result, config.collection.filters)
        result = filter_automated_accounts(result, config.collection.filters.max_tweets_per_day)

        result = apply_text_processing(result, config)
        result = filter_after_cleaning(result, config.preprocessing.filters)

        # A atividade é reavaliada por último: os filtros anteriores podem ter
        # reduzido um usuário abaixo do mínimo exigido.
        result = filter_users_by_activity(
            result,
            min_tweets=config.collection.user_history.min_tweets_per_user,
            min_active_days=config.collection.user_history.min_active_days,
        )

    require_non_empty(result, context="saída do preprocess")
    validate_frame(result, CleanTweetSchema, context="saída do preprocess")

    logger.info(
        "Pré-processamento concluído: %d tweets de %d usuários.",
        result.height,
        result["user_id"].n_unique(),
    )
    return result
