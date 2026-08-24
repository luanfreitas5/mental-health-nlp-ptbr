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
from preprocessing.text import (
    clean_text_expr,
    collapse_repeated_chars,
    finish_normalize_text_expr,
    normalize_text_expr,
)
from schemas.tweets import CleanTweetSchema, RawTweetSchema
from schemas.validation import validate_frame
from utils.lexicons import load_stopwords
from utils.timing import log_duration

logger = get_logger(__name__)


def apply_text_processing(frame: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Cria as colunas ``text_normalized`` e ``text_clean``.

    Implementada com expressões polars vetorizadas (``pl.Expr.str.*``/
    ``pl.Expr.list.eval``) em vez de uma função Python aplicada por tweet. A
    única exceção é o colapso de repetições de caractere
    (:func:`preprocessing.text.collapse_repeated_chars`): o padrão usa
    referência retroativa, que o motor de regex do polars (Rust ``regex``,
    sem backtracking) não suporta — só essa etapa isolada roda via
    ``map_elements``, entre as duas metades vetorizadas da normalização.

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

    frame = frame.with_columns(
        normalize_text_expr(pl.col(TEXT), normalization).alias(TEXT_NORMALIZED)
    )

    if normalization.collapse_repeated_chars:
        keep = normalization.collapse_repeated_chars
        frame = frame.with_columns(
            pl.col(TEXT_NORMALIZED)
            .map_elements(
                lambda text: collapse_repeated_chars(text, keep),
                return_dtype=pl.Utf8,
            )
            .alias(TEXT_NORMALIZED)
        )

    frame = frame.with_columns(
        finish_normalize_text_expr(pl.col(TEXT_NORMALIZED), normalization).alias(TEXT_NORMALIZED)
    )

    return frame.with_columns(
        clean_text_expr(pl.col(TEXT_NORMALIZED), cleaning, stopwords).alias(TEXT_CLEAN)
    )


def run_preprocessing(
    frame: pl.DataFrame, config: Config, *, allow_empty: bool = False
) -> pl.DataFrame:
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
    allow_empty : bool, optional
        Permite que a saída fique vazia sem levantar erro, by default
        ``False``. Usado quando a etapa processa um usuário por vez: um único
        usuário ser inteiramente filtrado (ex.: conta automatizada) é normal,
        não uma falha do pipeline — só o caso agregado (todo o lote vazio)
        deve derrubar a execução.

    Returns
    -------
    pl.DataFrame
        Tweets limpos, conforme :class:`schemas.tweets.CleanTweetSchema`.

    Raises
    ------
    SchemaValidationError
        Se a entrada ou a saída violarem seus contratos.
    EmptyDatasetError
        Se todos os tweets forem descartados pelos filtros e ``allow_empty``
        for ``False``.

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

    if not allow_empty:
        require_non_empty(result, context="saída do preprocess")
    validate_frame(result, CleanTweetSchema, context="saída do preprocess")

    logger.info(
        "Pré-processamento concluído: %d tweets de %d usuários.",
        result.height,
        result["user_id"].n_unique(),
    )
    return result
