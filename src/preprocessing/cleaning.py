"""Deduplicação e filtros de qualidade sobre o conjunto de tweets.

A deduplicação é feita em dois níveis, com semânticas opostas de propósito:

* **Dentro do usuário**: texto repetido é ruído (repost do próprio conteúdo)
  e é removido.
* **Entre usuários**: texto idêntico entre pessoas diferentes é *sinal* —
  correntes, letras de música, frases de campanha — e é preservado por
  padrão. Removê-lo apagaria um fenômeno linguístico real da população
  estudada.
"""

from __future__ import annotations

import polars as pl

from config.logging import get_logger
from config.settings import (
    CollectionFiltersSection,
    DeduplicationSection,
    PreprocessingFiltersSection,
)
from constants.columns import (
    CREATED_AT,
    LANGUAGE,
    TEXT,
    TEXT_CLEAN,
    TWEET_ID,
    USER_ID,
)
from utils.validation import require_columns

logger = get_logger(__name__)


def deduplicate(frame: pl.DataFrame, config: DeduplicationSection) -> pl.DataFrame:
    """Remove tweets duplicados.

    Parameters
    ----------
    frame : pl.DataFrame
        Tweets coletados.
    config : DeduplicationSection
        Seção ``deduplication`` de ``configs/preprocessing.yaml``.

    Returns
    -------
    pl.DataFrame
        Tweets sem duplicatas.

    Examples
    --------
    >>> from config.settings import DeduplicationSection
    >>> frame = pl.DataFrame(
    ...     {"user_id": ["u_a", "u_a"], "tweet_id": ["t1", "t2"], "text": ["oi", "oi"]}
    ... )
    >>> deduplicate(frame, DeduplicationSection()).height
    1
    """
    require_columns(frame, [USER_ID, TWEET_ID, TEXT], context="deduplicação")
    initial = frame.height
    result = frame

    if config.by_tweet_id:
        result = result.unique(subset=[TWEET_ID], keep=config.keep, maintain_order=True)
    if config.by_text_within_user:
        result = result.unique(subset=[USER_ID, TEXT], keep=config.keep, maintain_order=True)
    if config.by_text_global:
        result = result.unique(subset=[TEXT], keep=config.keep, maintain_order=True)

    removed = initial - result.height
    logger.info(
        "Deduplicação: %d de %d tweets removidos (%.1f%%).",
        removed,
        initial,
        100 * removed / max(initial, 1),
    )
    return result


def filter_by_quality(frame: pl.DataFrame, config: CollectionFiltersSection) -> pl.DataFrame:
    """Aplica os filtros de qualidade da coleta.

    Parameters
    ----------
    frame : pl.DataFrame
        Tweets deduplicados.
    config : CollectionFiltersSection
        Seção ``filters`` de ``configs/collection.yaml``.

    Returns
    -------
    pl.DataFrame
        Tweets que passaram nos filtros.

    Examples
    --------
    >>> from config.settings import CollectionFiltersSection
    >>> frame = pl.DataFrame({"text": ["curto", "um texto suficientemente longo"]})
    >>> filter_by_quality(frame, CollectionFiltersSection()).height
    1
    """
    require_columns(frame, [TEXT], context="filtro de qualidade")
    initial = frame.height

    length = pl.col(TEXT).str.len_chars()
    result = frame.filter(length.is_between(config.min_chars_per_tweet, config.max_chars_per_tweet))

    if config.require_language and LANGUAGE in frame.columns:
        result = result.filter(
            (pl.col(LANGUAGE) == config.require_language) | pl.col(LANGUAGE).is_null()
        )

    removed = initial - result.height
    logger.info("Filtro de qualidade: %d de %d tweets removidos.", removed, initial)
    return result


def filter_after_cleaning(
    frame: pl.DataFrame,
    config: PreprocessingFiltersSection,
) -> pl.DataFrame:
    """Descarta tweets que ficaram vazios ou curtos demais após a limpeza.

    Um tweet composto apenas por emoji, menções e URL vira texto vazio depois
    da normalização. Mantê-lo produziria vetores TF-IDF nulos e contaminaria
    as médias por usuário com zeros que não representam comportamento algum.

    Parameters
    ----------
    frame : pl.DataFrame
        Tweets já limpos.
    config : PreprocessingFiltersSection
        Seção ``filters`` de ``configs/preprocessing.yaml``.

    Returns
    -------
    pl.DataFrame
        Tweets com conteúdo textual suficiente.

    Examples
    --------
    >>> from config.settings import PreprocessingFiltersSection
    >>> frame = pl.DataFrame({"text_clean": ["", "um dois três"]})
    >>> filter_after_cleaning(frame, PreprocessingFiltersSection()).height
    1
    """
    require_columns(frame, [TEXT_CLEAN], context="filtro pós-limpeza")
    initial = frame.height
    result = frame

    if config.drop_empty_after_cleaning:
        result = result.filter(pl.col(TEXT_CLEAN).str.strip_chars().str.len_chars() > 0)

    if config.min_tokens_per_tweet > 0:
        token_count = pl.col(TEXT_CLEAN).str.split(" ").list.len()
        result = result.filter(token_count >= config.min_tokens_per_tweet)

    removed = initial - result.height
    logger.info("Filtro pós-limpeza: %d de %d tweets removidos.", removed, initial)
    return result


def filter_automated_accounts(
    frame: pl.DataFrame,
    max_tweets_per_day: int,
) -> pl.DataFrame:
    """Remove usuários com padrão de publicação automatizado.

    Contas de notícias, divulgação e bots publicam em volume incompatível com
    uso pessoal. Mantê-las distorceria tanto as features temporais quanto o
    rótulo: um perfil que republica manchetes sobre suicídio não é um usuário
    em risco.

    Parameters
    ----------
    frame : pl.DataFrame
        Tweets com ``user_id`` e ``created_at``.
    max_tweets_per_day : int
        Média diária máxima tolerada.

    Returns
    -------
    pl.DataFrame
        Tweets dos usuários que permaneceram.

    Examples
    --------
    >>> filter_automated_accounts(tweets, max_tweets_per_day=80)  # doctest: +SKIP
    """
    require_columns(frame, [USER_ID, CREATED_AT], context="filtro de contas automatizadas")

    daily = (
        frame.group_by(USER_ID)
        .agg(
            pl.len().alias("n_tweets"),
            (pl.col(CREATED_AT).max() - pl.col(CREATED_AT).min()).dt.total_days().alias("span"),
        )
        .with_columns(
            (pl.col("n_tweets") / pl.max_horizontal(pl.col("span"), pl.lit(1))).alias("per_day")
        )
    )

    keep = daily.filter(pl.col("per_day") <= max_tweets_per_day)[USER_ID]
    removed_users = daily.height - keep.len()

    if removed_users:
        logger.info(
            "Contas automatizadas: %d usuário(s) removido(s) por publicar mais de %d "
            "tweets/dia em média.",
            removed_users,
            max_tweets_per_day,
        )

    return frame.filter(pl.col(USER_ID).is_in(keep))


def filter_users_by_activity(
    frame: pl.DataFrame,
    min_tweets: int,
    min_active_days: int,
) -> pl.DataFrame:
    """Mantém apenas usuários com histórico suficiente para o estudo longitudinal.

    Um perfil temporal construído sobre poucos tweets ou poucos dias distintos
    não sustenta as features de tendência e persistência — que são justamente
    a contribuição central da abordagem centrada no usuário.

    Parameters
    ----------
    frame : pl.DataFrame
        Tweets com ``user_id`` e ``created_at``.
    min_tweets : int
        Número mínimo de tweets por usuário.
    min_active_days : int
        Número mínimo de dias distintos com atividade.

    Returns
    -------
    pl.DataFrame
        Tweets dos usuários que atendem aos critérios.

    Examples
    --------
    >>> filter_users_by_activity(tweets, min_tweets=30, min_active_days=15)  # doctest: +SKIP
    """
    require_columns(frame, [USER_ID, CREATED_AT], context="filtro de atividade")

    activity = frame.group_by(USER_ID).agg(
        pl.len().alias("n_tweets"),
        pl.col(CREATED_AT).dt.date().n_unique().alias("active_days"),
    )

    keep = activity.filter(
        (pl.col("n_tweets") >= min_tweets) & (pl.col("active_days") >= min_active_days)
    )[USER_ID]

    logger.info(
        "Filtro de atividade: %d de %d usuários mantidos (>= %d tweets e >= %d dias ativos).",
        keep.len(),
        activity.height,
        min_tweets,
        min_active_days,
    )
    return frame.filter(pl.col(USER_ID).is_in(keep))
