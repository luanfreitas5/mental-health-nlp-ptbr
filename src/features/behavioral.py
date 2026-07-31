"""Atributos comportamentais por usuário (prefixo ``behav_``).

Cobre a Seção 5 da proposta: engajamento recebido (curtidas, respostas,
repostagens) e audiência (seguidores, seguindo).

Duas decisões técnicas:

* **Transformação logarítmica.** Contagens de engajamento em rede social
  seguem lei de potência — a maioria dos tweets tem zero curtidas e alguns
  poucos têm milhares. Sem ``log1p``, um único tweet viral domina a média do
  usuário e o modelo passa a modelar viralidade, não saúde mental.
* **Razão seguidores/seguindo.** Preserva-se a razão além dos dois valores
  brutos porque ela é a grandeza com interpretação social direta (valores
  muito baixos sugerem isolamento ou conta pouco estabelecida).
"""

from __future__ import annotations

import polars as pl

from config.logging import get_logger
from config.settings import BehavioralSection
from constants.columns import (
    BEHAVIORAL_PREFIX,
    FOLLOWERS_COUNT,
    FOLLOWING_COUNT,
    IS_REPLY,
    IS_RETWEET,
    LIKE_COUNT,
    QUOTE_COUNT,
    REPLY_COUNT,
    RETWEET_COUNT,
    STATUSES_COUNT,
    USER_ID,
)
from features.emotional import build_aggregations
from utils.validation import require_columns

logger = get_logger(__name__)

#: Colunas de engajamento por tweet.
ENGAGEMENT_COLUMNS: tuple[str, ...] = (LIKE_COUNT, REPLY_COUNT, RETWEET_COUNT, QUOTE_COUNT)


def compute_engagement(
    tweets: pl.DataFrame,
    aggregations: list[str],
    *,
    log_transform: bool = True,
) -> pl.DataFrame:
    """Agrega o engajamento recebido pelos tweets de cada usuário.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets com as colunas de contagem de engajamento.
    aggregations : list of str
        Agregações a aplicar (``mean``, ``median``, ``sum``, ``max``, ...).
    log_transform : bool, optional
        Aplica ``log1p`` antes de agregar, by default True.

    Returns
    -------
    pl.DataFrame
        Engajamento agregado por usuário e taxa de engajamento nulo.

    Examples
    --------
    >>> compute_engagement(tweets, ["mean", "max"])  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID], context="engajamento")

    available = [column for column in ENGAGEMENT_COLUMNS if column in tweets.columns]
    if not available:
        logger.warning("Nenhuma coluna de engajamento encontrada nos tweets.")
        return tweets.select(USER_ID).unique().sort(USER_ID)

    frame = tweets
    if log_transform:
        frame = frame.with_columns(
            [pl.col(column).cast(pl.Float64).log1p().alias(column) for column in available]
        )

    expressions: list[pl.Expr] = []
    for column in available:
        expressions.extend(build_aggregations(column, aggregations, BEHAVIORAL_PREFIX))

    # Fração de tweets sem nenhuma interação: mede alcance social de forma
    # complementar à média, que é puxada pelos poucos tweets muito populares.
    if LIKE_COUNT in available:
        expressions.append(
            (pl.col(LIKE_COUNT) == 0).mean().alias(f"{BEHAVIORAL_PREFIX}zero_like_ratio")
        )

    return frame.group_by(USER_ID).agg(expressions).sort(USER_ID)


def compute_interaction_ratios(tweets: pl.DataFrame) -> pl.DataFrame:
    """Calcula a proporção de respostas e repostagens no histórico.

    Uma fração alta de respostas indica conversação; uma fração alta de
    publicações originais sem interação sugere um perfil mais monológico,
    padrão associado a isolamento na literatura.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets com ``is_reply`` e ``is_retweet``.

    Returns
    -------
    pl.DataFrame
        Razões de resposta, repostagem e conteúdo original.

    Examples
    --------
    >>> compute_interaction_ratios(tweets)  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID], context="razões de interação")

    expressions: list[pl.Expr] = []
    if IS_REPLY in tweets.columns:
        expressions.append(pl.col(IS_REPLY).mean().alias(f"{BEHAVIORAL_PREFIX}reply_ratio"))
    if IS_RETWEET in tweets.columns:
        expressions.append(pl.col(IS_RETWEET).mean().alias(f"{BEHAVIORAL_PREFIX}retweet_ratio"))

    if not expressions:
        return tweets.select(USER_ID).unique().sort(USER_ID)

    result = tweets.group_by(USER_ID).agg(expressions)

    if f"{BEHAVIORAL_PREFIX}reply_ratio" in result.columns:
        original = 1.0 - pl.col(f"{BEHAVIORAL_PREFIX}reply_ratio")
        if f"{BEHAVIORAL_PREFIX}retweet_ratio" in result.columns:
            original = original - pl.col(f"{BEHAVIORAL_PREFIX}retweet_ratio")
        result = result.with_columns(
            original.clip(0.0, 1.0).alias(f"{BEHAVIORAL_PREFIX}original_ratio")
        )

    return result.sort(USER_ID)


def compute_audience(metadata: pl.DataFrame, *, log_transform: bool = True) -> pl.DataFrame:
    """Deriva as features de audiência a partir dos metadados públicos.

    Parameters
    ----------
    metadata : pl.DataFrame
        Metadados do usuário (:class:`schemas.users.UserMetadataSchema`).
    log_transform : bool, optional
        Aplica ``log1p`` às contagens, by default True.

    Returns
    -------
    pl.DataFrame
        Seguidores, seguindo, razão entre eles e total de publicações.

    Examples
    --------
    >>> compute_audience(metadados)  # doctest: +SKIP
    """
    require_columns(metadata, [USER_ID], context="audiência")

    available = [
        column
        for column in (FOLLOWERS_COUNT, FOLLOWING_COUNT, STATUSES_COUNT)
        if column in metadata.columns
    ]
    if not available:
        logger.warning("Metadados sem colunas de audiência: features omitidas.")
        return metadata.select(USER_ID).unique().sort(USER_ID)

    result = metadata.select([USER_ID, *available])

    if FOLLOWERS_COUNT in available and FOLLOWING_COUNT in available:
        result = result.with_columns(
            (
                pl.col(FOLLOWERS_COUNT).cast(pl.Float64)
                / (pl.col(FOLLOWING_COUNT).cast(pl.Float64) + 1.0)
            ).alias(f"{BEHAVIORAL_PREFIX}follower_following_ratio")
        )

    for column in available:
        expression = pl.col(column).cast(pl.Float64)
        if log_transform:
            expression = expression.log1p()
        result = result.with_columns(expression.alias(f"{BEHAVIORAL_PREFIX}{column}"))

    return result.drop(available).sort(USER_ID)


def build_behavioral_features(
    tweets: pl.DataFrame,
    metadata: pl.DataFrame | None,
    config: BehavioralSection,
) -> pl.DataFrame:
    """Monta todas as features comportamentais por usuário.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos.
    metadata : pl.DataFrame or None
        Metadados públicos; ``None`` desativa as features de audiência.
    config : BehavioralSection
        Seção ``behavioral`` de ``configs/features.yaml``.

    Returns
    -------
    pl.DataFrame
        Uma linha por usuário, com as colunas de prefixo ``behav_``.

    Examples
    --------
    >>> build_behavioral_features(tweets, metadados, config.features.behavioral)  # doctest: +SKIP
    """
    frames: list[pl.DataFrame] = []

    if config.engagement:
        frames.append(
            compute_engagement(tweets, config.aggregations, log_transform=config.log_transform)
        )
    if config.reply_ratio:
        frames.append(compute_interaction_ratios(tweets))
    if config.audience and metadata is not None and not metadata.is_empty():
        frames.append(compute_audience(metadata, log_transform=config.log_transform))
    elif config.audience:
        logger.warning("Metadados de usuário ausentes: as features de audiência serão omitidas.")

    if not frames:
        return tweets.select(USER_ID).unique().sort(USER_ID)

    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on=USER_ID, how="full", coalesce=True)

    logger.info(
        "Features comportamentais: %d colunas para %d usuários.", result.width - 1, result.height
    )
    return result.sort(USER_ID)
