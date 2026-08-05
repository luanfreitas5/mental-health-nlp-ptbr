"""Montagem da matriz final de atributos por usuário.

Reúne os seis grupos de features, acrescenta as colunas descritivas do perfil
(usadas na avaliação por fatias) e trata os valores ausentes.

Sobre valores ausentes: em vez de imputar em silêncio, o módulo cria um
indicador binário ``<coluna>_is_missing`` antes de imputar. A ausência aqui
não é aleatória — uma tendência temporal só falta quando o histórico é curto
demais — então a própria ausência carrega informação, e apagá-la seria jogar
fora sinal.
"""

from __future__ import annotations

import polars as pl

from config.logging import get_logger
from config.settings import FeaturesConfig
from constants.columns import (
    ACTIVE_DAYS,
    CREATED_AT,
    FIRST_TWEET_AT,
    LAST_TWEET_AT,
    MISSING_INDICATOR_SUFFIX,
    N_TWEETS,
    NIGHT_ACTIVITY_RATIO,
    SPAN_DAYS,
    TEMPORAL_PREFIX,
    USER_ID,
    USER_LABEL,
)
from exceptions.data import InsufficientDataError
from features.behavioral import build_behavioral_features
from features.emotional import build_emotional_features
from features.linguistic import build_linguistic_features
from features.psychological import build_psychological_features
from features.semantic import build_semantic_features
from features.temporal import build_temporal_features
from schemas.features import list_feature_columns
from utils.timing import log_duration
from utils.validation import require_columns

logger = get_logger(__name__)


def build_profile_columns(tweets: pl.DataFrame) -> pl.DataFrame:
    """Calcula as colunas descritivas do perfil de cada usuário.

    Não são preditores: descrevem o histórico e são usadas para definir as
    fatias de avaliação e para auditar a qualidade da coleta.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos, com ``user_id`` e ``created_at``.

    Returns
    -------
    pl.DataFrame
        ``n_tweets``, ``active_days``, ``span_days``, ``first_tweet_at`` e
        ``last_tweet_at`` por usuário.

    Examples
    --------
    >>> build_profile_columns(tweets)  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID, CREATED_AT], context="colunas de perfil")

    return (
        tweets.group_by(USER_ID)
        .agg(
            pl.len().alias(N_TWEETS),
            pl.col(CREATED_AT).dt.date().n_unique().alias(ACTIVE_DAYS),
            pl.col(CREATED_AT).min().alias(FIRST_TWEET_AT),
            pl.col(CREATED_AT).max().alias(LAST_TWEET_AT),
        )
        .with_columns(
            (pl.col(LAST_TWEET_AT) - pl.col(FIRST_TWEET_AT)).dt.total_days().alias(SPAN_DAYS)
        )
        .sort(USER_ID)
    )


def handle_missing_values(
    frame: pl.DataFrame,
    config: FeaturesConfig,
) -> pl.DataFrame:
    """Cria indicadores de ausência e imputa os valores faltantes.

    Parameters
    ----------
    frame : pl.DataFrame
        Matriz de atributos com possíveis nulos/``NaN``.
    config : FeaturesConfig
        Seção ``features`` de ``configs/features.yaml``.

    Returns
    -------
    pl.DataFrame
        Matriz sem valores ausentes nas colunas de atributos.

    Notes
    -----
    A mediana usada na imputação é calculada sobre **todo** o conjunto. Para
    as features estruturais deste projeto isso é aceitável e amplamente
    praticado, mas é, a rigor, um vazamento leve de estatística descritiva.
    Uma imputação estritamente livre de vazamento exigiria mover esta etapa
    para dentro do ``Pipeline`` do scikit-learn — a alternativa está
    documentada em ``docs/guides/architecture.md``.

    Examples
    --------
    >>> handle_missing_values(matriz, config.features)  # doctest: +SKIP
    """
    feature_columns = list_feature_columns(frame)
    if not feature_columns:
        return frame

    # Colunas float podem trazer NaN (ex.: tendências temporais sem histórico
    # suficiente — ver features/temporal.py), que o Polars NÃO conta em
    # null_count(). Normalizamos para null aqui para que a detecção e a
    # imputação abaixo enxerguem os dois casos da mesma forma.
    float_columns = [column for column in feature_columns if frame.schema[column].is_float()]
    result = frame.with_columns([pl.col(column).fill_nan(None) for column in float_columns])

    with_nulls = [column for column in feature_columns if result[column].null_count() > 0]

    if config.aggregation.add_missing_indicators and with_nulls:
        result = result.with_columns(
            [
                pl.col(column)
                .is_null()
                .cast(pl.Float64)
                .alias(f"{column}{MISSING_INDICATOR_SUFFIX}")
                for column in with_nulls
            ]
        )
        logger.info("Criados %d indicadores de ausência.", len(with_nulls))

    strategy = config.aggregation.missing_strategy
    if strategy == "keep_nan":
        return result

    if strategy == "zero":
        return result.with_columns([pl.col(column).fill_null(0.0) for column in feature_columns])

    return result.with_columns(
        [
            pl.col(column).fill_null(pl.col(column).median()).fill_null(0.0)
            for column in feature_columns
        ]
    )


def build_user_features(
    tweets: pl.DataFrame,
    config: FeaturesConfig,
    *,
    metadata: pl.DataFrame | None = None,
    psychological_scores: pl.DataFrame | None = None,
    labels: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Monta a matriz completa de atributos por usuário.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos e rotulados por sentimento.
    config : FeaturesConfig
        Seção ``features`` de ``configs/features.yaml``.
    metadata : pl.DataFrame, optional
        Metadados públicos dos usuários (features de audiência).
    psychological_scores : pl.DataFrame, optional
        Vetores psicológicos extraídos pelo LLM.
    labels : pl.DataFrame, optional
        Rótulos por usuário; quando fornecidos, são unidos à matriz.

    Returns
    -------
    pl.DataFrame
        Uma linha por usuário: perfil, atributos e (se disponível) rótulo.

    Raises
    ------
    InsufficientDataError
        Se nenhum usuário atingir ``aggregation.min_tweets_per_user``.

    Examples
    --------
    >>> build_user_features(tweets, config.features, labels=rotulos)  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID, CREATED_AT], context="matriz de atributos")

    enabled = set(config.enabled_groups())
    logger.info("Grupos de atributos ativos: %s.", sorted(enabled) or "nenhum")

    result = build_profile_columns(tweets)

    with log_duration("Construção da matriz de atributos"):
        if "linguistic" in enabled:
            result = result.join(
                build_linguistic_features(tweets, config.linguistic), on=USER_ID, how="left"
            )
        if "emotional" in enabled:
            result = result.join(
                build_emotional_features(tweets, config.emotional), on=USER_ID, how="left"
            )
        if "temporal" in enabled:
            result = result.join(
                build_temporal_features(tweets, config.temporal), on=USER_ID, how="left"
            )
        if "behavioral" in enabled:
            result = result.join(
                build_behavioral_features(tweets, metadata, config.behavioral),
                on=USER_ID,
                how="left",
            )
        if "semantic" in enabled and config.semantic.enabled:
            result = result.join(
                build_semantic_features(tweets, config.semantic), on=USER_ID, how="left"
            )
        if "psychological" in enabled and config.psychological.enabled:
            scores = psychological_scores if psychological_scores is not None else pl.DataFrame()
            result = result.join(
                build_psychological_features(scores, config.psychological),
                on=USER_ID,
                how="left",
            )

    # A razão de atividade noturna é promovida a coluna de perfil: define uma
    # das fatias de avaliação em configs/evaluation.yaml.
    night_column = f"{TEMPORAL_PREFIX}night_activity_ratio"
    if night_column in result.columns:
        result = result.with_columns(pl.col(night_column).alias(NIGHT_ACTIVITY_RATIO))

    minimum = config.aggregation.min_tweets_per_user
    before = result.height
    result = result.filter(pl.col(N_TWEETS) >= minimum)
    if result.is_empty():
        raise InsufficientDataError(
            f"Nenhum usuário atingiu o mínimo de {minimum} tweets "
            f"(features.aggregation.min_tweets_per_user). Usuários avaliados: {before}."
        )
    if before != result.height:
        logger.info(
            "%d usuários removidos por terem menos de %d tweets.", before - result.height, minimum
        )

    result = handle_missing_values(result, config)

    if labels is not None:
        result = result.join(labels.select([USER_ID, USER_LABEL]), on=USER_ID, how="inner")
        logger.info("Matriz unida aos rótulos: %d usuários rotulados.", result.height)

    logger.info(
        "Matriz final: %d usuários × %d colunas (%d atributos).",
        result.height,
        result.width,
        len(list_feature_columns(result)),
    )
    return result.sort(USER_ID)


def select_groups(frame: pl.DataFrame, groups: list[str]) -> pl.DataFrame:
    """Seleciona a matriz restrita a determinados grupos de atributos.

    É a operação que sustenta o Ablation Study: remover um grupo por vez e
    medir o impacto na métrica principal.

    Parameters
    ----------
    frame : pl.DataFrame
        Matriz completa de atributos.
    groups : list of str
        Grupos a manter.

    Returns
    -------
    pl.DataFrame
        Colunas-chave (``user_id``, ``user_label``, perfil) mais os atributos
        dos grupos selecionados.

    Examples
    --------
    >>> select_groups(matriz, ["emotional", "temporal"])  # doctest: +SKIP
    """
    keys = [
        column
        for column in (USER_ID, USER_LABEL, N_TWEETS, SPAN_DAYS, ACTIVE_DAYS, NIGHT_ACTIVITY_RATIO)
        if column in frame.columns
    ]
    features = list_feature_columns(frame, groups)

    # Os indicadores de ausência acompanham o grupo da coluna que sinalizam.
    indicators = [
        column
        for column in frame.columns
        if column.endswith(MISSING_INDICATOR_SUFFIX)
        and column.removesuffix(MISSING_INDICATOR_SUFFIX) in features
    ]

    return frame.select([*keys, *features, *indicators])
