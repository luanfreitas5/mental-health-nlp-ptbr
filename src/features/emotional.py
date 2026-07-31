"""Atributos emocionais agregados por usuário (prefixo ``emo_``).

Cobre a Seção 2 da proposta: distribuição de sentimento, confiança do
classificador e intensidade das emoções finas.

Além da média, o módulo agrega desvio, máximo e percentil 90. A justificativa
é clínica: dois usuários podem ter a mesma tristeza média, mas um oscila
enquanto o outro se mantém constantemente triste — e a literatura associa
persistência, mais que intensidade média, ao quadro depressivo. Só a média
apagaria essa diferença.
"""

from __future__ import annotations

import polars as pl

from config.logging import get_logger
from config.settings import EmotionalSection
from constants.columns import (
    EMOTION_PREFIX,
    EMOTIONAL_PREFIX,
    SENTIMENT,
    SENTIMENT_POLARITY,
    SENTIMENT_SCORE,
    USER_ID,
)
from constants.labels import Sentiment
from utils.validation import require_columns

logger = get_logger(__name__)

#: Agregação declarada em ``features.yaml`` -> expressão polars correspondente.
_AGGREGATION_BUILDERS = {
    "mean": lambda column: pl.col(column).mean(),
    "std": lambda column: pl.col(column).std(),
    "max": lambda column: pl.col(column).max(),
    "min": lambda column: pl.col(column).min(),
    "median": lambda column: pl.col(column).median(),
    "sum": lambda column: pl.col(column).sum(),
    "p90": lambda column: pl.col(column).quantile(0.90),
}


def build_aggregations(column: str, aggregations: list[str], prefix: str) -> list[pl.Expr]:
    """Constrói as expressões de agregação de uma coluna numérica.

    Parameters
    ----------
    column : str
        Coluna a agregar.
    aggregations : list of str
        Nomes das agregações (``mean``, ``std``, ``max``, ``p90``, ...).
    prefix : str
        Prefixo do nome das colunas resultantes.

    Returns
    -------
    list of pl.Expr
        Expressões nomeadas ``<prefix><column>_<agregacao>``.

    Raises
    ------
    KeyError
        Se alguma agregação não for suportada.

    Examples
    --------
    >>> len(build_aggregations("x", ["mean", "std"], "emo_"))
    2
    """
    unknown = [name for name in aggregations if name not in _AGGREGATION_BUILDERS]
    if unknown:
        raise KeyError(
            f"Agregações não suportadas: {unknown}. Disponíveis: {sorted(_AGGREGATION_BUILDERS)}"
        )

    return [
        _AGGREGATION_BUILDERS[name](column).alias(f"{prefix}{column}_{name}")
        for name in aggregations
    ]


def compute_sentiment_distribution(tweets: pl.DataFrame) -> pl.DataFrame:
    """Calcula a distribuição de sentimento por usuário.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets rotulados, com ``sentiment`` e ``sentiment_polarity``.

    Returns
    -------
    pl.DataFrame
        Proporção de cada sentimento, polaridade média e razão
        negativo/positivo.

    Examples
    --------
    >>> compute_sentiment_distribution(tweets)  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID, SENTIMENT], context="distribuição de sentimento")

    aggregations = [
        (pl.col(SENTIMENT) == str(sentiment)).mean().alias(f"{EMOTIONAL_PREFIX}{sentiment}_ratio")
        for sentiment in (
            Sentiment.POSITIVO,
            Sentiment.NEGATIVO,
            Sentiment.NEUTRO,
            Sentiment.INDEFINIDO,
        )
    ]

    if SENTIMENT_POLARITY in tweets.columns:
        aggregations.extend(
            [
                pl.col(SENTIMENT_POLARITY).mean().alias(f"{EMOTIONAL_PREFIX}polarity_mean"),
                pl.col(SENTIMENT_POLARITY).std().alias(f"{EMOTIONAL_PREFIX}polarity_std"),
                pl.col(SENTIMENT_POLARITY).min().alias(f"{EMOTIONAL_PREFIX}polarity_min"),
            ]
        )

    # +0,01 no denominador: usuários sem nenhum tweet positivo são comuns e a
    # razão precisa continuar finita para não virar inf na matriz.
    return (
        tweets.group_by(USER_ID)
        .agg(aggregations)
        .with_columns(
            (
                pl.col(f"{EMOTIONAL_PREFIX}negativo_ratio")
                / (pl.col(f"{EMOTIONAL_PREFIX}positivo_ratio") + 0.01)
            ).alias(f"{EMOTIONAL_PREFIX}negative_positive_ratio")
        )
        .sort(USER_ID)
    )


def compute_sentiment_confidence(
    tweets: pl.DataFrame,
    aggregations: list[str],
) -> pl.DataFrame:
    """Agrega o score de confiança do classificador de sentimento.

    Confiança média baixa indica um histórico ambíguo para o modelo — em si,
    uma informação sobre o usuário, e não apenas uma limitação do encoder.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets rotulados, com ``sentiment_score``.
    aggregations : list of str
        Agregações a aplicar.

    Returns
    -------
    pl.DataFrame
        Confiança agregada por usuário.

    Examples
    --------
    >>> compute_sentiment_confidence(tweets, ["mean", "std"])  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID, SENTIMENT_SCORE], context="confiança do sentimento")
    return (
        tweets.group_by(USER_ID)
        .agg(build_aggregations(SENTIMENT_SCORE, aggregations, EMOTIONAL_PREFIX))
        .sort(USER_ID)
    )


def compute_emotion_intensity(
    tweets: pl.DataFrame,
    aggregations: list[str],
) -> pl.DataFrame:
    """Agrega as emoções finas produzidas pelo encoder multi-rótulo.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets com colunas ``emotion_<nome>``.
    aggregations : list of str
        Agregações a aplicar.

    Returns
    -------
    pl.DataFrame
        Emoções agregadas por usuário, e a intensidade emocional geral
        (média das emoções negativas).

    Examples
    --------
    >>> compute_emotion_intensity(tweets, ["mean", "max"])  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID], context="intensidade emocional")

    emotion_columns = [column for column in tweets.columns if column.startswith(EMOTION_PREFIX)]
    if not emotion_columns:
        logger.warning(
            "Nenhuma coluna '%s*' encontrada: a intensidade emocional será omitida. "
            "Ative labeling.emotion em configs/labeling.yaml.",
            EMOTION_PREFIX,
        )
        return tweets.select(USER_ID).unique().sort(USER_ID)

    expressions: list[pl.Expr] = []
    for column in emotion_columns:
        expressions.extend(build_aggregations(column, aggregations, EMOTIONAL_PREFIX))

    negative_emotions = [
        column
        for column in emotion_columns
        if column.removeprefix(EMOTION_PREFIX) in {"tristeza", "raiva", "medo", "nojo"}
    ]
    if negative_emotions:
        expressions.append(
            pl.mean_horizontal([pl.col(column) for column in negative_emotions])
            .mean()
            .alias(f"{EMOTIONAL_PREFIX}negative_intensity_mean")
        )

    return tweets.group_by(USER_ID).agg(expressions).sort(USER_ID)


def build_emotional_features(
    tweets: pl.DataFrame,
    config: EmotionalSection,
) -> pl.DataFrame:
    """Monta todas as features emocionais por usuário.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets rotulados por sentimento e emoção.
    config : EmotionalSection
        Seção ``emotional`` de ``configs/features.yaml``.

    Returns
    -------
    pl.DataFrame
        Uma linha por usuário, com as colunas de prefixo ``emo_``.

    Examples
    --------
    >>> build_emotional_features(tweets, config.features.emotional)  # doctest: +SKIP
    """
    frames: list[pl.DataFrame] = []

    if config.sentiment_distribution:
        frames.append(compute_sentiment_distribution(tweets))
    if config.sentiment_confidence:
        frames.append(compute_sentiment_confidence(tweets, config.aggregations))
    if config.emotion_intensity:
        frames.append(compute_emotion_intensity(tweets, config.aggregations))

    if not frames:
        return tweets.select(USER_ID).unique().sort(USER_ID)

    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on=USER_ID, how="full", coalesce=True)

    logger.info(
        "Features emocionais: %d colunas para %d usuários.", result.width - 1, result.height
    )
    return result.sort(USER_ID)
