"""Atributos temporais e circadianos por usuário (prefixo ``temp_``).

Cobre a Seção 4 da proposta e é o grupo que sustenta a hipótese H2 — que a
informação temporal e comportamental melhora a detecção. É também o grupo que
só existe **porque** a modelagem é centrada no usuário: nenhuma dessas
features pode ser calculada a partir de um tweet isolado.

Uma decisão explícita: quando o histórico é curto demais para estimar uma
tendência (menos de ``min_days_for_trend`` dias), a feature recebe ``NaN`` em
vez de zero. Zero significaria "tendência plana", que é uma afirmação sobre o
usuário; ``NaN`` significa "não sabemos", que é a verdade — e o indicador de
ausência criado em :mod:`features.builder` deixa o modelo aprender isso.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from config.logging import get_logger
from config.settings import TemporalSection
from constants.columns import (
    CREATED_AT,
    SENTIMENT_POLARITY,
    TEMPORAL_PREFIX,
    TEXT_CLEAN,
    USER_ID,
)
from utils.lexicons import load_lexicons
from utils.validation import require_columns

logger = get_logger(__name__)


def compute_volume(tweets: pl.DataFrame) -> pl.DataFrame:
    """Calcula o volume de publicação por usuário.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets com ``user_id`` e ``created_at``.

    Returns
    -------
    pl.DataFrame
        Tweets por dia, por semana, dias ativos, amplitude do histórico e
        taxa de dias ativos.

    Examples
    --------
    >>> compute_volume(tweets)  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID, CREATED_AT], context="volume temporal")

    return (
        tweets.group_by(USER_ID)
        .agg(
            pl.len().alias("_n_tweets"),
            pl.col(CREATED_AT).dt.date().n_unique().alias("_active_days"),
            (pl.col(CREATED_AT).max() - pl.col(CREATED_AT).min())
            .dt.total_days()
            .alias("_span_days"),
        )
        .with_columns(
            (pl.col("_n_tweets") / pl.max_horizontal(pl.col("_span_days"), pl.lit(1))).alias(
                f"{TEMPORAL_PREFIX}tweets_per_day"
            ),
            (pl.col("_n_tweets") / pl.max_horizontal(pl.col("_span_days") / 7, pl.lit(1))).alias(
                f"{TEMPORAL_PREFIX}tweets_per_week"
            ),
            (pl.col("_n_tweets") / pl.max_horizontal(pl.col("_active_days"), pl.lit(1))).alias(
                f"{TEMPORAL_PREFIX}tweets_per_active_day"
            ),
            (pl.col("_active_days") / pl.max_horizontal(pl.col("_span_days"), pl.lit(1))).alias(
                f"{TEMPORAL_PREFIX}active_day_rate"
            ),
            pl.col("_span_days").cast(pl.Float64).alias(f"{TEMPORAL_PREFIX}span_days"),
            pl.col("_active_days").cast(pl.Float64).alias(f"{TEMPORAL_PREFIX}active_days"),
        )
        .drop(["_n_tweets", "_active_days", "_span_days"])
        .sort(USER_ID)
    )


def compute_circadian(tweets: pl.DataFrame, config: TemporalSection) -> pl.DataFrame:
    """Calcula atividade noturna e regularidade do ritmo circadiano.

    A entropia da distribuição horária mede quão espalhada é a atividade nas
    24 horas: valores altos indicam ausência de rotina de sono, o que a
    literatura associa a quadros depressivos. É normalizada por ``log(24)``
    para ficar em ``[0, 1]`` e ser comparável entre usuários.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets com ``user_id`` e ``created_at``.
    config : TemporalSection
        Seção ``temporal`` de ``configs/features.yaml``.

    Returns
    -------
    pl.DataFrame
        Razão de atividade noturna, hora média de pico e entropia circadiana.

    Examples
    --------
    >>> compute_circadian(tweets, config.features.temporal)  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID, CREATED_AT], context="ritmo circadiano")

    start, end = config.insomnia_window
    hour = pl.col(CREATED_AT).dt.hour()
    is_night = (hour >= start) & (hour <= end) if start <= end else (hour >= start) | (hour <= end)

    frame = tweets.with_columns(hour.alias("_hour"), is_night.alias("_is_night"))

    base = frame.group_by(USER_ID).agg(
        pl.col("_is_night").mean().alias(f"{TEMPORAL_PREFIX}night_activity_ratio"),
        pl.col("_hour").mean().alias(f"{TEMPORAL_PREFIX}mean_hour"),
        pl.col("_hour").std().alias(f"{TEMPORAL_PREFIX}hour_std"),
    )

    if not config.circadian_entropy:
        return base.sort(USER_ID)

    records: list[dict[str, float | str]] = []
    for (user_id,), user_frame in frame.partition_by(
        USER_ID, as_dict=True, maintain_order=True
    ).items():
        counts = np.bincount(user_frame["_hour"].to_numpy(), minlength=24).astype(float)
        total = counts.sum()
        if total == 0:
            entropy = 0.0
        else:
            probabilities = counts / total
            nonzero = probabilities[probabilities > 0]
            entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(24))

        records.append({USER_ID: user_id, f"{TEMPORAL_PREFIX}circadian_entropy": entropy})

    return base.join(pl.DataFrame(records), on=USER_ID, how="left").sort(USER_ID)


def _linear_slope(days: np.ndarray, values: np.ndarray, min_points: int) -> float:
    """Calcula a inclinação da reta de mínimos quadrados, ou ``NaN`` se insuficiente."""
    if len(days) < min_points or np.all(days == days[0]):
        return float("nan")
    slope, _ = np.polyfit(days, values, deg=1)
    return float(slope)


def _compute_recent_polarity_stats(
    days: np.ndarray, polarity: np.ndarray, recent_window_days: float
) -> tuple[float, float]:
    """Calcula a média recente da polaridade e seu desvio em relação à média global.

    Janela recente: o estado atual do usuário pesa mais que a média de um ano
    inteiro para uma decisão de triagem.
    """
    horizon = float(days.max()) - recent_window_days
    recent_mask = days >= horizon
    if recent_mask.sum() < 3:
        return float("nan"), float("nan")

    recent_mean = float(polarity[recent_mask].mean())
    return recent_mean, recent_mean - float(polarity.mean())


def _compute_negative_streak(user_frame: pl.DataFrame) -> float:
    """Calcula a maior sequência de dias consecutivos com polaridade média negativa."""
    daily = (
        user_frame.group_by(pl.col(CREATED_AT).dt.date().alias("_day"))
        .agg(pl.col(SENTIMENT_POLARITY).mean().alias("_polarity"))
        .sort("_day")
    )
    streak = best = 0
    for value in daily["_polarity"].to_list():
        streak = streak + 1 if value is not None and value < 0 else 0
        best = max(best, streak)
    return float(best)


def _compute_monthly_polarity_shift(user_frame: pl.DataFrame) -> tuple[float, float]:
    """Calcula o desvio-padrão e a variação da polaridade média mensal."""
    monthly = (
        user_frame.group_by(pl.col(CREATED_AT).dt.truncate("1mo").alias("_month"))
        .agg(pl.col(SENTIMENT_POLARITY).mean().alias("_polarity"))
        .sort("_month")
    )
    values = monthly["_polarity"].to_numpy()
    if len(values) <= 1:
        return float("nan"), float("nan")

    return float(np.nanstd(values)), float(values[-1] - values[0])


def compute_sentiment_trend(tweets: pl.DataFrame, config: TemporalSection) -> pl.DataFrame:
    """Calcula tendência e variação temporal do sentimento.

    A inclinação da regressão da polaridade contra o tempo distingue um
    usuário que está piorando de outro estável — distinção que a média do
    sentimento, sozinha, não faz.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets com ``created_at`` e ``sentiment_polarity``.
    config : TemporalSection
        Seção ``temporal`` de ``configs/features.yaml``.

    Returns
    -------
    pl.DataFrame
        Inclinação global, polaridade recente, delta recente e maior
        sequência de dias negativos.

    Examples
    --------
    >>> compute_sentiment_trend(tweets, config.features.temporal)  # doctest: +SKIP
    """
    require_columns(
        tweets, [USER_ID, CREATED_AT, SENTIMENT_POLARITY], context="tendência de sentimento"
    )

    records: list[dict[str, float | str]] = []
    for (user_id,), user_frame in (
        tweets.sort(CREATED_AT).partition_by(USER_ID, as_dict=True, maintain_order=True).items()
    ):
        timestamps = user_frame[CREATED_AT]
        polarity = user_frame[SENTIMENT_POLARITY].to_numpy()
        origin = timestamps.min()
        days = np.array(
            [(value - origin).total_seconds() / 86400.0 for value in timestamps.to_list()]
        )

        record: dict[str, float | str] = {
            USER_ID: user_id,
            f"{TEMPORAL_PREFIX}polarity_slope": _linear_slope(
                days, polarity, config.min_days_for_trend
            ),
        }

        recent_mean, recent_delta = _compute_recent_polarity_stats(
            days, polarity, config.recent_window_days
        )
        record[f"{TEMPORAL_PREFIX}polarity_recent_mean"] = recent_mean
        record[f"{TEMPORAL_PREFIX}polarity_recent_delta"] = recent_delta

        if config.negative_persistence:
            record[f"{TEMPORAL_PREFIX}negative_streak_days"] = _compute_negative_streak(user_frame)

        if config.polarity_shift:
            monthly_std, monthly_shift = _compute_monthly_polarity_shift(user_frame)
            record[f"{TEMPORAL_PREFIX}monthly_polarity_std"] = monthly_std
            record[f"{TEMPORAL_PREFIX}monthly_polarity_shift"] = monthly_shift

        records.append(record)

    return pl.DataFrame(records).sort(USER_ID)


def _detect_risk_terms(text: str, lexicons: dict, risk_names: list[str]) -> bool:
    """Verifica se o texto contém algum termo de um léxico de risco."""
    return any(lexicons[name].contains(text) for name in risk_names)


def _build_risk_intensification_record(
    user_id: str, user_frame: pl.DataFrame, config: TemporalSection
) -> dict[str, float | str]:
    """Calcula a inclinação e a densidade recente de risco para um usuário."""
    daily = (
        user_frame.group_by(pl.col(CREATED_AT).dt.date().alias("_day"))
        .agg(pl.col("_risk").mean().alias("_density"))
        .sort("_day")
    )
    days_list = daily["_day"].to_list()
    origin = days_list[0]
    days = np.array([(value - origin).days for value in days_list], dtype=float)
    density = daily["_density"].to_numpy().astype(float)

    recent_mask = days >= (days.max() - config.recent_window_days)
    return {
        USER_ID: user_id,
        f"{TEMPORAL_PREFIX}risk_density_slope": _linear_slope(
            days, density, config.min_days_for_trend
        ),
        f"{TEMPORAL_PREFIX}risk_density_mean": float(density.mean()),
        f"{TEMPORAL_PREFIX}risk_density_recent": float(density[recent_mask].mean())
        if recent_mask.any()
        else float("nan"),
    }


def compute_risk_intensification(tweets: pl.DataFrame, config: TemporalSection) -> pl.DataFrame:
    """Mede se a densidade de termos de risco cresce ao longo do tempo.

    Uma inclinação positiva indica agravamento — sinal clinicamente mais
    relevante do que o nível absoluto, que pode ser alto e estável.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets com ``created_at`` e ``text_clean``.
    config : TemporalSection
        Seção ``temporal`` de ``configs/features.yaml``.

    Returns
    -------
    pl.DataFrame
        Inclinação da densidade de risco e densidade na janela recente.

    Examples
    --------
    >>> compute_risk_intensification(tweets, config.features.temporal)  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID, CREATED_AT, TEXT_CLEAN], context="intensificação de risco")

    lexicons = load_lexicons()
    risk_names = [name for name in ("death", "hopelessness", "loneliness") if name in lexicons]
    if not risk_names:
        logger.warning("Sem léxicos de risco: intensificação temporal não calculada.")
        return tweets.select(USER_ID).unique().sort(USER_ID)

    frame = tweets.with_columns(
        pl.col(TEXT_CLEAN)
        .map_elements(
            lambda text: _detect_risk_terms(text, lexicons, risk_names), return_dtype=pl.Boolean
        )
        .alias("_risk")
    ).sort(CREATED_AT)

    records = [
        _build_risk_intensification_record(user_id, user_frame, config)
        for (user_id,), user_frame in frame.partition_by(
            USER_ID, as_dict=True, maintain_order=True
        ).items()
    ]

    return pl.DataFrame(records).sort(USER_ID)


def _collect_activity_frames(tweets: pl.DataFrame, config: TemporalSection) -> list[pl.DataFrame]:
    """Calcula os subgrupos de volume e ritmo circadiano, quando habilitados."""
    frames: list[pl.DataFrame] = []
    if config.volume:
        frames.append(compute_volume(tweets))
    if config.night_activity:
        frames.append(compute_circadian(tweets, config))
    return frames


def _collect_trend_frames(tweets: pl.DataFrame, config: TemporalSection) -> list[pl.DataFrame]:
    """Calcula os subgrupos de tendência de sentimento e intensificação de risco."""
    frames: list[pl.DataFrame] = []
    if config.sentiment_trend and SENTIMENT_POLARITY in tweets.columns:
        frames.append(compute_sentiment_trend(tweets, config))
    if config.risk_intensification:
        frames.append(compute_risk_intensification(tweets, config))
    return frames


def _collect_temporal_frames(tweets: pl.DataFrame, config: TemporalSection) -> list[pl.DataFrame]:
    """Calcula todos os subgrupos de features temporais habilitados na configuração."""
    return _collect_activity_frames(tweets, config) + _collect_trend_frames(tweets, config)


def _join_temporal_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Une os subgrupos de features temporais pelo identificador de usuário."""
    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on=USER_ID, how="full", coalesce=True)
    return result


def build_temporal_features(
    tweets: pl.DataFrame,
    config: TemporalSection,
) -> pl.DataFrame:
    """Monta todas as features temporais por usuário.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos e rotulados.
    config : TemporalSection
        Seção ``temporal`` de ``configs/features.yaml``.

    Returns
    -------
    pl.DataFrame
        Uma linha por usuário, com as colunas de prefixo ``temp_``.

    Examples
    --------
    >>> build_temporal_features(tweets, config.features.temporal)  # doctest: +SKIP
    """
    frames = _collect_temporal_frames(tweets, config)

    if not frames:
        return tweets.select(USER_ID).unique().sort(USER_ID)

    result = _join_temporal_frames(frames)

    logger.info("Features temporais: %d colunas para %d usuários.", result.width - 1, result.height)
    return result.sort(USER_ID)
