"""Rotulação do usuário por supervisão fraca.

Rotular o **usuário** (e não cada tweet) é a decisão central da proposta: um
transtorno mental é uma condição persistente, e uma publicação isolada pode
ser ironia, sarcasmo, letra de música ou reação a um evento pontual.

Três fontes independentes votam, com pesos definidos em
``configs/labeling.yaml``:

1. **Grupo de coleta** — de qual consulta o usuário veio. Sinal forte, porém
   enviesado: quem posta ``#saudemental`` pode estar divulgando campanha.
2. **Evidência léxica** — densidade de termos de risco no histórico inteiro.
3. **Persistência temporal** — o sinal aparece em janelas distintas, e não
   concentrado num único episódio.

Quando as fontes não convergem o suficiente, o usuário é marcado como
``indefinido`` e descartado. Um rótulo ruidoso corrompe treino **e** avaliação
ao mesmo tempo, e nenhuma métrica revela isso.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from config.logging import get_logger
from config.settings import UserLabelingSection
from constants.columns import (
    CANDIDATE_LABEL,
    CREATED_AT,
    LABEL_AGREEMENT,
    LABEL_SOURCE,
    SENTIMENT,
    TEXT_CLEAN,
    USER_ID,
    USER_LABEL,
    USER_LABEL_MULTILABEL,
)
from constants.labels import CLASS_PRECEDENCE, Sentiment, UserLabel
from utils.lexicons import load_lexicons
from utils.validation import require_columns

logger = get_logger(__name__)


@dataclass(frozen=True)
class LabelVote:
    """Voto de uma fonte de rotulação.

    Attributes
    ----------
    source : str
        Nome da fonte (``collection_group``, ``lexical_evidence``, ...).
    label : str
        Classe votada.
    weight : float
        Peso da fonte no consenso.
    """

    source: str
    label: str
    weight: float


def compute_lexical_evidence(tweets: pl.DataFrame) -> pl.DataFrame:
    """Calcula a densidade de termos de risco por usuário.

    As razões são calculadas sobre a **proporção de tweets** que contêm ao
    menos um termo, e não sobre a contagem bruta de ocorrências: um único
    tweet repetindo "sozinho" dez vezes não é evidência equivalente a dez
    tweets distintos mencionando solidão.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos, com ``user_id``, ``text_clean`` e ``sentiment``.

    Returns
    -------
    pl.DataFrame
        Uma linha por usuário, com ``<lexico>_ratio``, ``negative_ratio``,
        ``death_hits`` e ``n_tweets``.

    Examples
    --------
    >>> compute_lexical_evidence(tweets)  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID, TEXT_CLEAN], context="evidência léxica")

    lexicons = load_lexicons()
    frame = tweets

    for name, lexicon in lexicons.items():
        frame = frame.with_columns(
            pl.col(TEXT_CLEAN)
            .map_elements(lexicon.contains, return_dtype=pl.Boolean)
            .alias(f"_has_{name}"),
            pl.col(TEXT_CLEAN)
            .map_elements(lexicon.count, return_dtype=pl.Int64)
            .alias(f"_hits_{name}"),
        )

    aggregations: list[pl.Expr] = [pl.len().alias("n_tweets")]
    for name in lexicons:
        aggregations.extend(
            (
                pl.col(f"_has_{name}").mean().alias(f"{name}_ratio"),
                pl.col(f"_hits_{name}").sum().alias(f"{name}_hits"),
            )
        )

    if SENTIMENT in frame.columns:
        aggregations.append(
            (pl.col(SENTIMENT) == str(Sentiment.NEGATIVO)).mean().alias("negative_ratio")
        )

    return frame.group_by(USER_ID).agg(aggregations).sort(USER_ID)


def compute_temporal_persistence(
    tweets: pl.DataFrame,
    config: UserLabelingSection,
) -> pl.DataFrame:
    """Mede se o sinal de risco persiste ao longo do tempo.

    Conta em quantas janelas distintas (por padrão, de 30 dias) o usuário
    apresentou termos de risco. É o critério que separa transtorno persistente
    de reação a um evento pontual — luto, término, uma notícia — que também
    produz linguagem negativa, mas concentrada.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos, com ``user_id``, ``text_clean`` e ``created_at``.
    config : UserLabelingSection
        Seção ``user_labeling`` de ``configs/labeling.yaml``.

    Returns
    -------
    pl.DataFrame
        Uma linha por usuário, com ``windows_with_signal``, ``span_days`` e
        ``has_persistence``.

    Examples
    --------
    >>> compute_temporal_persistence(tweets, config.labeling.user_labeling)  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID, TEXT_CLEAN, CREATED_AT], context="persistência temporal")

    lexicons = load_lexicons()
    risk_names = [name for name in ("death", "hopelessness", "loneliness") if name in lexicons]

    if not risk_names:
        logger.warning("Nenhum léxico de risco disponível: persistência temporal não calculada.")
        return tweets.group_by(USER_ID).agg(
            pl.lit(0).alias("windows_with_signal"),
            pl.lit(0).alias("span_days"),
            pl.lit(False).alias("has_persistence"),
        )

    settings = config.temporal_persistence
    has_risk = pl.lit(False)
    frame = tweets
    for name in risk_names:
        column = f"_risk_{name}"
        frame = frame.with_columns(
            pl.col(TEXT_CLEAN)
            .map_elements(lexicons[name].contains, return_dtype=pl.Boolean)
            .alias(column)
        )
        has_risk = has_risk | pl.col(column)

    return (
        frame.with_columns(has_risk.alias("_has_risk"))
        .with_columns(
            (
                (pl.col(CREATED_AT) - pl.col(CREATED_AT).min().over(USER_ID)).dt.total_days()
                // settings.window_days
            ).alias("_window")
        )
        .group_by(USER_ID)
        .agg(
            pl.col("_window").filter(pl.col("_has_risk")).n_unique().alias("windows_with_signal"),
            (pl.col(CREATED_AT).max() - pl.col(CREATED_AT).min())
            .dt.total_days()
            .alias("span_days"),
        )
        .with_columns(
            (
                (pl.col("windows_with_signal") >= settings.min_windows_with_signal)
                & (pl.col("span_days") >= settings.min_span_days)
            ).alias("has_persistence")
        )
        .sort(USER_ID)
    )


def label_from_lexical_evidence(row: dict[str, float], config: UserLabelingSection) -> str:
    """Deriva a classe a partir da evidência léxica de um usuário.

    A ideação suicida é avaliada primeiro por severidade: um usuário que
    atende aos dois critérios deve receber o rótulo mais grave.

    Parameters
    ----------
    row : dict
        Linha de :func:`compute_lexical_evidence`.
    config : UserLabelingSection
        Configuração da rotulação.

    Returns
    -------
    str
        Classe votada por esta fonte.

    Examples
    --------
    >>> label_from_lexical_evidence({"death_ratio": 0.1, "death_hits": 5}, cfg)  # doctest: +SKIP
    'ideacao_suicida'
    """
    thresholds = config.lexical_thresholds

    suicidal = thresholds.get("ideacao_suicida", {})
    if row.get("death_ratio", 0.0) >= suicidal.get("min_death_ratio", 1.0) or row.get(
        "death_hits", 0.0
    ) >= suicidal.get("min_suicidal_keyword_hits", float("inf")):
        return str(UserLabel.IDEACAO_SUICIDA)

    depression = thresholds.get("depressao", {})
    meets_negative = row.get("negative_ratio", 0.0) >= depression.get("min_negative_ratio", 1.0)
    meets_hopelessness = row.get("hopelessness_ratio", 0.0) >= depression.get(
        "min_hopelessness_ratio", 1.0
    )
    meets_loneliness = row.get("loneliness_ratio", 0.0) >= depression.get(
        "min_loneliness_ratio", 1.0
    )

    # Exige o sinal afetivo geral (negatividade) acompanhado de ao menos um
    # marcador específico: negatividade isolada também descreve alguém apenas
    # irritado ou sarcástico.
    if meets_negative and (meets_hopelessness or meets_loneliness):
        return str(UserLabel.DEPRESSAO)

    return str(UserLabel.CONTROLE)


def resolve_consensus(votes: list[LabelVote], config: UserLabelingSection) -> tuple[str, float]:
    """Combina os votos das fontes num rótulo final.

    Parameters
    ----------
    votes : list of LabelVote
        Votos das fontes ativas.
    config : UserLabelingSection
        Configuração da rotulação.

    Returns
    -------
    tuple
        ``(rótulo, concordância)``. A concordância é a fração do peso total
        que apoiou a classe vencedora; abaixo de ``consensus.min_agreement``,
        o rótulo vira ``indefinido``.

    Examples
    --------
    >>> resolve_consensus(
    ...     [LabelVote("a", "depressao", 0.6), LabelVote("b", "controle", 0.4)], cfg
    ... )  # doctest: +SKIP
    ('depressao', 0.6)
    """
    if not votes:
        return str(UserLabel.INDEFINIDO), 0.0

    scores: dict[str, float] = {}
    for vote in votes:
        scores[vote.label] = scores.get(vote.label, 0.0) + vote.weight

    total = sum(scores.values())
    if total <= 0:
        return str(UserLabel.INDEFINIDO), 0.0

    # Empate é resolvido pela precedência de severidade, não pela ordem de
    # inserção do dicionário — que seria arbitrária e não reprodutível.
    severity = {label: index for index, label in enumerate(CLASS_PRECEDENCE)}
    best = min(
        scores.items(),
        key=lambda item: (-item[1], severity.get(item[0], 99)),
    )
    label, weight = best
    agreement = weight / total

    if agreement < config.consensus.min_agreement:
        return str(UserLabel.INDEFINIDO), agreement
    return label, agreement


def _collect_candidate_labels(tweets: pl.DataFrame, lexical: pl.DataFrame) -> pl.DataFrame:
    """Deriva o rótulo candidato a partir do grupo de coleta mais frequente do usuário."""
    if "source_group" in tweets.columns:
        return (
            tweets.filter(pl.col("source_group").is_not_null())
            .group_by(USER_ID)
            .agg(pl.col("source_group").mode().first().alias(CANDIDATE_LABEL))
        )
    return lexical.select(USER_ID).with_columns(pl.lit(None, dtype=pl.Utf8).alias(CANDIDATE_LABEL))


def _collect_votes(row: dict[str, Any], config: UserLabelingSection) -> list[LabelVote]:
    """Coleta os votos de cada fonte ativa para um usuário.

    Parameters
    ----------
    row : dict
        Linha combinada de evidência léxica, persistência temporal e rótulo candidato.
    config : UserLabelingSection
        Configuração da rotulação.

    Returns
    -------
    list of LabelVote
        Votos das fontes ativas para este usuário.
    """
    sources = config.sources
    votes: list[LabelVote] = []

    group_source = sources.get("collection_group")
    if group_source and group_source.enabled and row.get(CANDIDATE_LABEL):
        votes.append(LabelVote("collection_group", str(row[CANDIDATE_LABEL]), group_source.weight))

    lexical_source = sources.get("lexical_evidence")
    if lexical_source and lexical_source.enabled:
        votes.append(
            LabelVote(
                "lexical_evidence",
                label_from_lexical_evidence(row, config),
                lexical_source.weight,
            )
        )

    temporal_source = sources.get("temporal_persistence")
    if temporal_source and temporal_source.enabled:
        # A persistência confirma o risco apontado pelo léxico; sozinha,
        # sua ausência vota em controle.
        lexical_label = label_from_lexical_evidence(row, config)
        persistent = bool(row.get("has_persistence", False))
        temporal_label = lexical_label if persistent else str(UserLabel.CONTROLE)
        votes.append(LabelVote("temporal_persistence", temporal_label, temporal_source.weight))

    return votes


def _build_user_record(row: dict[str, Any], config: UserLabelingSection) -> dict[str, object]:
    """Monta o registro final de rotulação de um usuário a partir dos votos das fontes."""
    votes = _collect_votes(row, config)
    label, agreement = resolve_consensus(votes, config)

    # Multirrótulo (análise de sensibilidade): registra todas as classes de
    # risco com evidência, preservando a coocorrência que o rótulo
    # multiclasse necessariamente colapsa.
    multilabel = sorted({vote.label for vote in votes if vote.label != str(UserLabel.CONTROLE)})

    return {
        USER_ID: row[USER_ID],
        USER_LABEL: label,
        USER_LABEL_MULTILABEL: "|".join(multilabel) if multilabel else None,
        CANDIDATE_LABEL: row.get(CANDIDATE_LABEL),
        LABEL_SOURCE: config.strategy,
        LABEL_AGREEMENT: agreement,
        "manual_label": None,
    }


def assign_user_labels(
    tweets: pl.DataFrame,
    config: UserLabelingSection,
) -> pl.DataFrame:
    """Atribui o rótulo final a cada usuário.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos e rotulados por sentimento.
    config : UserLabelingSection
        Seção ``user_labeling`` de ``configs/labeling.yaml``.

    Returns
    -------
    pl.DataFrame
        Conforme :class:`schemas.users.UserLabelSchema`.

    Examples
    --------
    >>> assign_user_labels(tweets, config.labeling.user_labeling)  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID, TEXT_CLEAN, CREATED_AT], context="rotulação de usuários")

    lexical = compute_lexical_evidence(tweets)
    persistence = compute_temporal_persistence(tweets, config)

    # Rótulo candidato: grupo de coleta mais frequente entre os tweets semente
    # do usuário (os tweets de histórico têm `source_group` nulo).
    candidates = _collect_candidate_labels(tweets, lexical)

    merged = lexical.join(persistence, on=USER_ID, how="left").join(
        candidates, on=USER_ID, how="left"
    )

    records = [_build_user_record(row, config) for row in merged.iter_rows(named=True)]

    result = pl.DataFrame(records)
    distribution = result.group_by(USER_LABEL).len().sort("len", descending=True)
    logger.info("Rótulos atribuídos: %s", dict(distribution.iter_rows()))
    return result
