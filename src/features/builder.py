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

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import polars as pl

from config.logging import get_logger
from config.settings import FeaturesConfig
from constants.columns import (
    ACTIVE_DAYS,
    CREATED_AT,
    EMOTIONAL_PREFIX,
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

#: Atributos idênticos (ou função direta) dos valores usados como limiar em
#: ``labeling.weak_supervision.label_from_lexical_evidence`` para ATRIBUIR o
#: rótulo do usuário (``negative_ratio`` — limiar ``min_negative_ratio`` da
#: classe depressão). Os léxicos ``death``/``loneliness``/``hopelessness`` já
#: são excluídos na origem, em ``configs/features.yaml``; estes dois são
#: calculados a partir do classificador de sentimento e não têm um toggle
#: próprio, por isso são removidos aqui, na montagem final da matriz. Mantê-
#: los como atributo do classificador vazaria o rótulo para o modelo — a
#: "predição" reconstruiria a própria regra de rotulação, não um padrão
#: independente.
LABEL_LEAKING_COLUMNS: Final[tuple[str, ...]] = (
    f"{EMOTIONAL_PREFIX}negativo_ratio",
    f"{EMOTIONAL_PREFIX}negative_positive_ratio",  # função direta de negativo_ratio
)


def _drop_label_leaking_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Remove da matriz os atributos que vazam a regra de rotulação do usuário.

    Parameters
    ----------
    frame : pl.DataFrame
        Matriz de atributos com todos os grupos já unidos.

    Returns
    -------
    pl.DataFrame
        Matriz sem as colunas listadas em :data:`LABEL_LEAKING_COLUMNS`.

    Examples
    --------
    >>> _drop_label_leaking_columns(pl.DataFrame({"emo_negativo_ratio": [0.1]})).columns
    []
    """
    present = [column for column in LABEL_LEAKING_COLUMNS if column in frame.columns]
    if present:
        logger.warning("Atributos removidos por vazamento de rótulo: %s.", present)
    return frame.drop(present)


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


def _normalize_float_nan_to_null(frame: pl.DataFrame, feature_columns: list[str]) -> pl.DataFrame:
    """Converte ``NaN`` em ``null`` nas colunas float, para uniformizar a detecção de ausência.

    Colunas float podem trazer NaN (ex.: tendências temporais sem histórico
    suficiente — ver features/temporal.py), que o Polars NÃO conta em
    null_count(). Normalizamos para null aqui para que a detecção e a
    imputação enxerguem os dois casos da mesma forma.
    """
    float_columns = [column for column in feature_columns if frame.schema[column].is_float()]
    return frame.with_columns([pl.col(column).fill_nan(None) for column in float_columns])


def _add_missing_indicators(
    frame: pl.DataFrame,
    columns_with_nulls: list[str],
    config: FeaturesConfig,
) -> pl.DataFrame:
    """Cria as colunas indicadoras binárias de ausência, quando habilitado na configuração."""
    if not (config.aggregation.add_missing_indicators and columns_with_nulls):
        return frame

    result = frame.with_columns(
        [
            pl.col(column).is_null().cast(pl.Float64).alias(f"{column}{MISSING_INDICATOR_SUFFIX}")
            for column in columns_with_nulls
        ]
    )
    logger.info("Criados %d indicadores de ausência.", len(columns_with_nulls))
    return result


def _impute_missing(frame: pl.DataFrame, feature_columns: list[str], strategy: str) -> pl.DataFrame:
    """Aplica a estratégia de imputação configurada às colunas de atributos."""
    if strategy == "keep_nan":
        return frame

    if strategy == "zero":
        return frame.with_columns([pl.col(column).fill_null(0.0) for column in feature_columns])

    return frame.with_columns(
        [
            pl.col(column).fill_null(pl.col(column).median()).fill_null(0.0)
            for column in feature_columns
        ]
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

    result = _normalize_float_nan_to_null(frame, feature_columns)
    with_nulls = [column for column in feature_columns if result[column].null_count() > 0]
    result = _add_missing_indicators(result, with_nulls, config)

    return _impute_missing(result, feature_columns, config.aggregation.missing_strategy)


def _independent_group_builders(
    tweets: pl.DataFrame,
    enabled: set[str],
    config: FeaturesConfig,
    metadata: pl.DataFrame | None,
) -> dict[str, Callable[[], pl.DataFrame]]:
    """Mapeia nome -> construtor de cada grupo independente entre si, se habilitado.

    Linguístico, temporal e comportamental só leem ``tweets`` (e, no caso
    comportamental, ``metadata``) — nenhum consome a saída de outro grupo —,
    por isso podem rodar em paralelo em vez de em sequência. Emocional,
    semântico e psicológico ficam fora deste lote: dependem de artefatos
    externos (embeddings da etapa 5, vetores do LLM da etapa 4) e continuam
    unidos sequencialmente em :func:`build_user_features_raw`.
    """
    builders: dict[str, Callable[[], pl.DataFrame]] = {}
    if "linguistic" in enabled:
        builders["linguistic"] = lambda: build_linguistic_features(tweets, config.linguistic)
    if "temporal" in enabled:
        builders["temporal"] = lambda: build_temporal_features(tweets, config.temporal)
    if "behavioral" in enabled:
        builders["behavioral"] = lambda: build_behavioral_features(
            tweets, metadata, config.behavioral
        )
    return builders


def _build_independent_groups(
    tweets: pl.DataFrame,
    enabled: set[str],
    config: FeaturesConfig,
    metadata: pl.DataFrame | None,
) -> list[pl.DataFrame]:
    """Constrói em paralelo os grupos linguístico, temporal e comportamental habilitados.

    Cada grupo é independente dos demais (ver :func:`_independent_group_builders`),
    então roda em uma thread própria em vez de esperar os outros terminarem —
    o polars libera o GIL na maior parte do trabalho pesado (agregações,
    joins), então threads bastam, sem o custo de serializar ``tweets`` entre
    processos.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets do(s) usuário(s) sendo processados.
    enabled : set of str
        Grupos ativos em ``config.features.groups``.
    config : FeaturesConfig
        Seção ``features`` de ``configs/features.yaml``.
    metadata : pl.DataFrame, optional
        Metadados públicos dos usuários (features de audiência).

    Returns
    -------
    list of pl.DataFrame
        Um frame por grupo habilitado, pronto para ser unido a ``result``.
    """
    builders = _independent_group_builders(tweets, enabled, config, metadata)
    if not builders:
        return []

    with ThreadPoolExecutor(max_workers=len(builders)) as executor:
        futures = {name: executor.submit(builder) for name, builder in builders.items()}
        return [futures[name].result() for name in builders]


def _join_emotional_group(
    result: pl.DataFrame, tweets: pl.DataFrame, enabled: set[str], config: FeaturesConfig
) -> pl.DataFrame:
    """Une o grupo de features emocionais, quando habilitado."""
    if "emotional" in enabled:
        return result.join(
            build_emotional_features(tweets, config.emotional), on=USER_ID, how="left"
        )
    return result


def _join_semantic_group(
    result: pl.DataFrame, tweets: pl.DataFrame, enabled: set[str], config: FeaturesConfig
) -> pl.DataFrame:
    """Une o grupo de features semânticas, quando habilitado."""
    if "semantic" in enabled and config.semantic.enabled:
        return result.join(build_semantic_features(tweets, config.semantic), on=USER_ID, how="left")
    return result


def _join_psychological_group(
    result: pl.DataFrame,
    enabled: set[str],
    config: FeaturesConfig,
    psychological_scores: pl.DataFrame | None,
) -> pl.DataFrame:
    """Une o grupo de features psicológicas, quando habilitado."""
    if "psychological" in enabled and config.psychological.enabled:
        scores = psychological_scores if psychological_scores is not None else pl.DataFrame()
        return result.join(
            build_psychological_features(scores, config.psychological),
            on=USER_ID,
            how="left",
        )
    return result


def _promote_night_activity_column(result: pl.DataFrame) -> pl.DataFrame:
    """Promove a razão de atividade noturna a coluna de perfil, quando presente.

    Sustenta uma das fatias de avaliação em configs/evaluation.yaml.
    """
    night_column = f"{TEMPORAL_PREFIX}night_activity_ratio"
    if night_column in result.columns:
        return result.with_columns(pl.col(night_column).alias(NIGHT_ACTIVITY_RATIO))
    return result


def _filter_min_tweets(result: pl.DataFrame, minimum: int) -> pl.DataFrame:
    """Remove usuários abaixo do mínimo de tweets configurado, registrando o total removido."""
    before = result.height
    result = result.filter(pl.col(N_TWEETS) >= minimum)
    if before != result.height:
        logger.debug(
            "%d usuário(s) removido(s) por terem menos de %d tweets.",
            before - result.height,
            minimum,
        )
    return result


def build_user_features_raw(
    tweets: pl.DataFrame,
    config: FeaturesConfig,
    *,
    metadata: pl.DataFrame | None = None,
    psychological_scores: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Monta as colunas de atributos por usuário, sem imputação global nem rótulos.

    É a metade decomponível por usuário de :func:`build_user_features`: todos
    os seis grupos e o filtro de ``min_tweets_per_user`` dependem apenas dos
    dados do(s) usuário(s) recebidos em ``tweets`` — por isso esta função pode
    ser chamada com um único usuário por vez, o que sustenta o processamento
    incremental de ``pipelines.features.FeaturesStage``. A imputação por
    mediana e a junção com os rótulos ficam em :func:`finalize_user_features`,
    que precisa da população inteira e por isso roda uma única vez, no final.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos e rotulados por sentimento (de um ou mais usuários).
    config : FeaturesConfig
        Seção ``features`` de ``configs/features.yaml``.
    metadata : pl.DataFrame, optional
        Metadados públicos dos usuários (features de audiência).
    psychological_scores : pl.DataFrame, optional
        Vetores psicológicos extraídos pelo LLM.

    Returns
    -------
    pl.DataFrame
        Uma linha por usuário que atingiu ``aggregation.min_tweets_per_user``
        (pode ficar vazio, com o mesmo esquema de colunas, se nenhum atingir).

    Examples
    --------
    >>> build_user_features_raw(tweets, config.features)  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID, CREATED_AT], context="matriz de atributos")

    enabled = set(config.enabled_groups())
    logger.info("Grupos de atributos ativos: %s.", sorted(enabled) or "nenhum")

    result = build_profile_columns(tweets)

    with log_duration("Construção da matriz de atributos"):
        for group_frame in _build_independent_groups(tweets, enabled, config, metadata):
            result = result.join(group_frame, on=USER_ID, how="left")
        result = _join_emotional_group(result, tweets, enabled, config)
        result = _join_semantic_group(result, tweets, enabled, config)
        result = _join_psychological_group(result, enabled, config, psychological_scores)

    result = _drop_label_leaking_columns(result)
    result = _promote_night_activity_column(result)

    return _filter_min_tweets(result, config.aggregation.min_tweets_per_user).sort(USER_ID)


def _drop_duplicate_users(raw: pl.DataFrame) -> pl.DataFrame:
    """Remove linhas duplicadas de ``user_id``, mantendo a primeira ocorrência.

    ``user_features_raw`` é acumulado de arquivos particionados por usuário
    (ver :func:`data.reader.read_partitioned`); se o mesmo pseudônimo aparecer
    em mais de um arquivo — por exemplo, uma reprocessamento parcial ou uma
    coleta duplicada do mesmo usuário —, a concatenação traria a linha
    repetida. Sem esta deduplicação, o usuário poderia ser sorteado tanto para
    treino quanto para teste em :func:`data.splitter.create_splits`, violando
    a garantia central do projeto de que a partição é por usuário.
    """
    duplicated = raw.filter(pl.col(USER_ID).is_duplicated())
    if duplicated.is_empty():
        return raw

    n_users = duplicated[USER_ID].n_unique()
    logger.warning(
        "Removidas linhas duplicadas de %d usuário(s) em user_features_raw (mesmo user_id).",
        n_users,
    )
    return raw.unique(subset=[USER_ID], keep="first", maintain_order=True)


def finalize_user_features(
    raw: pl.DataFrame,
    config: FeaturesConfig,
    *,
    labels: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Aplica a imputação global e une os rótulos à matriz de atributos acumulada.

    É a metade **não** decomponível por usuário de :func:`build_user_features`:
    a mediana usada na imputação (ver :func:`handle_missing_values`) precisa
    da população inteira, não de um usuário isolado. Roda uma única vez, sobre
    o acumulado de :func:`build_user_features_raw`.

    Parameters
    ----------
    raw : pl.DataFrame
        Saída acumulada de :func:`build_user_features_raw` (todos os usuários
        já processados).
    config : FeaturesConfig
        Seção ``features`` de ``configs/features.yaml``.
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
    >>> finalize_user_features(bruto, config.features, labels=rotulos)  # doctest: +SKIP
    """
    if raw.is_empty():
        raise InsufficientDataError(
            "Nenhum usuário atingiu o mínimo de "
            f"{config.aggregation.min_tweets_per_user} tweets "
            "(features.aggregation.min_tweets_per_user)."
        )

    raw = _drop_duplicate_users(raw)
    result = handle_missing_values(raw, config)

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


def build_user_features(
    tweets: pl.DataFrame,
    config: FeaturesConfig,
    *,
    metadata: pl.DataFrame | None = None,
    psychological_scores: pl.DataFrame | None = None,
    labels: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Monta a matriz completa de atributos por usuário.

    Fachada que encadeia :func:`build_user_features_raw` e
    :func:`finalize_user_features` num único lote — use estas duas funções
    diretamente para processar os usuários incrementalmente (ver
    ``pipelines.features.FeaturesStage``).

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
    raw = build_user_features_raw(
        tweets, config, metadata=metadata, psychological_scores=psychological_scores
    )
    return finalize_user_features(raw, config, labels=labels)


def _select_key_columns(frame: pl.DataFrame) -> list[str]:
    """Seleciona as colunas-chave (identificador, rótulo e perfil) presentes na matriz."""
    return [
        column
        for column in (USER_ID, USER_LABEL, N_TWEETS, SPAN_DAYS, ACTIVE_DAYS, NIGHT_ACTIVITY_RATIO)
        if column in frame.columns
    ]


def _select_missing_indicators(frame: pl.DataFrame, features: list[str]) -> list[str]:
    """Seleciona os indicadores de ausência que acompanham as features escolhidas."""
    return [
        column
        for column in frame.columns
        if column.endswith(MISSING_INDICATOR_SUFFIX)
        and column.removesuffix(MISSING_INDICATOR_SUFFIX) in features
    ]


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
    keys = _select_key_columns(frame)
    features = list_feature_columns(frame, groups)

    # Os indicadores de ausência acompanham o grupo da coluna que sinalizam.
    indicators = _select_missing_indicators(frame, features)

    return frame.select([*keys, *features, *indicators])
