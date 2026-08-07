"""Atributos linguísticos agregados por usuário (prefixo ``ling_``).

Cobre a Seção 1 da proposta: densidade de léxicos de risco, comprimento dos
textos, diversidade lexical e uso de pronomes.

Duas escolhas metodológicas explícitas:

* **Razão de tweets, não contagem bruta.** Um usuário que menciona "sozinho"
  dez vezes num único tweet não apresenta a mesma evidência que dez tweets
  distintos mencionando solidão. As features de léxico medem a *proporção de
  publicações* que contêm o termo.
* **Diversidade lexical corrigida por comprimento.** A *type-token ratio*
  simples cai mecanicamente à medida que o texto cresce, então um usuário
  com histórico longo pareceria "menos diverso" só por ter escrito mais. O
  índice de Guiraud (tipos / raiz de tokens) é reportado ao lado, por ser
  muito menos sensível ao tamanho da amostra.
"""

from __future__ import annotations

import math

import polars as pl

from config.logging import get_logger
from config.settings import LinguisticSection
from constants.columns import LINGUISTIC_PREFIX, TEXT_CLEAN, USER_ID
from constants.defaults import NEGATION_TERMS, PRONOUN_GROUPS
from utils.lexicons import load_lexicons, normalize_term
from utils.validation import require_columns

logger = get_logger(__name__)


def _annotate_lexicon_hits(
    frame: pl.DataFrame, lexicons: dict, available: list[str]
) -> pl.DataFrame:
    """Anota, por tweet, se cada léxico ocorre no texto e quantas vezes."""
    for name in available:
        lexicon = lexicons[name]
        frame = frame.with_columns(
            pl.col(TEXT_CLEAN)
            .map_elements(lexicon.contains, return_dtype=pl.Boolean)
            .alias(f"_has_{name}"),
            pl.col(TEXT_CLEAN)
            .map_elements(lexicon.count, return_dtype=pl.Int64)
            .alias(f"_hits_{name}"),
        )
    return frame


def _build_lexicon_aggregations(available: list[str]) -> list[pl.Expr]:
    """Monta as expressões de razão e média de ocorrências por léxico."""
    aggregations: list[pl.Expr] = []
    for name in available:
        aggregations.extend(
            (
                pl.col(f"_has_{name}").mean().alias(f"{LINGUISTIC_PREFIX}{name}_ratio"),
                pl.col(f"_hits_{name}").mean().alias(f"{LINGUISTIC_PREFIX}{name}_hits_per_tweet"),
            )
        )
    return aggregations


def compute_lexicon_ratios(tweets: pl.DataFrame, lexicon_names: list[str]) -> pl.DataFrame:
    """Calcula a proporção de tweets do usuário que contêm cada léxico.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos, com ``user_id`` e ``text_clean``.
    lexicon_names : list of str
        Léxicos a considerar (ex.: ``["death", "loneliness"]``).

    Returns
    -------
    pl.DataFrame
        Uma linha por usuário, com ``ling_<lexico>_ratio`` e
        ``ling_<lexico>_hits_per_tweet``.

    Examples
    --------
    >>> compute_lexicon_ratios(tweets, ["death"])  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID, TEXT_CLEAN], context="razões lexicais")

    lexicons = load_lexicons()
    available = [name for name in lexicon_names if name in lexicons]

    missing = set(lexicon_names) - set(available)
    if missing:
        logger.warning("Léxicos indisponíveis, ignorados nas features: %s.", sorted(missing))

    if not available:
        return tweets.select(USER_ID).unique().sort(USER_ID)

    frame = _annotate_lexicon_hits(tweets, lexicons, available)
    aggregations = _build_lexicon_aggregations(available)

    return frame.group_by(USER_ID).agg(aggregations).sort(USER_ID)


def compute_text_length(tweets: pl.DataFrame) -> pl.DataFrame:
    """Calcula estatísticas de comprimento dos textos por usuário.

    O desvio-padrão acompanha a média porque descreve regularidade: um perfil
    que alterna textos longos e curtos tem comportamento distinto de outro
    que escreve sempre no mesmo tamanho, mesmo com médias iguais.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos.

    Returns
    -------
    pl.DataFrame
        Média, desvio e mediana de caracteres e de tokens por usuário.

    Examples
    --------
    >>> compute_text_length(tweets)  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID, TEXT_CLEAN], context="comprimento dos textos")

    chars = pl.col(TEXT_CLEAN).str.len_chars()
    tokens = pl.col(TEXT_CLEAN).str.split(" ").list.len()

    return (
        tweets.group_by(USER_ID)
        .agg(
            chars.mean().alias(f"{LINGUISTIC_PREFIX}chars_mean"),
            chars.std().alias(f"{LINGUISTIC_PREFIX}chars_std"),
            chars.median().alias(f"{LINGUISTIC_PREFIX}chars_median"),
            tokens.mean().alias(f"{LINGUISTIC_PREFIX}tokens_mean"),
            tokens.std().alias(f"{LINGUISTIC_PREFIX}tokens_std"),
        )
        .sort(USER_ID)
    )


def _tokenize_user_text(texts: list[str]) -> list[str]:
    """Tokeniza os textos de um usuário em uma lista única de tokens (sem normalização)."""
    return [token for text in texts for token in str(text).split() if token]


def _build_diversity_record(user_id: str, tokens: list[str]) -> dict[str, float | str]:
    """Calcula TTR, índice de Guiraud e tamanho do vocabulário para um usuário."""
    total = len(tokens)
    types = len(set(tokens))
    return {
        USER_ID: user_id,
        f"{LINGUISTIC_PREFIX}ttr": types / total if total else 0.0,
        f"{LINGUISTIC_PREFIX}guiraud": types / math.sqrt(total) if total else 0.0,
        f"{LINGUISTIC_PREFIX}vocabulary_size": float(types),
    }


def compute_lexical_diversity(tweets: pl.DataFrame) -> pl.DataFrame:
    """Calcula a diversidade lexical do histórico de cada usuário.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos.

    Returns
    -------
    pl.DataFrame
        ``ling_ttr`` (type-token ratio), ``ling_guiraud`` (índice corrigido
        por comprimento) e ``ling_vocabulary_size``.

    Examples
    --------
    >>> frame = pl.DataFrame({"user_id": ["u_a"], "text_clean": ["oi oi tudo bem"]})
    >>> resultado = compute_lexical_diversity(frame)
    >>> round(resultado["ling_ttr"][0], 3)
    0.75
    """
    require_columns(tweets, [USER_ID, TEXT_CLEAN], context="diversidade lexical")

    records: list[dict[str, float | str]] = []
    for (user_id,), user_frame in tweets.partition_by(
        USER_ID, as_dict=True, maintain_order=True
    ).items():
        tokens = _tokenize_user_text(user_frame[TEXT_CLEAN].to_list())
        records.append(_build_diversity_record(user_id, tokens))

    return pl.DataFrame(records).sort(USER_ID)


def _normalize_pronoun_groups(groups: dict[str, list[str]]) -> dict[str, set[str]]:
    """Normaliza os termos de cada grupo pronominal para comparação."""
    return {group: {normalize_term(term) for term in terms} for group, terms in groups.items()}


def _tokenize_user_text_normalized(texts: list[str]) -> list[str]:
    """Tokeniza e normaliza os textos de um usuário em uma lista única de tokens."""
    return [normalize_term(token) for text in texts for token in str(text).split() if token]


def _count_pronoun_groups(
    tokens: list[str], normalized_groups: dict[str, set[str]]
) -> dict[str, int]:
    """Conta as ocorrências de tokens em cada grupo pronominal."""
    return {
        group: sum(1 for token in tokens if token in terms)
        for group, terms in normalized_groups.items()
    }


def _build_pronoun_record(
    user_id: str,
    tokens: list[str],
    counts: dict[str, int],
    normalized_negations: set[str],
) -> dict[str, float | str]:
    """Monta o registro de uso de pronomes e negação de um usuário."""
    total = max(len(tokens), 1)

    record: dict[str, float | str] = {USER_ID: user_id}
    for group, count in counts.items():
        record[f"{LINGUISTIC_PREFIX}pronoun_{group}"] = count / total

    # +1 no denominador evita divisão por zero e satura a razão para
    # usuários que nunca usam a primeira pessoa do plural.
    record[f"{LINGUISTIC_PREFIX}pronoun_i_we_ratio"] = counts["first_singular"] / (
        counts["first_plural"] + 1
    )
    record[f"{LINGUISTIC_PREFIX}negation_ratio"] = (
        sum(1 for token in tokens if token in normalized_negations) / total
    )
    return record


def compute_pronoun_usage(tweets: pl.DataFrame) -> pl.DataFrame:
    """Calcula a proporção de pronomes por pessoa gramatical.

    O uso elevado de pronomes de primeira pessoa do singular é um dos achados
    mais replicados da literatura de detecção de depressão: reflete foco
    atencional em si mesmo. A razão ``eu``/``nós`` é reportada separadamente
    por capturar isolamento social melhor do que qualquer das duas isoladas.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos.

    Returns
    -------
    pl.DataFrame
        Proporções por grupo de pronome, razão eu/nós e taxa de negação.

    Examples
    --------
    >>> frame = pl.DataFrame({"user_id": ["u_a"], "text_clean": ["eu não estou bem"]})
    >>> compute_pronoun_usage(frame)["ling_pronoun_first_singular"][0] > 0
    True
    """
    require_columns(tweets, [USER_ID, TEXT_CLEAN], context="uso de pronomes")

    normalized_groups = _normalize_pronoun_groups(PRONOUN_GROUPS)
    normalized_negations = {normalize_term(term) for term in NEGATION_TERMS}

    records: list[dict[str, float | str]] = []
    for (user_id,), user_frame in tweets.partition_by(
        USER_ID, as_dict=True, maintain_order=True
    ).items():
        tokens = _tokenize_user_text_normalized(user_frame[TEXT_CLEAN].to_list())
        counts = _count_pronoun_groups(tokens, normalized_groups)
        records.append(_build_pronoun_record(user_id, tokens, counts, normalized_negations))

    return pl.DataFrame(records).sort(USER_ID)


def _collect_linguistic_frames(
    tweets: pl.DataFrame, config: LinguisticSection
) -> list[pl.DataFrame]:
    """Calcula os subgrupos de features linguísticas habilitados na configuração."""
    frames: list[pl.DataFrame] = []
    if config.lexicon_ratios:
        frames.append(compute_lexicon_ratios(tweets, config.lexicon_ratios))
    if config.text_length:
        frames.append(compute_text_length(tweets))
    if config.lexical_diversity:
        frames.append(compute_lexical_diversity(tweets))
    if config.pronouns:
        frames.append(compute_pronoun_usage(tweets))
    return frames


def _join_linguistic_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Une os subgrupos de features linguísticas pelo identificador de usuário."""
    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on=USER_ID, how="full", coalesce=True)
    return result


def build_linguistic_features(
    tweets: pl.DataFrame,
    config: LinguisticSection,
) -> pl.DataFrame:
    """Monta todas as features linguísticas por usuário.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos.
    config : LinguisticSection
        Seção ``linguistic`` de ``configs/features.yaml``.

    Returns
    -------
    pl.DataFrame
        Uma linha por usuário, com as colunas de prefixo ``ling_``.

    Notes
    -----
    Os n-grams **não** são calculados aqui. O vetorizador TF-IDF precisa ser
    ajustado apenas no split de treino, senão o vocabulário do teste vaza
    para o modelo; por isso ele vive dentro do ``Pipeline`` do scikit-learn
    (ver :mod:`features.ngrams`).

    Examples
    --------
    >>> build_linguistic_features(tweets, config.features.linguistic)  # doctest: +SKIP
    """
    frames = _collect_linguistic_frames(tweets, config)

    if not frames:
        return tweets.select(USER_ID).unique().sort(USER_ID)

    result = _join_linguistic_frames(frames)

    logger.info(
        "Features linguísticas: %d colunas para %d usuários.", result.width - 1, result.height
    )
    return result.sort(USER_ID)
