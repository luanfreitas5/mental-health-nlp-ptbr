"""Nomes canônicos das colunas de cada estágio do pipeline.

Constantes em vez de literais soltos: renomear uma coluna passa a ser uma
alteração em um único lugar, e o *type checker* pega o erro de digitação que
uma string literal esconderia até a execução.

Os agrupamentos ``*_COLUMNS`` são usados pelos contratos pandera
(:mod:`schemas`) e pelas seleções de features (:mod:`features.builder`).
"""

from __future__ import annotations

from typing import Final

# --- Identificação ----------------------------------------------------------
#: Identificador pseudonimizado do usuário (SHA-256 com salt). Chave de tudo.
USER_ID: Final[str] = "user_id"
#: Identificador pseudonimizado do tweet.
TWEET_ID: Final[str] = "tweet_id"

# --- Tweet bruto / normalizado ----------------------------------------------
TEXT: Final[str] = "text"
TEXT_NORMALIZED: Final[str] = "text_normalized"
TEXT_CLEAN: Final[str] = "text_clean"
TOKENS: Final[str] = "tokens"
CREATED_AT: Final[str] = "created_at"
LANGUAGE: Final[str] = "language"
IS_REPLY: Final[str] = "is_reply"
IS_RETWEET: Final[str] = "is_retweet"
SOURCE_QUERY: Final[str] = "source_query"
SOURCE_GROUP: Final[str] = "source_group"

# --- Engajamento do tweet ---------------------------------------------------
LIKE_COUNT: Final[str] = "like_count"
REPLY_COUNT: Final[str] = "reply_count"
RETWEET_COUNT: Final[str] = "retweet_count"
QUOTE_COUNT: Final[str] = "quote_count"

# --- Metadados públicos do usuário ------------------------------------------
FOLLOWERS_COUNT: Final[str] = "followers_count"
FOLLOWING_COUNT: Final[str] = "following_count"
STATUSES_COUNT: Final[str] = "statuses_count"
ACCOUNT_CREATED_AT: Final[str] = "account_created_at"
IS_VERIFIED: Final[str] = "is_verified"

# --- Rotulação --------------------------------------------------------------
SENTIMENT: Final[str] = "sentiment"
SENTIMENT_SCORE: Final[str] = "sentiment_score"
SENTIMENT_POLARITY: Final[str] = "sentiment_polarity"
EMOTION_PREFIX: Final[str] = "emotion_"
USER_LABEL: Final[str] = "user_label"
USER_LABEL_MULTILABEL: Final[str] = "user_label_multilabel"
LABEL_SOURCE: Final[str] = "label_source"
LABEL_AGREEMENT: Final[str] = "label_agreement"
CANDIDATE_LABEL: Final[str] = "candidate_label"
MANUAL_LABEL: Final[str] = "manual_label"

# --- Partições --------------------------------------------------------------
SPLIT: Final[str] = "split"
FOLD: Final[str] = "fold"

# --- Prefixos dos grupos de atributos ---------------------------------------
# O prefixo é o que permite ao Ablation Study selecionar/remover um grupo
# inteiro sem manter uma lista explícita de centenas de nomes de colunas.
LINGUISTIC_PREFIX: Final[str] = "ling_"
EMOTIONAL_PREFIX: Final[str] = "emo_"
SEMANTIC_PREFIX: Final[str] = "sem_"
TEMPORAL_PREFIX: Final[str] = "temp_"
BEHAVIORAL_PREFIX: Final[str] = "behav_"
PSYCHOLOGICAL_PREFIX: Final[str] = "psy_"
MISSING_INDICATOR_SUFFIX: Final[str] = "_is_missing"

#: Grupo de atributos -> prefixo das suas colunas.
FEATURE_GROUP_PREFIXES: Final[dict[str, str]] = {
    "linguistic": LINGUISTIC_PREFIX,
    "emotional": EMOTIONAL_PREFIX,
    "semantic": SEMANTIC_PREFIX,
    "temporal": TEMPORAL_PREFIX,
    "behavioral": BEHAVIORAL_PREFIX,
    "psychological": PSYCHOLOGICAL_PREFIX,
}

# --- Colunas descritivas do perfil (usadas nas fatias de avaliação) ---------
N_TWEETS: Final[str] = "n_tweets"
SPAN_DAYS: Final[str] = "span_days"
ACTIVE_DAYS: Final[str] = "active_days"
FIRST_TWEET_AT: Final[str] = "first_tweet_at"
LAST_TWEET_AT: Final[str] = "last_tweet_at"
NIGHT_ACTIVITY_RATIO: Final[str] = "night_activity_ratio"

# --- Agrupamentos -----------------------------------------------------------

#: Colunas mínimas de um tweet coletado (contrato ``raw``).
RAW_TWEET_COLUMNS: Final[tuple[str, ...]] = (
    USER_ID,
    TWEET_ID,
    TEXT,
    CREATED_AT,
    LANGUAGE,
    IS_REPLY,
    IS_RETWEET,
    LIKE_COUNT,
    REPLY_COUNT,
    RETWEET_COUNT,
    QUOTE_COUNT,
    SOURCE_QUERY,
    SOURCE_GROUP,
)

#: Colunas adicionadas pela etapa de preprocessing.
CLEAN_TWEET_COLUMNS: Final[tuple[str, ...]] = (
    *RAW_TWEET_COLUMNS,
    TEXT_NORMALIZED,
    TEXT_CLEAN,
)

#: Colunas adicionadas pela etapa de rotulação.
LABELED_TWEET_COLUMNS: Final[tuple[str, ...]] = (
    *CLEAN_TWEET_COLUMNS,
    SENTIMENT,
    SENTIMENT_SCORE,
    SENTIMENT_POLARITY,
)

#: Metadados públicos do usuário (contrato ``users``).
USER_METADATA_COLUMNS: Final[tuple[str, ...]] = (
    USER_ID,
    FOLLOWERS_COUNT,
    FOLLOWING_COUNT,
    STATUSES_COUNT,
    ACCOUNT_CREATED_AT,
    IS_VERIFIED,
)

#: Colunas descritivas do perfil, presentes na matriz final ao lado das features.
PROFILE_COLUMNS: Final[tuple[str, ...]] = (
    N_TWEETS,
    SPAN_DAYS,
    ACTIVE_DAYS,
    FIRST_TWEET_AT,
    LAST_TWEET_AT,
    NIGHT_ACTIVITY_RATIO,
)

#: Colunas que nunca entram como preditor (identificadores, alvo, metadados).
NON_FEATURE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        USER_ID,
        USER_LABEL,
        USER_LABEL_MULTILABEL,
        LABEL_SOURCE,
        LABEL_AGREEMENT,
        CANDIDATE_LABEL,
        MANUAL_LABEL,
        SPLIT,
        FOLD,
        FIRST_TWEET_AT,
        LAST_TWEET_AT,
    }
)
