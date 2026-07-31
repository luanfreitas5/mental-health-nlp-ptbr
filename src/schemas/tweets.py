"""Contratos de dados dos tweets, um por estágio do pipeline.

A progressão ``RawTweetSchema -> CleanTweetSchema -> LabeledTweetSchema``
espelha ``data/raw -> data/interim``: cada contrato herda o anterior e
acrescenta apenas as colunas que sua etapa produz. Herdar em vez de repetir
garante que uma mudança no contrato de origem se propague automaticamente.

Todos usam ``strict = True``: uma coluna inesperada quase sempre indica que
duas fontes de dados foram unidas por engano, e é melhor descobrir isso na
fronteira do que num modelo treinado com uma feature vazada.
"""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import Series

#: Prefixo obrigatório dos identificadores pseudonimizados (ver `utils.hashing`).
PSEUDONYM_REGEX: str = r"^u_[0-9a-f]{8,64}$"


class RawTweetSchema(pa.DataFrameModel):
    """Contrato do tweet recém-coletado (``data/raw``).

    O ``user_id`` já chega pseudonimizado: a conversão acontece dentro do
    coletor, antes de qualquer gravação em disco, de modo que nenhum
    identificador direto chega a ser persistido.
    """

    user_id: Series[str] = pa.Field(nullable=False, str_matches=PSEUDONYM_REGEX)
    tweet_id: Series[str] = pa.Field(nullable=False, unique=True)
    text: Series[str] = pa.Field(nullable=False)
    created_at: Series[pl.Datetime] = pa.Field(nullable=False)
    language: Series[str] = pa.Field(nullable=True)
    is_reply: Series[bool] = pa.Field(nullable=False)
    is_retweet: Series[bool] = pa.Field(nullable=False)
    like_count: Series[int] = pa.Field(ge=0, nullable=False)
    reply_count: Series[int] = pa.Field(ge=0, nullable=False)
    retweet_count: Series[int] = pa.Field(ge=0, nullable=False)
    quote_count: Series[int] = pa.Field(ge=0, nullable=False)
    source_query: Series[str] = pa.Field(nullable=True)
    source_group: Series[str] = pa.Field(nullable=True)

    # Padrão documentado do pandera: a classe `Config` aninhada não herda de
    # `BaseConfig`, o que o type checker interpreta como um override
    # incompatível — falso positivo conhecido dos stubs do pandera (repete
    # nos demais schemas deste módulo).
    class Config:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Configuração do contrato."""

        strict = True
        coerce = True


class CleanTweetSchema(RawTweetSchema):
    """Contrato do tweet após limpeza e normalização (``data/interim``).

    Mantém dois textos de propósito: ``text_normalized`` preserva pontuação,
    caixa e emoji (entrada dos Transformers e do LLM), enquanto ``text_clean``
    é agressivamente reduzido (entrada de TF-IDF, n-grams e léxicos).
    """

    text_normalized: Series[str] = pa.Field(nullable=False)
    text_clean: Series[str] = pa.Field(nullable=False)

    class Config:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Configuração do contrato."""

        strict = True
        coerce = True


class LabeledTweetSchema(CleanTweetSchema):
    """Contrato do tweet após a rotulação automática de sentimento."""

    sentiment: Series[str] = pa.Field(
        nullable=False,
        isin=["positivo", "negativo", "neutro", "indefinido"],
    )
    sentiment_score: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=False)
    sentiment_polarity: Series[float] = pa.Field(ge=-1.0, le=1.0, nullable=False)

    class Config:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Configuração do contrato."""

        strict = True
        coerce = True


class PsychologicalScoreSchema(pa.DataFrameModel):
    """Contrato do vetor psicológico extraído por LLM.

    Uma linha por lote de tweets de um usuário (``features.psychological.
    granularity = batch``). Todas as dimensões vivem em ``[0, 1]``: valores
    fora da faixa indicam que o LLM devolveu algo fora do schema e que o
    reparo da resposta falhou silenciosamente.
    """

    user_id: Series[str] = pa.Field(nullable=False, str_matches=PSEUDONYM_REGEX)
    batch_index: Series[int] = pa.Field(ge=0, nullable=False)
    n_tweets: Series[int] = pa.Field(gt=0, nullable=False)
    window_start: Series[pl.Datetime] = pa.Field(nullable=False)
    window_end: Series[pl.Datetime] = pa.Field(nullable=False)
    tristeza: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=False)
    isolamento: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=False)
    esperanca: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=False)
    ansiedade: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=False)
    risco_suicida: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=False)
    model: Series[str] = pa.Field(nullable=False)
    prompt_version: Series[str] = pa.Field(nullable=False)

    class Config:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Configuração do contrato."""

        strict = True
        coerce = True
