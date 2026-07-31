"""Contratos de dados no nível do usuário — a unidade amostral do projeto."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import Series

from schemas.tweets import PSEUDONYM_REGEX

#: Classes válidas da variável-alvo (ver `constants.labels.UserLabel`).
VALID_LABELS: list[str] = ["controle", "depressao", "ideacao_suicida"]

#: Inclui `indefinido`, permitido apenas antes do descarte por baixo consenso.
VALID_LABELS_WITH_UNDECIDED: list[str] = [*VALID_LABELS, "indefinido"]


class UserMetadataSchema(pa.DataFrameModel):
    """Contrato dos metadados públicos do usuário.

    Nenhum campo identificável direto (handle, nome de exibição, biografia,
    URL de foto) aparece aqui: são descartados na ingestão por minimização de
    dados. Sem eles, o dataset continua respondendo à pergunta de pesquisa.
    """

    user_id: Series[str] = pa.Field(nullable=False, unique=True, str_matches=PSEUDONYM_REGEX)
    followers_count: Series[int] = pa.Field(ge=0, nullable=True)
    following_count: Series[int] = pa.Field(ge=0, nullable=True)
    statuses_count: Series[int] = pa.Field(ge=0, nullable=True)
    account_created_at: Series[pl.Datetime] = pa.Field(nullable=True)
    is_verified: Series[bool] = pa.Field(nullable=True)

    # Padrão documentado do pandera: a classe `Config` aninhada não herda de
    # `BaseConfig`, o que o type checker interpreta como um override
    # incompatível — falso positivo conhecido dos stubs do pandera (repete
    # nos demais schemas deste módulo).
    class Config:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Configuração do contrato."""

        strict = True
        coerce = True


class UserLabelSchema(pa.DataFrameModel):
    """Contrato do rótulo do usuário produzido pela supervisão fraca.

    ``label_agreement`` guarda o grau de concordância entre as fontes de
    rotulação. Preservá-lo é o que permite, na análise, separar erro do
    modelo de ruído do rótulo — sem essa coluna, os dois efeitos ficariam
    confundidos.
    """

    user_id: Series[str] = pa.Field(nullable=False, unique=True, str_matches=PSEUDONYM_REGEX)
    user_label: Series[str] = pa.Field(nullable=False, isin=VALID_LABELS_WITH_UNDECIDED)
    user_label_multilabel: Series[str] = pa.Field(nullable=True)
    candidate_label: Series[str] = pa.Field(nullable=True, isin=VALID_LABELS)
    label_source: Series[str] = pa.Field(nullable=False)
    label_agreement: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=False)
    manual_label: Series[str] = pa.Field(nullable=True, isin=VALID_LABELS)

    class Config:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Configuração do contrato."""

        strict = True
        coerce = True


class UserProfileSchema(pa.DataFrameModel):
    """Contrato do perfil temporal agregado do usuário.

    Colunas descritivas (não preditoras) usadas na avaliação por fatias e no
    controle de qualidade da coleta.
    """

    user_id: Series[str] = pa.Field(nullable=False, unique=True, str_matches=PSEUDONYM_REGEX)
    n_tweets: Series[int] = pa.Field(gt=0, nullable=False)
    active_days: Series[int] = pa.Field(gt=0, nullable=False)
    span_days: Series[int] = pa.Field(ge=0, nullable=False)
    first_tweet_at: Series[pl.Datetime] = pa.Field(nullable=False)
    last_tweet_at: Series[pl.Datetime] = pa.Field(nullable=False)
    night_activity_ratio: Series[float] = pa.Field(ge=0.0, le=1.0, nullable=False)

    class Config:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Configuração do contrato."""

        strict = True
        coerce = True


class SplitSchema(pa.DataFrameModel):
    """Contrato da atribuição de partição por usuário.

    A partição é atribuída ao **usuário**, nunca ao tweet: é o que impede o
    vazamento de estilo entre treino e teste.
    """

    user_id: Series[str] = pa.Field(nullable=False, unique=True, str_matches=PSEUDONYM_REGEX)
    user_label: Series[str] = pa.Field(nullable=False, isin=VALID_LABELS)
    split: Series[str] = pa.Field(nullable=False, isin=["train", "val", "test"])
    fold: Series[int] = pa.Field(ge=-1, nullable=False)

    class Config:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Configuração do contrato."""

        strict = True
        coerce = True
