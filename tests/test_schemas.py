"""Testes dos contratos de dados (pandera) aplicados entre estágios."""

from __future__ import annotations

import polars as pl
import pytest

from exceptions.data import SchemaValidationError
from schemas.features import list_feature_columns, validate_feature_matrix
from schemas.tweets import CleanTweetSchema, LabeledTweetSchema, RawTweetSchema
from schemas.users import SplitSchema, UserLabelSchema
from schemas.validation import is_valid, validate_frame


class TestContratoDeTweets:
    """Testes dos contratos ao longo do processamento dos tweets."""

    def test_tweets_brutos_validos(self, raw_tweets: pl.DataFrame) -> None:
        """Os dados sintéticos satisfazem o contrato de entrada."""
        assert validate_frame(raw_tweets, RawTweetSchema).height == raw_tweets.height

    def test_tweets_limpos_validos(self, clean_tweets: pl.DataFrame) -> None:
        """O contrato de saída do preprocess é satisfeito."""
        assert is_valid(clean_tweets, CleanTweetSchema)

    def test_tweets_rotulados_validos(self, labeled_tweets: pl.DataFrame) -> None:
        """O contrato de saída da rotulação é satisfeito."""
        assert is_valid(labeled_tweets, LabeledTweetSchema)

    def test_identificador_nao_pseudonimizado_e_rejeitado(self, raw_tweets: pl.DataFrame) -> None:
        """Um handle real na coluna de identificador é violação de privacidade."""
        invalido = raw_tweets.with_columns(pl.lit("fulano_da_silva").alias("user_id"))

        with pytest.raises(SchemaValidationError):
            validate_frame(invalido, RawTweetSchema)

    def test_coluna_extra_e_rejeitada(self, raw_tweets: pl.DataFrame) -> None:
        """Coluna inesperada quase sempre indica união equivocada de fontes."""
        invalido = raw_tweets.with_columns(pl.lit(1).alias("coluna_inesperada"))

        with pytest.raises(SchemaValidationError):
            validate_frame(invalido, RawTweetSchema)

    def test_engajamento_negativo_e_rejeitado(self, raw_tweets: pl.DataFrame) -> None:
        """Contagens negativas indicam corrupção na ingestão."""
        invalido = raw_tweets.with_columns(pl.lit(-5).alias("like_count"))

        with pytest.raises(SchemaValidationError):
            validate_frame(invalido, RawTweetSchema)

    def test_sentimento_fora_do_vocabulario_e_rejeitado(self, labeled_tweets: pl.DataFrame) -> None:
        """Um rótulo fora do vocabulário indica falha no mapeamento do encoder."""
        invalido = labeled_tweets.with_columns(pl.lit("muito_negativo").alias("sentiment"))

        with pytest.raises(SchemaValidationError):
            validate_frame(invalido, LabeledTweetSchema)

    def test_score_fora_da_faixa_e_rejeitado(self, labeled_tweets: pl.DataFrame) -> None:
        """Confiança fora de [0, 1] não é uma probabilidade."""
        invalido = labeled_tweets.with_columns(pl.lit(1.5).alias("sentiment_score"))

        with pytest.raises(SchemaValidationError):
            validate_frame(invalido, LabeledTweetSchema)

    def test_tweet_id_duplicado_e_rejeitado(self, raw_tweets: pl.DataFrame) -> None:
        """Identificador duplicado indica falha na deduplicação."""
        invalido = pl.concat([raw_tweets, raw_tweets.head(1)])

        with pytest.raises(SchemaValidationError):
            validate_frame(invalido, RawTweetSchema)

    def test_mensagem_de_erro_nomeia_o_contexto(self, raw_tweets: pl.DataFrame) -> None:
        """A mensagem indica em qual fronteira o contrato foi violado."""
        invalido = raw_tweets.with_columns(pl.lit(-1).alias("like_count"))

        with pytest.raises(SchemaValidationError, match="etapa de teste"):
            validate_frame(invalido, RawTweetSchema, context="etapa de teste")


class TestContratoDeUsuarios:
    """Testes dos contratos no nível do usuário."""

    def test_rotulos_validos(self, user_labels: pl.DataFrame) -> None:
        """Os rótulos sintéticos satisfazem o contrato."""
        assert is_valid(user_labels, UserLabelSchema)

    def test_classe_desconhecida_e_rejeitada(self, user_labels: pl.DataFrame) -> None:
        """Uma classe fora do vocabulário quebra o mapeamento dos modelos."""
        invalido = user_labels.with_columns(pl.lit("classe_inventada").alias("user_label"))

        with pytest.raises(SchemaValidationError):
            validate_frame(invalido, UserLabelSchema)

    def test_usuario_duplicado_e_rejeitado(self, user_labels: pl.DataFrame) -> None:
        """Um usuário com dois rótulos indica falha no consenso."""
        invalido = pl.concat([user_labels, user_labels.head(1)])

        with pytest.raises(SchemaValidationError):
            validate_frame(invalido, UserLabelSchema)

    def test_particoes_validas(self, splits: pl.DataFrame) -> None:
        """A tabela de partições satisfaz o contrato."""
        assert is_valid(splits, SplitSchema)

    def test_particao_desconhecida_e_rejeitada(self, splits: pl.DataFrame) -> None:
        """Só treino, validação e teste são partições válidas."""
        invalido = splits.with_columns(pl.lit("holdout").alias("split"))

        with pytest.raises(SchemaValidationError):
            validate_frame(invalido, SplitSchema)


class TestMatrizDeAtributos:
    """Testes da validação estrutural da matriz de atributos."""

    def test_matriz_valida(self, feature_matrix: pl.DataFrame) -> None:
        """A matriz sintética passa na validação estrutural."""
        assert validate_feature_matrix(feature_matrix).height == feature_matrix.height

    def test_grupos_esperados_sao_exigidos(self, feature_matrix: pl.DataFrame) -> None:
        """Um grupo declarado como obrigatório precisa ter colunas."""
        sem_semanticos = feature_matrix.drop(
            [name for name in feature_matrix.columns if name.startswith("sem_")]
        )
        with pytest.raises(SchemaValidationError, match="semantic"):
            validate_feature_matrix(sem_semanticos, expected_groups=["semantic"])

    def test_valores_ausentes_sao_rejeitados(self, feature_matrix: pl.DataFrame) -> None:
        """NaN que chega ao modelo é tratado de formas diferentes por cada família."""
        com_nulo = feature_matrix.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("emo_polarity_mean"))
            .alias("emo_polarity_mean")
        )
        with pytest.raises(SchemaValidationError, match="ausentes"):
            validate_feature_matrix(com_nulo)

    def test_matriz_sem_identificador_e_rejeitada(self, feature_matrix: pl.DataFrame) -> None:
        """Sem `user_id`, não há como associar atributos a rótulos."""
        with pytest.raises(SchemaValidationError, match="user_id"):
            validate_feature_matrix(feature_matrix.drop("user_id"))

    def test_matriz_sem_atributos_e_rejeitada(self) -> None:
        """Uma matriz só com chaves não serve para treinar nada."""
        frame = pl.DataFrame({"user_id": ["u_a"], "user_label": ["controle"]})

        with pytest.raises(SchemaValidationError, match="nenhuma coluna de atributos"):
            validate_feature_matrix(frame)

    def test_selecao_de_colunas_e_deterministica(self, feature_matrix: pl.DataFrame) -> None:
        """A ordem das colunas é estável entre chamadas."""
        assert list_feature_columns(feature_matrix) == list_feature_columns(feature_matrix)

    def test_selecao_ignora_colunas_nao_preditoras(self, feature_matrix: pl.DataFrame) -> None:
        """Identificador e rótulo nunca entram como atributo."""
        features = list_feature_columns(feature_matrix)

        assert "user_id" not in features
        assert "user_label" not in features
