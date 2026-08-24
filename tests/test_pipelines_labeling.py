"""Testes do agrupamento de usuários por lote na etapa 'label'.

Reproduz a otimização que substitui a inferência usuário a usuário pelo
acúmulo de tweets de vários usuários até atingir o ``batch_size`` real dos
encoders Transformer de sentimento/emoção (:mod:`labeling.sentiment`,
:mod:`labeling.emotion`) — sem essa mudança, um usuário com poucos tweets
levava o ``pipeline()`` a rodar lotes bem menores que o configurado.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from pipelines.labeling import LabelingStage

#: Dtypes explícitos, para que até uma partição com 0 linhas fique conforme
#: `schemas.tweets.CleanTweetSchema` (a inferência de tipo do polars vira
#: `Null` para colunas totalmente vazias sem essa declaração).
_SCHEMA_OVERRIDES: dict[str, Any] = {
    "user_id": pl.Utf8,
    "tweet_id": pl.Utf8,
    "text": pl.Utf8,
    "text_normalized": pl.Utf8,
    "text_clean": pl.Utf8,
    "created_at": pl.Datetime("us"),
    "language": pl.Utf8,
    "is_reply": pl.Boolean,
    "is_retweet": pl.Boolean,
    "like_count": pl.Int64,
    "reply_count": pl.Int64,
    "retweet_count": pl.Int64,
    "quote_count": pl.Int64,
    "source_query": pl.Utf8,
    "source_group": pl.Utf8,
}


@pytest.fixture
def stage() -> LabelingStage:
    """Instância da etapa de rotulação, sem contexto de execução."""
    return LabelingStage()


def _write_user(directory: Path, user_id: str, n_tweets: int) -> pl.DataFrame:
    """Grava a partição de um usuário com ``n_tweets`` linhas conformes a ``CleanTweetSchema``."""
    origin = datetime(2024, 1, 1)
    texts = [f"texto numero {i} do usuario {user_id}" for i in range(n_tweets)]
    frame = pl.DataFrame(
        {
            "user_id": [user_id] * n_tweets,
            "tweet_id": [f"{user_id}_t{i}" for i in range(n_tweets)],
            "text": texts,
            "text_normalized": texts,
            "text_clean": [text.lower() for text in texts],
            "created_at": [origin + timedelta(minutes=i) for i in range(n_tweets)],
            "language": ["pt"] * n_tweets,
            "is_reply": [False] * n_tweets,
            "is_retweet": [False] * n_tweets,
            "like_count": [0] * n_tweets,
            "reply_count": [0] * n_tweets,
            "retweet_count": [0] * n_tweets,
            "quote_count": [0] * n_tweets,
            "source_query": ["controle"] * n_tweets,
            "source_group": ["controle"] * n_tweets,
        },
        schema_overrides=_SCHEMA_OVERRIDES,
    )
    frame.write_parquet(directory / f"{user_id}.parquet")
    return frame


class _FakeSentimentLabeler:
    """Rotulador de sentimento falso: marca cada linha com um score distinto.

    O score é derivado do comprimento do texto, para que testes possam
    verificar se, após dividir um lote combinado de volta por usuário, cada
    linha manteve o valor correspondente ao seu próprio texto (ou seja, o
    corte não embaralhou/deslocou linhas entre usuários).
    """

    def label_frame(
        self, frame: pl.DataFrame, text_column: str = "text_normalized"
    ) -> pl.DataFrame:
        score = pl.col(text_column).str.len_chars().cast(pl.Float64) / 1000.0
        return frame.with_columns(
            pl.lit("neutro").alias("sentiment"),
            score.alias("sentiment_score"),
            pl.lit(0.0).alias("sentiment_polarity"),
        )


class TestResolveInferenceBatchSize:
    """Testes do tamanho de acúmulo derivado da configuração."""

    def _config(self, *, sentiment_enabled: bool, emotion_enabled: bool) -> SimpleNamespace:
        return SimpleNamespace(
            labeling=SimpleNamespace(
                sentiment=SimpleNamespace(enabled=sentiment_enabled, batch_size=16),
                emotion=SimpleNamespace(enabled=emotion_enabled, batch_size=64),
            )
        )

    def test_usa_o_maior_batch_size_entre_os_habilitados(self, stage: LabelingStage) -> None:
        """Com os dois encoders ativos, acumula até o maior batch_size."""
        config = self._config(sentiment_enabled=True, emotion_enabled=True)
        assert stage._resolve_inference_batch_size(config) == 64

    def test_ignora_batch_size_de_encoder_desativado(self, stage: LabelingStage) -> None:
        """Só o batch_size do encoder habilitado conta."""
        config = self._config(sentiment_enabled=True, emotion_enabled=False)
        assert stage._resolve_inference_batch_size(config) == 16

    def test_sem_nenhum_encoder_habilitado_preserva_particionamento_por_usuario(
        self, stage: LabelingStage
    ) -> None:
        """Sem inferência nenhuma, o agrupamento não deve alterar o comportamento."""
        config = self._config(sentiment_enabled=False, emotion_enabled=False)
        assert stage._resolve_inference_batch_size(config) == 1


class TestIterUserBatches:
    """Testes do agrupamento de usuários pendentes em lotes de inferência."""

    def test_acumula_varios_usuarios_pequenos_ate_o_batch_size(
        self, stage: LabelingStage, tmp_path: Path
    ) -> None:
        """Usuários com poucos tweets são combinados até fechar o lote."""
        clean_dir = tmp_path / "tweets_clean"
        clean_dir.mkdir()
        users = ["u_00000001", "u_00000002", "u_00000003"]
        for user_id in users:
            _write_user(clean_dir, user_id, 2)

        batches = list(stage._iter_user_batches(users, clean_dir, batch_size=4))

        assert [[user_id for user_id, _ in batch] for batch in batches] == [
            ["u_00000001", "u_00000002"],
            ["u_00000003"],
        ]

    def test_usuario_com_tweets_suficientes_forma_lote_sozinho(
        self, stage: LabelingStage, tmp_path: Path
    ) -> None:
        """Um usuário que já atinge o batch_size sozinho não espera por outros."""
        clean_dir = tmp_path / "tweets_clean"
        clean_dir.mkdir()
        _write_user(clean_dir, "u_0000000a", 10)
        _write_user(clean_dir, "u_0000000b", 1)

        batches = list(
            stage._iter_user_batches(["u_0000000a", "u_0000000b"], clean_dir, batch_size=4)
        )

        assert [[user_id for user_id, _ in batch] for batch in batches] == [
            ["u_0000000a"],
            ["u_0000000b"],
        ]

    def test_ultimo_lote_parcial_ainda_e_produzido(
        self, stage: LabelingStage, tmp_path: Path
    ) -> None:
        """Um resto abaixo do batch_size no fim da lista ainda vira um lote."""
        clean_dir = tmp_path / "tweets_clean"
        clean_dir.mkdir()
        _write_user(clean_dir, "u_00000001", 1)

        batches = list(stage._iter_user_batches(["u_00000001"], clean_dir, batch_size=32))

        assert len(batches) == 1
        assert batches[0][0][0] == "u_00000001"


class TestLabelAndWriteBatch:
    """Testes da rotulação combinada e da escrita de volta por usuário."""

    def test_grava_um_arquivo_por_usuario_com_as_linhas_corretas(
        self, stage: LabelingStage, tmp_path: Path
    ) -> None:
        """Após o corte do lote combinado, cada usuário recebe só as suas linhas."""
        clean_dir = tmp_path / "tweets_clean"
        clean_dir.mkdir()
        labeled_dir = tmp_path / "tweets_labeled"
        labeled_dir.mkdir()

        frame_a = _write_user(clean_dir, "u_0000000a", 2)
        frame_b = _write_user(clean_dir, "u_0000000b", 3)
        batch = [("u_0000000a", frame_a), ("u_0000000b", frame_b)]

        stage._label_and_write_batch(
            batch,
            _FakeSentimentLabeler(),  # pyright: ignore[reportArgumentType]
            None,
            labeled_dir,
        )

        result_a = pl.read_parquet(labeled_dir / "u_0000000a.parquet")
        result_b = pl.read_parquet(labeled_dir / "u_0000000b.parquet")

        assert result_a.height == 2
        assert result_b.height == 3
        assert result_a["text_normalized"].to_list() == frame_a["text_normalized"].to_list()
        assert result_b["text_normalized"].to_list() == frame_b["text_normalized"].to_list()

        # O score é derivado do texto: se o corte do lote combinado tivesse
        # deslocado linhas entre usuários, o score gravado não bateria mais
        # com o comprimento do texto daquele usuário.
        expected_a = [len(text) / 1000.0 for text in frame_a["text_normalized"].to_list()]
        expected_b = [len(text) / 1000.0 for text in frame_b["text_normalized"].to_list()]
        assert result_a["sentiment_score"].to_list() == pytest.approx(expected_a)
        assert result_b["sentiment_score"].to_list() == pytest.approx(expected_b)

    def test_usuario_sem_tweets_pendentes_ainda_e_gravado_vazio(
        self, stage: LabelingStage, tmp_path: Path
    ) -> None:
        """Um usuário sem linhas de entrada continua marcado como processado."""
        clean_dir = tmp_path / "tweets_clean"
        clean_dir.mkdir()
        labeled_dir = tmp_path / "tweets_labeled"
        labeled_dir.mkdir()

        empty_frame = _write_user(clean_dir, "u_00000000", 0)
        batch = [("u_00000000", empty_frame)]

        stage._label_and_write_batch(
            batch,
            _FakeSentimentLabeler(),  # pyright: ignore[reportArgumentType]
            None,
            labeled_dir,
        )

        assert (labeled_dir / "u_00000000.parquet").is_file()
        assert pl.read_parquet(labeled_dir / "u_00000000.parquet").is_empty()

    def test_lote_misto_preserva_vazios_e_rotula_apenas_os_nao_vazios(
        self, stage: LabelingStage, tmp_path: Path
    ) -> None:
        """Um lote com usuários vazios e não vazios grava ambos corretamente."""
        clean_dir = tmp_path / "tweets_clean"
        clean_dir.mkdir()
        labeled_dir = tmp_path / "tweets_labeled"
        labeled_dir.mkdir()

        empty_frame = _write_user(clean_dir, "u_00000000", 0)
        frame_c = _write_user(clean_dir, "u_0000000c", 2)
        batch = [("u_00000000", empty_frame), ("u_0000000c", frame_c)]

        stage._label_and_write_batch(
            batch,
            _FakeSentimentLabeler(),  # pyright: ignore[reportArgumentType]
            None,
            labeled_dir,
        )

        assert pl.read_parquet(labeled_dir / "u_00000000.parquet").is_empty()
        assert pl.read_parquet(labeled_dir / "u_0000000c.parquet").height == 2
