"""Testes da etapa 'embed', com foco na resolução do usuário-dono do cache.

Reproduz o bug em que o índice de embeddings ficava indexado pelo nome do
arquivo de origem (handle bruto da coleta) em vez do ``user_id``
pseudonimizado presente na coluna dos tweets — divergência que deixava as
sequências inacessíveis para os modelos recorrentes (BiLSTM) e para o grupo
de atributos 'semantic'.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pipelines.embedding import EmbeddingStage


@pytest.fixture
def partition_dir(tmp_path: Path) -> Path:
    """Diretório particionado por usuário com nome de arquivo != user_id."""
    directory = tmp_path / "tweets_clean"
    directory.mkdir()

    frame = pl.DataFrame(
        {
            "user_id": ["u_423dd5375f74d5d1", "u_423dd5375f74d5d1"],
            "text_normalized": ["texto um", "texto dois"],
        }
    )
    frame.write_parquet(directory / "000c7X_fulljson.parquet")
    return directory


def test_resolve_owner_id_reads_real_pseudonym_from_partition_content(
    tmp_path: Path, partition_dir: Path
) -> None:
    """A resolução deve devolver o pseudônimo real, não o nome do arquivo."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    stage = EmbeddingStage()

    owner_id = stage._resolve_owner_id(cache_dir, "000c7X_fulljson", partition_dir)

    assert owner_id == "u_423dd5375f74d5d1"
    assert (cache_dir / "000c7X_fulljson.owner").read_text(encoding="utf-8") == owner_id


def test_resolve_owner_id_uses_cached_sidecar_without_rereading_partition(
    tmp_path: Path, partition_dir: Path
) -> None:
    """Uma segunda resolução deve usar o arquivo ``.owner``, não reler o parquet."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "000c7X_fulljson.owner").write_text("u_cached_value", encoding="utf-8")
    stage = EmbeddingStage()

    # Diretório de origem inexistente: se caísse no fallback, levantaria erro
    # ou devolveria o nome do arquivo, não o valor do sidecar.
    owner_id = stage._resolve_owner_id(cache_dir, "000c7X_fulljson", tmp_path / "nao_existe")

    assert owner_id == "u_cached_value"


def test_load_cached_embeddings_indexes_by_pseudonym_not_filename(
    tmp_path: Path, partition_dir: Path
) -> None:
    """O índice consolidado deve usar o pseudônimo real como dono de cada vetor."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.save(cache_dir / "000c7X_fulljson.npy", np.ones((2, 4), dtype=np.float32))
    stage = EmbeddingStage()

    embeddings, owners = stage._load_cached_embeddings(
        cache_dir, ["000c7X_fulljson"], partition_dir
    )

    assert embeddings.shape == (2, 4)
    assert owners == ["u_423dd5375f74d5d1", "u_423dd5375f74d5d1"]


def test_encode_and_cache_user_writes_owner_sidecar(tmp_path: Path, partition_dir: Path) -> None:
    """A codificação de um usuário pendente já grava o pseudônimo real ao lado do cache."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    stage = EmbeddingStage()

    class _FakeEncoder:
        def encode(self, texts: list[str]) -> np.ndarray:
            return np.zeros((len(texts), 4), dtype=np.float32)

    ok = stage._encode_and_cache_user(
        "000c7X_fulljson",
        partition_dir,
        _FakeEncoder(),
        "modelo",
        cache_dir,  # pyright: ignore[reportArgumentType]
    )

    assert ok is True
    assert (cache_dir / "000c7X_fulljson.owner").read_text(encoding="utf-8") == "u_423dd5375f74d5d1"
