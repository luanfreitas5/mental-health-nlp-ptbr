"""Testes de particionamento, IO de dados e construção de consultas."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from config.settings import CrossValidationSection, SplitSection
from data.catalog import build_catalog, compare_manifest, write_dataset_manifest
from data.queries import build_query_string, summarize_queries
from data.reader import (
    count_users,
    list_collected_users,
    read_parquet,
    read_partitioned,
    read_user_histories,
)
from data.splitter import assign_folds, build_split_table, create_splits, filter_split
from data.writer import append_parquet, write_parquet, write_partitioned
from exceptions.data import DatasetNotFoundError, InsufficientDataError


class TestParticionamento:
    """Testes do particionamento agrupado por usuário."""

    def test_cria_tres_particoes(self, feature_matrix: pl.DataFrame) -> None:
        """Treino, validação e teste são criados."""
        result = create_splits(
            feature_matrix.select(["user_id", "user_label"]),
            SplitSection(test_size=0.2, val_size=0.1),
            42,
        )
        assert set(result["split"].unique().to_list()) == {"train", "val", "test"}

    def test_sem_vazamento_entre_particoes(self, splits: pl.DataFrame) -> None:
        """Nenhum usuário aparece em duas partições — a garantia central."""
        by_split = {
            name: set(splits.filter(pl.col("split") == name)["user_id"].to_list())
            for name in ("train", "val", "test")
        }

        assert not by_split["train"] & by_split["test"]
        assert not by_split["train"] & by_split["val"]
        assert not by_split["val"] & by_split["test"]

    def test_todos_os_usuarios_particionados(
        self, feature_matrix: pl.DataFrame, splits: pl.DataFrame
    ) -> None:
        """Nenhum usuário se perde no particionamento."""
        assert splits.height == feature_matrix.height

    def test_estratificacao_preserva_classes(self, splits: pl.DataFrame) -> None:
        """Todas as classes aparecem no treino e no teste."""
        treino = splits.filter(pl.col("split") == "train")["user_label"].n_unique()
        teste = splits.filter(pl.col("split") == "test")["user_label"].n_unique()

        assert treino == teste

    def test_particionamento_e_reprodutivel(self, feature_matrix: pl.DataFrame) -> None:
        """A mesma semente produz exatamente a mesma divisão."""
        users = feature_matrix.select(["user_id", "user_label"])
        config = SplitSection(test_size=0.2, val_size=0.1)

        assert create_splits(users, config, 42).equals(create_splits(users, config, 42))

    def test_dataset_vazio_e_rejeitado(self) -> None:
        """Não há como particionar um conjunto vazio."""
        empty = pl.DataFrame(
            {"user_id": [], "user_label": []}, schema={"user_id": pl.Utf8, "user_label": pl.Utf8}
        )

        with pytest.raises(InsufficientDataError):
            create_splits(empty, SplitSection(), 42)

    def test_classe_minuscula_e_rejeitada(self) -> None:
        """Menos de três usuários numa classe impede estratificar em 3 partições."""
        users = pl.DataFrame({"user_id": ["u_a", "u_b"], "user_label": ["controle", "depressao"]})
        with pytest.raises(InsufficientDataError, match="estratificar"):
            create_splits(users, SplitSection(), 42)

    def test_teste_recebe_fold_negativo(self, splits: pl.DataFrame) -> None:
        """O teste nunca participa da validação cruzada."""
        teste = splits.filter(pl.col("split") == "test")
        assert (teste["fold"] == -1).all()

    def test_folds_cobrem_desenvolvimento(self, splits: pl.DataFrame) -> None:
        """Todo usuário de desenvolvimento recebe um fold válido."""
        desenvolvimento = splits.filter(pl.col("split") != "test")
        assert (desenvolvimento["fold"] >= 0).all()

    def test_folds_insuficientes_sao_rejeitados(self, feature_matrix: pl.DataFrame) -> None:
        """Mais folds que usuários por classe é erro de configuração."""
        splits = create_splits(feature_matrix.select(["user_id", "user_label"]), SplitSection(), 42)
        with pytest.raises(InsufficientDataError, match="folds"):
            assign_folds(splits, CrossValidationSection(n_splits=100), 42)

    def test_filter_split(self, feature_matrix: pl.DataFrame, splits: pl.DataFrame) -> None:
        """A seleção por partição devolve apenas os usuários correspondentes."""
        treino = filter_split(feature_matrix, splits, "train")
        esperado = splits.filter(pl.col("split") == "train").height

        assert treino.height == esperado

    def test_build_split_table_produz_ambas_as_colunas(self, feature_matrix: pl.DataFrame) -> None:
        """A tabela final traz partição e fold."""
        result = build_split_table(
            feature_matrix.select(["user_id", "user_label"]),
            SplitSection(),
            CrossValidationSection(n_splits=3),
            42,
        )
        assert {"split", "fold"}.issubset(set(result.columns))


class TestConsultas:
    """Testes da construção das consultas de coleta."""

    def test_expressao_exata_para_termo_composto(self) -> None:
        """Sem aspas, 'quero morrer' viraria busca por 'quero' OU 'morrer'."""
        query = build_query_string(
            "quero morrer",
            language="pt",
            since=date(2024, 1, 1),
            until=date(2024, 2, 1),
            exclude_retweets=True,
            exclude_replies=False,
        )
        assert '"quero morrer"' in query
        assert "-filter:retweets" in query

    def test_hashtag_nao_recebe_aspas(self) -> None:
        """Hashtags são termos únicos e não precisam de aspas."""
        query = build_query_string(
            "#depressao",
            language="pt",
            since=date(2024, 1, 1),
            until=date(2024, 2, 1),
            exclude_retweets=False,
            exclude_replies=False,
        )
        assert '"' not in query

    def test_filtros_opcionais(self) -> None:
        """Os filtros de tipo de publicação são opcionais."""
        query = build_query_string(
            "tristeza",
            language="pt",
            since=date(2024, 1, 1),
            until=date(2024, 2, 1),
            exclude_retweets=False,
            exclude_replies=True,
        )
        assert "-filter:retweets" not in query
        assert "-filter:replies" in query

    def test_resumo_de_consultas(self) -> None:
        """O resumo conta as consultas por grupo e tipo."""
        from data.queries import SearchQuery

        queries = [
            SearchQuery("q1", "t1", "keyword", "depressao", "depressao"),
            SearchQuery("q2", "t2", "hashtag", "depressao", "depressao"),
        ]
        summary = summarize_queries(queries)

        assert summary["depressao"]["total"] == 2
        assert summary["depressao"]["keyword"] == 1

    def test_consultas_reais_sao_construidas(self, config) -> None:
        """Os arquivos .txt versionados produzem consultas válidas."""
        from data.queries import build_queries

        queries = build_queries(config.collection.seed_search)

        assert len(queries) > 10
        assert {query.group for query in queries} == set(config.collection.seed_search.groups)


class TestEscritaELeitura:
    """Testes da persistência em Parquet."""

    def test_ida_e_volta(self, tmp_path: Path) -> None:
        """O conteúdo gravado é lido de volta sem alteração."""
        frame = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        target = write_parquet(frame, tmp_path / "dados.parquet")

        assert read_parquet(target).equals(frame)

    def test_escrita_e_atomica(self, tmp_path: Path) -> None:
        """Nenhum arquivo temporário sobrevive à gravação."""
        write_parquet(pl.DataFrame({"a": [1]}), tmp_path / "dados.parquet")
        assert not list(tmp_path.glob("*.tmp"))

    def test_leitura_de_colunas_especificas(self, tmp_path: Path) -> None:
        """Ler um subconjunto de colunas reduz memória em matrizes largas."""
        frame = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        target = write_parquet(frame, tmp_path / "dados.parquet")

        assert read_parquet(target, columns=["a"]).columns == ["a"]

    def test_arquivo_ausente_indica_a_etapa(self, tmp_path: Path) -> None:
        """A mensagem de erro aponta qual etapa deveria ter produzido o artefato."""
        with pytest.raises(DatasetNotFoundError, match="preprocess"):
            read_parquet(tmp_path / "tweets_clean.parquet")

    def test_particionamento_por_usuario(self, tmp_path: Path) -> None:
        """Um arquivo por usuário é o que torna a coleta retomável."""
        frame = pl.DataFrame({"user_id": ["u_a", "u_b", "u_a"], "x": [1, 2, 3]})
        written = write_partitioned(frame, tmp_path, "user_id")

        assert len(written) == 2
        assert count_users(tmp_path) == 2
        assert list_collected_users(tmp_path) == {"u_a", "u_b"}

    def test_particionamento_exige_a_coluna(self, tmp_path: Path) -> None:
        """Coluna de partição ausente falha com erro claro."""
        with pytest.raises(KeyError, match="ausente"):
            write_partitioned(pl.DataFrame({"x": [1]}), tmp_path, "user_id")

    def test_particionamento_com_clear_remove_arquivos_antigos(self, tmp_path: Path) -> None:
        """``clear=True`` descarta usuários da execução anterior que sumiram nesta.

        Sem isso, um usuário removido no reprocessamento (ex.: filtrado por uma
        nova regra) deixaria seu arquivo antigo para trás, e a etapa seguinte o
        leria como se ainda fizesse parte do dataset atual.
        """
        write_partitioned(pl.DataFrame({"user_id": ["u_antigo"], "x": [1]}), tmp_path, "user_id")
        write_partitioned(
            pl.DataFrame({"user_id": ["u_novo"], "x": [2]}), tmp_path, "user_id", clear=True
        )

        assert list_collected_users(tmp_path) == {"u_novo"}

    def test_particionamento_sem_clear_preserva_arquivos_antigos(self, tmp_path: Path) -> None:
        """Sem ``clear``, o padrão retomável da coleta continua funcionando."""
        write_partitioned(pl.DataFrame({"user_id": ["u_a"], "x": [1]}), tmp_path, "user_id")
        write_partitioned(pl.DataFrame({"user_id": ["u_b"], "x": [2]}), tmp_path, "user_id")

        assert list_collected_users(tmp_path) == {"u_a", "u_b"}

    def test_concatenacao(self, tmp_path: Path) -> None:
        """Blocos compatíveis são concatenados ao arquivo existente."""
        target = tmp_path / "dados.parquet"
        write_parquet(pl.DataFrame({"a": [1]}), target)
        append_parquet(pl.DataFrame({"a": [2]}), target)

        assert read_parquet(target).height == 2

    def test_concatenacao_com_esquema_incompativel(self, tmp_path: Path) -> None:
        """Esquemas divergentes indicam união equivocada de fontes."""
        target = tmp_path / "dados.parquet"
        write_parquet(pl.DataFrame({"a": [1]}), target)

        with pytest.raises(ValueError, match="incompatíveis"):
            append_parquet(pl.DataFrame({"b": [2]}), target)

    def test_leitura_de_historicos_remove_tweet_id_duplicado_entre_arquivos(
        self, tmp_path: Path
    ) -> None:
        """Uma conta exportada duas vezes sob screen_names diferentes não deve
        duplicar tweets ao consolidar os históricos."""
        tweet = {
            "user_id": "u_mesma_conta",
            "tweet_id": "u_tweet_repetido",
            "text": "oi",
            "created_at": None,
        }
        write_parquet(pl.DataFrame([tweet]), tmp_path / "handle_antigo.parquet")
        write_parquet(pl.DataFrame([tweet]), tmp_path / "handle_novo.parquet")

        combined = read_user_histories(tmp_path)

        assert combined.height == 1

    def test_leitura_de_historicos_preserva_tweets_distintos(self, tmp_path: Path) -> None:
        """Usuários diferentes com tweets distintos não são afetados pelo dedupe."""
        write_parquet(
            pl.DataFrame([{"user_id": "u_a", "tweet_id": "u_t1", "text": "a", "created_at": None}]),
            tmp_path / "usuario_a.parquet",
        )
        write_parquet(
            pl.DataFrame([{"user_id": "u_b", "tweet_id": "u_t2", "text": "b", "created_at": None}]),
            tmp_path / "usuario_b.parquet",
        )

        combined = read_user_histories(tmp_path)

        assert combined.height == 2

    def test_leitura_particionada_concatena_e_ordena(self, tmp_path: Path) -> None:
        """Os arquivos de um diretório particionado voltam concatenados e ordenados."""
        write_partitioned(
            pl.DataFrame({"user_id": ["u_b", "u_a"], "text": ["y", "x"]}), tmp_path, "user_id"
        )

        combined = read_partitioned(tmp_path)

        assert combined.height == 2
        assert combined["user_id"].to_list() == ["u_a", "u_b"]

    def test_leitura_particionada_de_diretorio_vazio_indica_a_etapa(self, tmp_path: Path) -> None:
        """Diretório ausente ou vazio falha com a mesma dica de etapa do reader único."""
        with pytest.raises(DatasetNotFoundError, match="preprocess"):
            read_partitioned(tmp_path / "tweets_clean", stage="preprocess")


class TestCatalogo:
    """Testes do catálogo de artefatos e do manifesto."""

    def test_catalogo_lista_artefatos_esperados(self) -> None:
        """Todos os artefatos do pipeline aparecem no catálogo."""
        from config.paths import get_paths

        catalog = build_catalog(get_paths())

        assert "user_features" in catalog
        assert catalog["user_features"].stage == "features"

    def test_manifesto_registra_versao(self, tmp_path: Path) -> None:
        """O manifesto fecha a tríade código + ambiente + dados."""
        from config.paths import get_paths
        from utils.files import read_json

        paths = get_paths()
        target = write_dataset_manifest(paths, {"n_usuarios": 10})
        payload = read_json(target)

        assert "git_sha" in payload
        assert payload["metadata"]["n_usuarios"] == 10

    def test_comparacao_de_manifesto(self) -> None:
        """A comparação detecta artefatos novos, alterados ou removidos."""
        from config.paths import get_paths

        changes = compare_manifest(get_paths())
        assert isinstance(changes, dict)

    def test_artefato_particionado_conta_como_existente(self, tmp_path: Path) -> None:
        """Um diretório com pelo menos um Parquet é reconhecido como artefato presente."""
        from data.catalog import _artifact_exists, _artifact_size_mb

        directory = tmp_path / "tweets_clean"
        write_partitioned(pl.DataFrame({"user_id": ["u_a"], "x": [1]}), directory, "user_id")

        assert _artifact_exists(directory)
        assert _artifact_size_mb(directory) >= 0.0

    def test_diretorio_vazio_nao_conta_como_existente(self, tmp_path: Path) -> None:
        """Um diretório criado mas sem Parquet ainda não é um artefato produzido."""
        from data.catalog import _artifact_exists

        directory = tmp_path / "tweets_clean"
        directory.mkdir()

        assert not _artifact_exists(directory)
