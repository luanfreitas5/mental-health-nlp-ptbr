"""Testes dos utilitários transversais: hashing, arquivos, léxicos e validação."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from exceptions.data import ClassImbalanceError, EmptyDatasetError, InsufficientDataError
from utils.files import (
    NumpyJSONEncoder,
    list_files,
    read_json,
    read_terms_file,
    write_json,
    write_text,
)
from utils.hashing import (
    build_manifest,
    hash_dataframe,
    hash_directory,
    hash_file,
    hash_payload,
    hash_text,
    pseudonymize,
    pseudonymize_column,
)
from utils.lexicons import load_lexicons, load_stopwords, normalize_term
from utils.timing import Timer, format_duration, log_duration, timed
from utils.validation import (
    check_class_balance,
    check_finite,
    check_no_group_leakage,
    require_columns,
    require_min_rows,
    require_non_empty,
    summarize_missing,
)


class TestPseudonimizacao:
    """Testes da pseudonimização — a proteção central de privacidade (LGPD)."""

    def test_e_deterministica(self) -> None:
        """O mesmo usuário precisa receber o mesmo pseudônimo entre coletas."""
        assert pseudonymize("usuario", "salt") == pseudonymize("usuario", "salt")

    def test_salts_diferentes_produzem_pseudonimos_diferentes(self) -> None:
        """O salt é o que impede a reversão por força bruta sobre handles."""
        assert pseudonymize("usuario", "salt_a") != pseudonymize("usuario", "salt_b")

    def test_usuarios_diferentes_produzem_pseudonimos_diferentes(self) -> None:
        """Identificadores distintos não colidem."""
        assert pseudonymize("a", "salt") != pseudonymize("b", "salt")

    def test_salt_vazio_e_rejeitado(self) -> None:
        """Pseudonimizar sem salt não protege nada e deve falhar alto."""
        with pytest.raises(ValueError, match="Salt vazio"):
            pseudonymize("usuario", "")

    def test_formato_do_pseudonimo(self) -> None:
        """O pseudônimo segue o formato validado pelos contratos pandera."""
        result = pseudonymize("usuario", "salt")
        assert result.startswith("u_")
        assert len(result) == 18

    def test_aceita_identificador_numerico(self) -> None:
        """Identificadores numéricos da API são aceitos sem conversão prévia."""
        assert pseudonymize(12345, "salt").startswith("u_")

    def test_pseudonymize_column_remove_original(self) -> None:
        """A coluna com o identificador direto não sobrevive à transformação."""
        frame = pl.DataFrame({"handle": ["alice", "bob"]})
        result = pseudonymize_column(frame, "handle", "salt", alias="user_id")

        assert "handle" not in result.columns
        assert result["user_id"].to_list()[0].startswith("u_")

    def test_pseudonymize_column_falha_sem_a_coluna(self) -> None:
        """Coluna ausente falha com mensagem que lista as disponíveis."""
        with pytest.raises(KeyError, match="ausente"):
            pseudonymize_column(pl.DataFrame({"x": [1]}), "handle", "salt")


class TestHashing:
    """Testes dos hashes usados na rastreabilidade de dados."""

    def test_hash_text_e_estavel(self) -> None:
        """O hash de um texto conhecido não muda entre execuções."""
        assert hash_text("abc").startswith("ba7816bf")

    def test_hash_dataframe_ignora_ordem_das_colunas(self) -> None:
        """Reordenar colunas não muda a identidade do conteúdo."""
        a = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        b = pl.DataFrame({"y": [3, 4], "x": [1, 2]})
        assert hash_dataframe(a) == hash_dataframe(b)

    def test_hash_dataframe_detecta_mudanca_de_valor(self) -> None:
        """Qualquer alteração de conteúdo muda o hash."""
        a = pl.DataFrame({"x": [1, 2]})
        b = pl.DataFrame({"x": [1, 3]})
        assert hash_dataframe(a) != hash_dataframe(b)

    def test_hash_payload_ignora_ordem_das_chaves(self) -> None:
        """O cache do LLM depende disso: mesma requisição, mesma chave."""
        assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})

    def test_hash_file_de_arquivo_inexistente(self, tmp_path: Path) -> None:
        """Arquivo ausente falha com FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            hash_file(tmp_path / "inexistente.txt")

    def test_manifesto_marca_artefato_ausente(self, tmp_path: Path) -> None:
        """Artefatos ausentes aparecem no manifesto como 'ausente'."""
        manifest = build_manifest({"faltante": tmp_path / "x.parquet"})
        assert manifest["faltante"]["status"] == "ausente"

    def test_manifesto_calcula_hash_de_artefato_existente(self, tmp_path: Path) -> None:
        """Artefatos existentes recebem hash e tamanho."""
        target = tmp_path / "dados.txt"
        target.write_text("conteúdo", encoding="utf-8")

        entry = build_manifest({"dados": target})["dados"]
        assert "sha256" in entry
        assert entry["size_bytes"] > 0

    def test_hash_directory_de_diretorio_inexistente(self, tmp_path: Path) -> None:
        """Diretório ausente ou sem Parquet falha com FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            hash_directory(tmp_path / "inexistente")

    def test_hash_directory_e_independente_da_ordem_de_listagem(self, tmp_path: Path) -> None:
        """O hash combinado não depende da ordem em que os arquivos são criados."""
        (tmp_path / "u_b.parquet").write_bytes(b"b")
        (tmp_path / "u_a.parquet").write_bytes(b"a")

        outro = tmp_path.parent / "outra_ordem"
        outro.mkdir()
        (outro / "u_a.parquet").write_bytes(b"a")
        (outro / "u_b.parquet").write_bytes(b"b")

        assert hash_directory(tmp_path) == hash_directory(outro)

    def test_hash_directory_detecta_mudanca_de_conteudo(self, tmp_path: Path) -> None:
        """Alterar o conteúdo de um arquivo muda o hash combinado."""
        (tmp_path / "u_a.parquet").write_bytes(b"a")
        antes = hash_directory(tmp_path)

        (tmp_path / "u_a.parquet").write_bytes(b"outro conteudo")
        assert hash_directory(tmp_path) != antes

    def test_manifesto_trata_diretorio_particionado_como_artefato(self, tmp_path: Path) -> None:
        """Artefatos particionados em vários arquivos recebem hash combinado e n_files."""
        directory = tmp_path / "tweets_clean"
        directory.mkdir()
        (directory / "u_a.parquet").write_bytes(b"a")
        (directory / "u_b.parquet").write_bytes(b"b")

        entry = build_manifest({"tweets_clean": directory})["tweets_clean"]
        assert "sha256" in entry
        assert entry["n_files"] == 2
        assert entry["size_bytes"] == 2

    def test_manifesto_marca_diretorio_vazio_como_ausente(self, tmp_path: Path) -> None:
        """Um diretório criado mas sem nenhum Parquet ainda é 'ausente' no manifesto."""
        directory = tmp_path / "tweets_clean"
        directory.mkdir()

        entry = build_manifest({"tweets_clean": directory})["tweets_clean"]
        assert entry["status"] == "ausente"


class TestArquivos:
    """Testes de leitura de termos e escrita atômica."""

    def test_le_lexico_real(self) -> None:
        """Os léxicos versionados são legíveis e não estão vazios."""
        from config.paths import get_paths

        assert len(read_terms_file(get_paths().lexicons.death)) > 5

    def test_ignora_comentarios_mas_preserva_hashtags(self, tmp_path: Path) -> None:
        """'# ' inicia comentário; '#termo' é conteúdo (hashtag)."""
        target = tmp_path / "termos.txt"
        target.write_text("# comentário\n#hashtag\n\ntermo\n", encoding="utf-8")

        assert read_terms_file(target) == ["#hashtag", "termo"]

    def test_remove_duplicatas_preservando_ordem(self, tmp_path: Path) -> None:
        """Termos repetidos aparecem uma única vez, na ordem original."""
        target = tmp_path / "termos.txt"
        target.write_text("b\na\nb\n", encoding="utf-8")

        assert read_terms_file(target) == ["b", "a"]

    def test_arquivo_ausente_levanta_erro(self, tmp_path: Path) -> None:
        """Léxico ausente falha com mensagem que aponta configs/paths.yaml."""
        with pytest.raises(FileNotFoundError, match=r"paths\.yaml"):
            read_terms_file(tmp_path / "inexistente.txt")

    def test_json_serializa_tipos_numpy(self, tmp_path: Path) -> None:
        """Métricas do scikit-learn são escalares NumPy e precisam ser aceitas."""
        target = tmp_path / "metricas.json"
        write_json(target, {"f1": np.float32(0.75), "n": np.int64(10), "v": np.array([1, 2])})

        payload = read_json(target)
        assert payload["f1"] == pytest.approx(0.75)
        assert payload["n"] == 10
        assert payload["v"] == [1, 2]

    def test_escrita_cria_diretorios(self, tmp_path: Path) -> None:
        """Diretórios intermediários são criados automaticamente."""
        target = write_text(tmp_path / "a" / "b" / "c.md", "conteúdo")
        assert target.is_file()

    def test_escrita_nao_deixa_arquivo_temporario(self, tmp_path: Path) -> None:
        """A escrita atômica não deixa resíduo após concluir."""
        write_json(tmp_path / "dados.json", {"a": 1})
        assert not list(tmp_path.glob("*.tmp"))

    def test_encoder_rejeita_tipo_desconhecido(self) -> None:
        """Tipos não previstos continuam falhando, em vez de virar string silenciosa."""
        with pytest.raises(TypeError):
            NumpyJSONEncoder().default(object())

    def test_list_files_e_ordenado(self, tmp_path: Path) -> None:
        """A ordenação garante reprodutibilidade ao concatenar arquivos."""
        for name in ("c.txt", "a.txt", "b.txt"):
            (tmp_path / name).write_text("x", encoding="utf-8")

        assert [path.name for path in list_files(tmp_path, "*.txt")] == [
            "a.txt",
            "b.txt",
            "c.txt",
        ]

    def test_list_files_de_diretorio_inexistente(self, tmp_path: Path) -> None:
        """Diretório ausente devolve lista vazia, sem exceção."""
        assert list_files(tmp_path / "nada") == []


class TestLexicos:
    """Testes dos léxicos psicolinguísticos."""

    def test_normalize_term_remove_acentos_e_caixa(self) -> None:
        """A comparação com o texto é robusta a variação ortográfica."""
        assert normalize_term("Solidão") == "solidao"

    def test_lexicos_carregados(self) -> None:
        """Todos os léxicos de risco declarados são carregados."""
        lexicons = load_lexicons()
        assert {"death", "loneliness", "hopelessness"}.issubset(set(lexicons))

    def test_contains_detecta_termo(self) -> None:
        """Um termo do léxico é detectado no texto."""
        assert load_lexicons()["loneliness"].contains("me sinto sozinho hoje")

    def test_contains_ignora_texto_sem_termo(self) -> None:
        """Texto sem nenhum termo não é falso positivo."""
        assert not load_lexicons()["death"].contains("hoje o dia foi ensolarado")

    def test_contains_e_robusto_a_acento(self) -> None:
        """'solidao' sem acento casa com o termo acentuado do léxico."""
        assert load_lexicons()["loneliness"].contains("sentindo solidao demais")

    def test_count_conta_todas_as_ocorrencias(self) -> None:
        """A contagem reflete repetições dentro do mesmo texto."""
        assert load_lexicons()["loneliness"].count("sozinho e sozinho de novo") == 2

    def test_nao_casa_dentro_de_outra_palavra(self) -> None:
        """A fronteira de palavra impede casamento parcial."""
        assert not load_lexicons()["death"].contains("amortecedor do carro")

    def test_stopwords_carregadas(self) -> None:
        """As stopwords são carregadas e contêm termos esperados."""
        assert "de" in load_stopwords()

    def test_stopwords_nao_contem_primeira_pessoa(self) -> None:
        """Pronomes de 1ª pessoa são feature central e não podem ser stopword."""
        assert "eu" not in load_stopwords()


class TestValidacao:
    """Testes das verificações defensivas."""

    def test_require_columns_aceita_colunas_presentes(self) -> None:
        """Colunas presentes passam sem exceção."""
        require_columns(pl.DataFrame({"a": [1], "b": [2]}), ["a", "b"])

    def test_require_columns_lista_as_ausentes(self) -> None:
        """A mensagem de erro nomeia as colunas que faltam."""
        with pytest.raises(KeyError, match="ausentes"):
            require_columns(pl.DataFrame({"a": [1]}), ["a", "b"])

    def test_require_non_empty_rejeita_vazio(self) -> None:
        """DataFrame vazio depois de filtros é erro, não resultado válido."""
        with pytest.raises(EmptyDatasetError):
            require_non_empty(pl.DataFrame({"a": []}))

    def test_require_min_rows(self) -> None:
        """Volume abaixo do mínimo exigido falha com erro tipado."""
        with pytest.raises(InsufficientDataError):
            require_min_rows(pl.DataFrame({"a": [1]}), minimum=5)

    def test_check_class_balance_conta_classes(self) -> None:
        """A contagem por classe é devolvida mesmo sem violação."""
        assert check_class_balance(["a", "a", "b"], max_ratio=3.0) == {"a": 2, "b": 1}

    def test_check_class_balance_levanta_quando_configurado(self) -> None:
        """Com `raise_on_violation`, o desbalanceamento severo interrompe."""
        with pytest.raises(ClassImbalanceError):
            check_class_balance(["a"] * 10 + ["b"], max_ratio=2.0, raise_on_violation=True)

    def test_check_no_group_leakage_aceita_particoes_disjuntas(self) -> None:
        """Partições sem interseção passam."""
        check_no_group_leakage(["u1", "u2"], ["u3"])

    def test_check_no_group_leakage_detecta_vazamento(self) -> None:
        """Usuário em treino e teste é o vazamento mais grave do projeto."""
        with pytest.raises(ValueError, match="Vazamento"):
            check_no_group_leakage(["u1", "u2"], ["u2"])

    def test_check_finite_detecta_nan(self) -> None:
        """NaN na matriz de atributos é erro, não valor válido."""
        with pytest.raises(ValueError, match="não finitos"):
            check_finite(np.array([1.0, np.nan]))

    def test_summarize_missing_ordena_por_ausencia(self) -> None:
        """A coluna com mais ausentes aparece primeiro."""
        frame = pl.DataFrame({"a": [1, None], "b": [1, 2]})
        assert summarize_missing(frame).row(0)[0] == "a"


class TestTiming:
    """Testes de medição de tempo."""

    def test_format_duration_em_segundos(self) -> None:
        """Durações curtas usam vírgula decimal (pt-BR)."""
        assert format_duration(0.42) == "0,42s"

    def test_format_duration_em_minutos(self) -> None:
        """Durações médias são quebradas em minutos e segundos."""
        assert format_duration(65.3) == "1min 5,3s"

    def test_format_duration_em_horas(self) -> None:
        """Durações longas são quebradas em horas, minutos e segundos."""
        assert format_duration(3725) == "1h 2min 5s"

    def test_timer_mede_tempo(self) -> None:
        """O cronômetro registra tempo não negativo."""
        timer = Timer()
        with timer:
            sum(range(100))
        assert timer.elapsed >= 0

    def test_log_duration_devolve_cronometro(self) -> None:
        """O contexto expõe o cronômetro após a saída."""
        with log_duration("teste") as timer:
            sum(range(100))
        assert timer.elapsed >= 0

    def test_timed_preserva_retorno(self) -> None:
        """O decorador não altera o resultado da função."""

        @timed
        def calculate_sum(n: int) -> int:
            return sum(range(n))

        assert calculate_sum(10) == 45
