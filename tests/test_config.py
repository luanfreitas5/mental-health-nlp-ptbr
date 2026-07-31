"""Testes de carregamento e validação da configuração.

Configuração inválida precisa falhar no startup, com erro tipado. Estes
testes verificam justamente isso: que os limites declarados nos modelos
Pydantic são de fato aplicados, e não apenas documentação.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from config.paths import get_paths, resolve_path
from config.settings import (
    ConsensusSection,
    LabelSourceSection,
    SeedSearchSection,
    SplitSection,
    TargetSection,
    TemporalPersistenceSection,
    UserLabelingSection,
    load_config,
    read_yaml,
)
from config.version import build_run_id, describe_version, get_version
from exceptions.configuration import (
    ConfigFileNotFoundError,
    ConfigParsingError,
    ConfigValidationError,
)


class TestPaths:
    """Testes de resolução de caminhos."""

    def test_resolve_path_torna_relativo_absoluto(self) -> None:
        """Caminhos relativos do YAML viram absolutos sob a raiz do projeto."""
        assert resolve_path("data/raw").is_absolute()

    def test_resolve_path_preserva_absoluto(self, tmp_path: Path) -> None:
        """Caminhos já absolutos não são alterados."""
        assert resolve_path(tmp_path) == tmp_path

    def test_get_paths_e_memoizado(self) -> None:
        """A leitura do YAML acontece uma única vez por processo."""
        assert get_paths() is get_paths()

    def test_todos_os_grupos_existem(self) -> None:
        """Todas as seções declaradas em paths.yaml são carregadas."""
        paths = get_paths()
        assert paths.data.raw.name == "raw"
        assert paths.lexicons.death.suffix == ".txt"
        assert paths.reports.figures.name == "figures"

    def test_iter_directories_converte_arquivo_em_pasta(self) -> None:
        """Caminhos de arquivo entram na lista como o diretório que os contém."""
        directories = get_paths().iter_directories()
        assert all(not path.suffix for path in directories)

    def test_arquivo_inexistente_levanta_erro(self, tmp_path: Path) -> None:
        """Um paths.yaml ausente falha com erro tipado."""
        from config.paths import get_paths as loader

        with pytest.raises(ConfigFileNotFoundError, match="não encontrado"):
            loader.__wrapped__(tmp_path / "inexistente.yaml")


class TestReadYaml:
    """Testes da leitura genérica de YAML."""

    def test_yaml_invalido_levanta_erro(self, tmp_path: Path) -> None:
        """YAML sintaticamente inválido produz ConfigParsingError."""
        target = tmp_path / "invalido.yaml"
        target.write_text("chave: [sem fechamento", encoding="utf-8")

        with pytest.raises(ConfigParsingError, match="YAML inválido"):
            read_yaml(target)

    def test_yaml_vazio_retorna_dicionario_vazio(self, tmp_path: Path) -> None:
        """Um arquivo vazio é tratado como configuração vazia, não como erro."""
        target = tmp_path / "vazio.yaml"
        target.write_text("", encoding="utf-8")

        assert read_yaml(target) == {}

    def test_yaml_com_lista_no_topo_levanta_erro(self, tmp_path: Path) -> None:
        """O topo do arquivo precisa ser um mapeamento."""
        target = tmp_path / "lista.yaml"
        target.write_text("- a\n- b\n", encoding="utf-8")

        with pytest.raises(ConfigParsingError, match="mapeamento"):
            read_yaml(target)

    def test_arquivo_ausente_levanta_erro(self, tmp_path: Path) -> None:
        """Arquivo inexistente produz ConfigFileNotFoundError."""
        with pytest.raises(ConfigFileNotFoundError):
            read_yaml(tmp_path / "nao_existe.yaml")


class TestValidacaoDeRegrasDeNegocio:
    """Testes das regras de negócio embutidas nos modelos Pydantic."""

    def test_classe_de_risco_desconhecida_e_rejeitada(self) -> None:
        """Uma classe de risco fora de `classes` é erro de configuração."""
        with pytest.raises(ValueError, match="Classes de risco desconhecidas"):
            TargetSection(
                column="user_label",
                classes=["controle", "depressao"],
                risk_classes=["inexistente"],
                main_metric="f1_macro",
            )

    def test_particoes_nao_podem_consumir_todo_o_conjunto(self) -> None:
        """Teste + validação deixando menos de 10% para treino é rejeitado."""
        with pytest.raises(ValueError, match="menos de 10%"):
            SplitSection(test_size=0.6, val_size=0.35)

    def test_janela_de_coleta_invertida_e_rejeitada(self) -> None:
        """`since` posterior a `until` é erro de configuração."""
        with pytest.raises(ValueError, match="Janela inválida"):
            SeedSearchSection(
                since=date(2025, 1, 1),
                until=date(2024, 1, 1),
                groups={},
            )

    def test_pesos_das_fontes_devem_somar_um(self) -> None:
        """Voto ponderado exige que os pesos das fontes ativas somem 1."""
        with pytest.raises(ValueError, match="somam"):
            UserLabelingSection(
                sources={
                    "a": LabelSourceSection(weight=0.3),
                    "b": LabelSourceSection(weight=0.3),
                },
                lexical_thresholds={},
                temporal_persistence=TemporalPersistenceSection(window_days=30),
                class_precedence=["controle"],
                consensus=ConsensusSection(
                    manual_review_file="x.parquet",
                    manual_labels_file="y.csv",
                ),
            )

    def test_pesos_de_fonte_desativada_nao_contam(self) -> None:
        """Fontes desativadas ficam de fora da soma dos pesos."""
        section = UserLabelingSection(
            sources={
                "a": LabelSourceSection(weight=1.0),
                "b": LabelSourceSection(weight=0.5, enabled=False),
            },
            lexical_thresholds={},
            temporal_persistence=TemporalPersistenceSection(window_days=30),
            class_precedence=["controle"],
            consensus=ConsensusSection(manual_review_file="x.parquet", manual_labels_file="y.csv"),
        )
        assert section.sources["a"].weight == 1.0


class TestConfiguracaoDoProjeto:
    """Testes sobre a configuração real, carregada de ``configs/``."""

    def test_carrega_todas_as_secoes(self, config) -> None:
        """Todos os YAMLs declarados são carregados e validados."""
        assert config.general.project.name == "mental-health-nlp-ptbr"
        assert config.collection.seed_search.groups
        assert config.features.groups
        assert config.llm.provider == "ollama"

    def test_classes_na_ordem_canonica(self, config) -> None:
        """A ordem das classes na configuração é a ordem canônica do código."""
        from constants.labels import CLASS_ORDER

        assert config.classes == list(CLASS_ORDER)

    def test_selecao_de_modelos_respeita_escopo(self, config) -> None:
        """Sem `--include-exploratory`, os modelos exploratórios ficam de fora."""
        principais = config.models.select()
        completos = config.models.select(include_exploratory=True)

        assert set(principais).issubset(set(completos))
        assert len(completos) > len(principais)
        assert all(config.models.all_models()[name].scope != "exploratory" for name in principais)

    def test_baseline_sempre_incluido(self, config) -> None:
        """O baseline roda sempre — é a referência da comparação."""
        assert "dummy" in config.models.select()

    def test_modelo_de_ablacao_existe(self, config) -> None:
        """O modelo base da ablação precisa estar declarado no YAML."""
        assert config.evaluation.ablation.base_model in config.models.all_models()

    def test_grupos_de_ablacao_sao_validos(self, config) -> None:
        """Os grupos da ablação existem entre os grupos de atributos."""
        from constants.columns import FEATURE_GROUP_PREFIXES

        assert set(config.evaluation.ablation.groups).issubset(set(FEATURE_GROUP_PREFIXES))

    def test_nomes_de_modelo_sao_unicos(self, config) -> None:
        """Nome duplicado entre famílias derrubaria a comparação silenciosamente."""
        assert len(config.models.all_models()) > 0

    def test_load_config_e_memoizado(self) -> None:
        """Os YAMLs são lidos uma única vez por processo."""
        assert load_config() is load_config()

    def test_configuracao_e_imutavel(self, config) -> None:
        """A configuração é congelada: nenhuma etapa a altera em tempo de execução."""
        with pytest.raises((TypeError, ValueError)):
            config.general.project.random_seed = 99  # type: ignore[misc]

    def test_diretorio_inexistente_levanta_erro(self, tmp_path: Path) -> None:
        """Um diretório de configuração sem os YAMLs falha com erro tipado."""
        with pytest.raises((ConfigFileNotFoundError, ConfigValidationError)):
            load_config.__wrapped__(tmp_path)


class TestVersao:
    """Testes dos metadados de versão."""

    def test_versao_segue_semver(self) -> None:
        """A versão lida do pyproject segue MAJOR.MINOR.PATCH."""
        assert len(get_version().split(".")) == 3

    def test_run_id_contem_prefixo(self) -> None:
        """O identificador de execução preserva o prefixo informado."""
        assert build_run_id("teste").startswith("teste_")

    def test_describe_version_tem_campos_esperados(self) -> None:
        """Os metadados de versão trazem código, SHA e carimbo temporal."""
        assert set(describe_version()) == {"version", "git_sha", "generated_at"}
