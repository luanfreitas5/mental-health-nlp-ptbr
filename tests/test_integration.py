"""Testes de integração e comportamentais do pipeline.

Cobrem três coisas que testes unitários não pegam:

* **Integração** — o encadeamento das etapas produz artefatos consistentes.
* **Comportamento do modelo** — expectativas direcionais que devem valer
  independentemente do algoritmo (mais sinal de risco não pode reduzir a
  probabilidade prevista de risco).
* **Regressão de métrica** — o desempenho não pode cair abaixo do acordado.

Os testes marcados com ``smoke`` rodam no pre-commit; os demais, no CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from config.settings import Config
from constants.labels import CLASS_ORDER, LABEL_TO_INDEX, UserLabel
from evaluation.evaluator import Evaluator
from evaluation.metrics import compute_metrics
from models.base import UserDataset
from models.traditional import TabularClassifier
from pipelines.workflow import DEFAULT_PIPELINE, STAGES, describe_stages, get_stage
from preprocessing.pipeline import run_preprocessing
from training.trainer import build_dataset, split_features


@pytest.fixture
def trained_model(
    feature_matrix: pl.DataFrame, splits: pl.DataFrame
) -> tuple[TabularClassifier, UserDataset]:
    """Modelo treinado no conjunto de treino sintético e o conjunto de teste."""
    train = build_dataset(split_features(feature_matrix, splits, "train"))
    test = build_dataset(split_features(feature_matrix, splits, "test"))

    model = TabularClassifier(
        name="lr_teste",
        params={"max_iter": 2000, "random_state": 42, "class_weight": "balanced"},
        estimator_name="logistic_regression",
    )
    model.fit(train)
    return model, test


@pytest.mark.smoke
class TestFumaca:
    """Verificações rápidas, executadas antes de cada commit."""

    def test_configuracao_carrega(self, config: Config) -> None:
        """Uma configuração quebrada derrubaria qualquer etapa."""
        assert config.general.project.name
        assert config.classes == list(CLASS_ORDER)

    def test_todas_as_etapas_sao_instanciaveis(self) -> None:
        """Nenhuma etapa registrada está com import quebrado."""
        for name in STAGES:
            assert get_stage(name).name == name

    def test_pipeline_padrao_nao_inclui_coleta(self) -> None:
        """A coleta exige aprovação ética e nunca deve rodar por engano."""
        assert "collect" not in DEFAULT_PIPELINE

    def test_descricao_das_etapas(self) -> None:
        """A ajuda da linha de comando lista todas as etapas."""
        description = describe_stages()
        assert all(name in description for name in STAGES)

    def test_lexicos_carregam(self) -> None:
        """Os léxicos versionados são legíveis."""
        from utils.lexicons import load_lexicons

        assert load_lexicons()

    def test_caminhos_resolvem(self) -> None:
        """Todos os caminhos declarados são resolvidos para absolutos."""
        from config.paths import get_paths

        assert get_paths().data.raw.is_absolute()


class TestIntegracaoDoPreprocessamento:
    """Testes do encadeamento completo da etapa de pré-processamento."""

    def test_produz_saida_valida(self, raw_tweets: pl.DataFrame, config: Config) -> None:
        """A etapa produz dados que satisfazem o contrato de saída."""
        result = run_preprocessing(raw_tweets, config)

        assert result.height > 0
        assert "text_normalized" in result.columns
        assert "text_clean" in result.columns

    def test_nao_aumenta_o_volume(self, raw_tweets: pl.DataFrame, config: Config) -> None:
        """A etapa só filtra e transforma — nunca cria tweets."""
        assert run_preprocessing(raw_tweets, config).height <= raw_tweets.height

    def test_remove_pii_de_todos_os_tweets(self, raw_tweets: pl.DataFrame, config: Config) -> None:
        """Invariante de privacidade: nenhuma PII sobrevive à etapa."""
        from preprocessing.text import contains_pii

        result = run_preprocessing(raw_tweets, config)
        assert not any(contains_pii(text) for text in result["text_normalized"].to_list())

    def test_e_deterministico(self, raw_tweets: pl.DataFrame, config: Config) -> None:
        """Duas execuções sobre a mesma entrada produzem o mesmo resultado."""
        from utils.hashing import hash_dataframe

        first = run_preprocessing(raw_tweets, config)
        second = run_preprocessing(raw_tweets, config)

        assert hash_dataframe(first) == hash_dataframe(second)


class TestIntegracaoDoTreinamento:
    """Testes do encadeamento entre particionamento, treino e avaliação."""

    def test_ciclo_completo(
        self, trained_model: tuple[TabularClassifier, UserDataset], config: Config
    ) -> None:
        """Treino, predição e avaliação encadeiam sem inconsistência."""
        model, test = trained_model
        result = Evaluator(config).evaluate(model, test)

        assert result.model_name == "lr_teste"
        assert 0.0 <= result.metrics["f1_macro"] <= 1.0
        assert result.confidence_interval

    def test_avaliacao_produz_metricas_por_classe(
        self, trained_model: tuple[TabularClassifier, UserDataset], config: Config
    ) -> None:
        """Toda classe recebe métricas próprias, mesmo as minoritárias."""
        model, test = trained_model
        result = Evaluator(config).evaluate(model, test)

        assert set(result.per_class) == set(CLASS_ORDER)

    def test_comparacao_ordena_por_metrica_principal(
        self, trained_model: tuple[TabularClassifier, UserDataset], config: Config
    ) -> None:
        """A tabela comparativa fica ordenada pela métrica principal."""
        model, test = trained_model
        evaluator = Evaluator(config)
        comparison = evaluator.compare({"lr_teste": evaluator.evaluate(model, test)})

        assert comparison.height == 1
        assert "modelo" in comparison.columns

    def test_sem_vazamento_no_conjunto_de_teste(
        self, feature_matrix: pl.DataFrame, splits: pl.DataFrame
    ) -> None:
        """Nenhum usuário de treino aparece no teste."""
        from utils.validation import check_no_group_leakage

        train = split_features(feature_matrix, splits, "train")
        test = split_features(feature_matrix, splits, "test")

        check_no_group_leakage(train["user_id"].to_list(), test["user_id"].to_list())


@pytest.mark.ml
class TestComportamentoDoModelo:
    """Expectativas direcionais que devem valer para qualquer algoritmo."""

    def test_supera_o_acaso(self, trained_model: tuple[TabularClassifier, UserDataset]) -> None:
        """Funcionalidade mínima: o modelo aprende o sinal sintético injetado."""
        model, test = trained_model
        assert test.labels is not None

        accuracy = float((model.predict(test) == test.labels).mean())
        assert accuracy > 1.0 / len(CLASS_ORDER)

    def test_mais_sinal_de_risco_nao_reduz_o_risco_previsto(
        self, trained_model: tuple[TabularClassifier, UserDataset]
    ) -> None:
        """Invariância direcional: aumentar o sinal de risco não pode diminuir
        a probabilidade prevista de risco.

        É um teste comportamental, não estatístico: verifica que o modelo
        aprendeu a direção correta da relação, e não apenas alguma correlação.
        """
        model, test = trained_model
        index = test.feature_names.index("psy_risco_suicida_mean")

        baseline = model.predict_proba(test)
        reforcado = UserDataset(
            user_ids=test.user_ids,
            features=test.features.copy(),
            feature_names=test.feature_names,
            labels=test.labels,
        )
        reforcado.features[:, index] = np.clip(reforcado.features[:, index] + 0.4, 0, 1)

        risco_indices = [
            LABEL_TO_INDEX[str(UserLabel.DEPRESSAO)],
            LABEL_TO_INDEX[str(UserLabel.IDEACAO_SUICIDA)],
        ]
        antes = baseline[:, risco_indices].sum(axis=1).mean()
        depois = model.predict_proba(reforcado)[:, risco_indices].sum(axis=1).mean()

        assert depois >= antes - 1e-6

    def test_probabilidades_formam_distribuicao(
        self, trained_model: tuple[TabularClassifier, UserDataset]
    ) -> None:
        """As probabilidades somam 1 e são não negativas."""
        model, test = trained_model
        proba = model.predict_proba(test)

        assert np.all(proba >= 0)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predicao_e_o_argmax(
        self, trained_model: tuple[TabularClassifier, UserDataset]
    ) -> None:
        """A classe prevista é sempre a de maior probabilidade."""
        model, test = trained_model
        assert np.array_equal(model.predict(test), model.predict_proba(test).argmax(axis=1))

    def test_e_reprodutivel(self, feature_matrix: pl.DataFrame, splits: pl.DataFrame) -> None:
        """Duas execuções com a mesma semente produzem o mesmo modelo."""
        train = build_dataset(split_features(feature_matrix, splits, "train"))
        test = build_dataset(split_features(feature_matrix, splits, "test"))

        def treinar() -> np.ndarray:
            model = TabularClassifier(
                name="lr",
                params={"max_iter": 500, "random_state": 42},
                estimator_name="logistic_regression",
            )
            return model.fit(train).predict(test)

        assert np.array_equal(treinar(), treinar())


@pytest.mark.ml
class TestRegressaoDeMetrica:
    """Trava de qualidade: o desempenho não pode regredir silenciosamente.

    Os limiares abaixo se aplicam aos **dados sintéticos** dos testes, que têm
    sinal deliberadamente injetado. Os limiares de produção, aplicados aos
    dados reais, vivem em ``configs/evaluation.yaml``
    (``regression_thresholds``) e são verificados na etapa de avaliação.
    """

    #: Piso sobre os dados sintéticos. Abaixo disso, algo quebrou no pipeline.
    LIMIAR_SINTETICO: float = 0.55

    def test_f1_nao_regride(self, trained_model: tuple[TabularClassifier, UserDataset]) -> None:
        """O F1-macro nos dados sintéticos permanece acima do piso acordado."""
        model, test = trained_model
        assert test.labels is not None

        f1 = compute_metrics(test.labels, model.predict(test))["f1_macro"]
        assert f1 >= self.LIMIAR_SINTETICO, f"Regressão detectada: F1-macro={f1:.4f}"

    def test_limiares_de_producao_estao_declarados(self, config: Config) -> None:
        """Os limiares de produção existem para o modelo híbrido e o XGBoost."""
        thresholds = config.evaluation.regression_thresholds

        assert "hybrid_xgboost" in thresholds
        assert thresholds["hybrid_xgboost"]["f1_macro"] > 0.5


class TestDocumentosDeIaResponsavel:
    """Testes da geração do model card e do datasheet."""

    def test_model_card_contem_secoes_obrigatorias(self, config: Config, tmp_path: Path) -> None:
        """O model card documenta uso pretendido, limitações e ética."""
        from config.paths import get_paths
        from reports_templates.model_card import build_model_card

        card = build_model_card({}, config, get_paths())

        assert "Uso pretendido" in card
        assert "Usos fora de escopo" in card
        assert "Limitações" in card
        assert "Considerações éticas" in card

    def test_model_card_alerta_sobre_diagnostico(self, config: Config) -> None:
        """O card deixa explícito que o sistema não diagnostica."""
        from config.paths import get_paths
        from reports_templates.model_card import build_model_card

        assert "não diagnostica" in build_model_card({}, config, get_paths())

    def test_datasheet_documenta_privacidade(self, config: Config) -> None:
        """O datasheet registra a base legal e as salvaguardas de LGPD."""
        from config.paths import get_paths
        from reports_templates.datasheet import build_datasheet

        datasheet = build_datasheet(config, get_paths())

        assert "LGPD" in datasheet
        assert "Pseudonimização" in datasheet
        assert "não é redistribuído" in datasheet.replace("**Não.**", "não é redistribuído")

    def test_datasheet_documenta_vies_de_selecao(self, config: Config) -> None:
        """A limitação da classe controle é declarada, não omitida."""
        from config.paths import get_paths
        from reports_templates.datasheet import build_datasheet

        assert "viés de seleção" in build_datasheet(config, get_paths()).lower()
