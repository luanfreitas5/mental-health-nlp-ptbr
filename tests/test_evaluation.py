"""Testes das métricas, calibração, fatias e testes estatísticos."""

from __future__ import annotations

import numpy as np
import pytest

from config.settings import SliceDefinition, SlicesSection, StatisticsSection, TestToggle
from constants.metrics import METRIC_DIRECTION, is_higher_better
from evaluation.calibration import (
    compute_brier_score,
    compute_expected_calibration_error,
    evaluate_calibration,
)
from evaluation.metrics import (
    bootstrap_confidence_interval,
    compute_confusion_matrix,
    compute_metrics,
    compute_per_class_metrics,
    format_metric_with_ci,
    summarize,
)
from evaluation.slices import assign_slices, evaluate_by_slice
from evaluation.statistics import (
    cliffs_delta,
    compare_all_models,
    friedman_test,
    holm_correction,
    interpret_effect_size,
    mcnemar_test,
    nemenyi_critical_difference,
    wilcoxon_test,
)


class TestMetricas:
    """Testes do cálculo de métricas de classificação."""

    def test_metricas_basicas(self) -> None:
        """As métricas agregadas são calculadas corretamente."""
        metrics = compute_metrics(np.array([0, 1, 2]), np.array([0, 1, 1]))
        assert metrics["accuracy"] == pytest.approx(2 / 3)

    def test_predicao_perfeita(self) -> None:
        """Predição perfeita produz métricas máximas."""
        y = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_metrics(y, y)

        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_macro"] == pytest.approx(1.0)
        assert metrics["mcc"] == pytest.approx(1.0)

    def test_metricas_de_probabilidade_exigem_proba(self) -> None:
        """Sem probabilidades, ROC-AUC e PR-AUC são omitidos — não zerados."""
        metrics = compute_metrics(np.array([0, 1]), np.array([0, 1]))
        assert "roc_auc_ovr" not in metrics

    def test_metricas_de_probabilidade_com_proba(
        self, predictions: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Com probabilidades, as métricas de ranqueamento são calculadas."""
        y_true, y_pred, y_proba = predictions
        metrics = compute_metrics(y_true, y_pred, y_proba)

        assert 0.0 <= metrics["roc_auc_ovr"] <= 1.0

    def test_metricas_por_classe(self) -> None:
        """Cada classe recebe precisão, revocação, F1 e suporte."""
        result = compute_per_class_metrics(np.array([0, 1]), np.array([0, 0]))

        assert result["controle"]["recall"] == pytest.approx(1.0)
        assert result["depressao"]["recall"] == pytest.approx(0.0)

    def test_matriz_de_confusao(self) -> None:
        """A matriz de confusão respeita a ordem canônica das classes."""
        matrix = compute_confusion_matrix(np.array([0, 1]), np.array([0, 0]), n_classes=2)
        assert matrix.tolist() == [[1, 0], [1, 0]]

    def test_matriz_normalizada_por_linha(self) -> None:
        """Normalizada, cada linha soma 1 (revocação por classe)."""
        matrix = compute_confusion_matrix(
            np.array([0, 0, 1]), np.array([0, 1, 1]), n_classes=2, normalize=True
        )
        assert matrix[0].sum() == pytest.approx(1.0)

    def test_orientacao_das_metricas(self) -> None:
        """Nem toda métrica é 'maior é melhor' — o Brier é o contraexemplo."""
        assert is_higher_better("f1_macro")
        assert not is_higher_better("brier_score")

    def test_metrica_desconhecida_e_rejeitada(self) -> None:
        """Uma métrica não registrada falha em vez de assumir uma direção."""
        with pytest.raises(KeyError, match="desconhecida"):
            is_higher_better("metrica_inventada")

    def test_todas_as_metricas_padrao_tem_direcao(self) -> None:
        """Toda métrica computada tem orientação declarada."""
        from constants.metrics import DEFAULT_METRICS

        assert set(DEFAULT_METRICS).issubset(set(METRIC_DIRECTION))

    def test_resumo_em_pt_br(self) -> None:
        """O resumo usa vírgula decimal."""
        assert summarize({"f1_macro": 0.7412}) == "f1_macro=0,7412"


class TestIncerteza:
    """Testes do intervalo de confiança por bootstrap."""

    def test_intervalo_contem_o_ponto(self) -> None:
        """O ponto estimado fica dentro do intervalo."""
        interval = bootstrap_confidence_interval(
            np.array([0, 1, 0, 1, 0, 1]), np.array([0, 1, 1, 1, 0, 0]), n_bootstrap=100
        )
        assert interval["lower"] <= interval["point"] <= interval["upper"]

    def test_intervalo_e_reprodutivel(self) -> None:
        """A mesma semente produz exatamente o mesmo intervalo."""
        args = (np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]))
        first = bootstrap_confidence_interval(*args, n_bootstrap=100, random_state=7)
        second = bootstrap_confidence_interval(*args, n_bootstrap=100, random_state=7)

        assert first == second

    def test_amostra_maior_estreita_o_intervalo(self) -> None:
        """Mais dados reduzem a incerteza — comportamento esperado do bootstrap."""
        rng = np.random.default_rng(42)
        pequeno = rng.integers(0, 2, 30)
        grande = rng.integers(0, 2, 600)

        margem_pequena = bootstrap_confidence_interval(pequeno, pequeno.copy(), n_bootstrap=200)[
            "margin"
        ]
        margem_grande = bootstrap_confidence_interval(grande, grande.copy(), n_bootstrap=200)[
            "margin"
        ]

        assert margem_grande <= margem_pequena

    def test_metrica_nao_suportada_e_rejeitada(self) -> None:
        """Só métricas registradas podem ser reamostradas."""
        with pytest.raises(KeyError, match="não suportada"):
            bootstrap_confidence_interval(np.array([0, 1]), np.array([0, 1]), metric="inexistente")

    def test_formatacao_com_intervalo(self) -> None:
        """A formatação segue o padrão pt-BR."""
        formatted = format_metric_with_ci({"point": 0.5, "lower": 0.4, "upper": 0.6}, precision=2)
        assert formatted == "0,50 [0,40; 0,60]"


class TestCalibracao:
    """Testes da avaliação de calibração."""

    def test_brier_perfeito_e_zero(self) -> None:
        """Probabilidade certa e confiante produz Brier zero."""
        assert compute_brier_score(np.array([0]), np.array([[1.0, 0.0]]), 2) == pytest.approx(0.0)

    def test_ece_nao_negativo(self, predictions: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """O erro de calibração esperado é não negativo por construção."""
        y_true, _, y_proba = predictions
        assert compute_expected_calibration_error(y_true, y_proba)["ece"] >= 0

    def test_avaliacao_completa(
        self, predictions: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """A avaliação devolve Brier, ECE e uma interpretação legível."""
        y_true, _, y_proba = predictions
        result = evaluate_calibration(y_true, y_proba)

        assert "brier_score" in result
        assert "interpretation" in result
        assert result["reliability_bins"]

    def test_soma_dos_bins_cobre_todas_as_amostras(
        self, predictions: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Cada amostra cai em exatamente um bin."""
        y_true, _, y_proba = predictions
        result = compute_expected_calibration_error(y_true, y_proba)

        assert sum(item["count"] for item in result["bins"]) == len(y_true)


class TestFatias:
    """Testes da avaliação por subgrupos."""

    def test_atribui_faixas(self) -> None:
        """Os valores são atribuídos às faixas corretas."""
        result = assign_slices(np.array([5, 50]), [0, 10, 100], ["baixo", "alto"])
        assert result.tolist() == ["baixo", "alto"]

    def test_fatia_pequena_e_ignorada(self) -> None:
        """Fatias abaixo do mínimo produziriam métricas puro ruído."""
        config = SlicesSection(definitions={}, min_samples_per_slice=10)
        result = evaluate_by_slice(np.array([0, 1]), np.array([0, 1]), np.array(["a", "a"]), config)
        assert "a" in result["skipped"]

    def test_detecta_disparidade(self) -> None:
        """Uma diferença grande entre fatias é sinalizada."""
        config = SlicesSection(definitions={}, min_samples_per_slice=2, max_acceptable_gap=0.1)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        slices = np.array(["boa", "boa", "ruim", "ruim"])

        result = evaluate_by_slice(y_true, y_pred, slices, config)
        assert result["exceeds_threshold"]

    def test_definicao_valida_bins_e_rotulos(self) -> None:
        """`bins` precisa ter um elemento a mais que `labels`."""
        with pytest.raises(ValueError, match="esperado"):
            SliceDefinition(column="n_tweets", bins=[0, 10], labels=["a", "b"])


class TestEstatistica:
    """Testes dos testes de significância entre modelos."""

    def test_cliffs_delta_maximo(self) -> None:
        """Separação completa produz delta 1."""
        assert cliffs_delta(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == pytest.approx(1.0)

    def test_cliffs_delta_nulo(self) -> None:
        """Amostras idênticas produzem delta zero."""
        assert cliffs_delta(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(0.0)

    def test_interpretacao_do_efeito(self) -> None:
        """As faixas convencionais da literatura são aplicadas."""
        assert interpret_effect_size(0.05) == "desprezível"
        assert interpret_effect_size(0.5) == "grande"

    def test_holm_e_monotonico(self) -> None:
        """A correção de Holm nunca reduz um p-valor."""
        raw = [0.01, 0.04, 0.03]
        corrected = holm_correction(raw)

        assert all(after >= before for before, after in zip(raw, corrected, strict=True))

    def test_holm_com_lista_vazia(self) -> None:
        """Sem comparações, não há o que corrigir."""
        assert holm_correction([]) == []

    def test_mcnemar_sem_discordancia(self) -> None:
        """Modelos idênticos não podem ser declarados diferentes."""
        y = np.array([0, 1, 1, 0])
        result = mcnemar_test(y, y.copy(), y.copy())

        assert not result.significant
        assert result.p_value == pytest.approx(1.0)

    def test_mcnemar_detecta_diferenca(self) -> None:
        """Um modelo claramente melhor é detectado no mesmo conjunto de teste."""
        y_true = np.array([0] * 40)
        melhor = np.array([0] * 40)
        pior = np.array([1] * 40)

        assert mcnemar_test(y_true, melhor, pior).significant

    def test_wilcoxon_exige_pareamento(self) -> None:
        """Amostras de tamanhos diferentes não são pareáveis."""
        with pytest.raises(ValueError, match="pareadas"):
            wilcoxon_test(np.array([0.1, 0.2]), np.array([0.1]))

    def test_wilcoxon_com_scores_identicos(self) -> None:
        """Desempenho idêntico em todos os folds não é diferença significativa."""
        scores = np.array([0.7, 0.72, 0.71])
        assert not wilcoxon_test(scores, scores.copy()).significant

    def test_friedman_exige_tres_modelos(self) -> None:
        """Com dois modelos, o teste correto é o de Wilcoxon."""
        with pytest.raises(ValueError, match="ao menos 3"):
            friedman_test({"a": np.array([0.7]), "b": np.array([0.6])})

    def test_friedman_reporta_ranking(self) -> None:
        """A interpretação traz o ranking médio, essencial para o pós-teste."""
        result = friedman_test(
            {
                "a": np.array([0.9, 0.9, 0.9, 0.9]),
                "b": np.array([0.7, 0.7, 0.7, 0.7]),
                "c": np.array([0.5, 0.5, 0.5, 0.5]),
            }
        )
        assert "Ranking médio" in result.interpretation

    def test_diferenca_critica_positiva(self) -> None:
        """A diferença crítica de Nemenyi é sempre positiva."""
        assert nemenyi_critical_difference(3, 5) > 0

    def test_comparacao_completa_aplica_correcao(self) -> None:
        """A bateria completa corrige para múltiplas comparações."""
        config = StatisticsSection(
            mcnemar=TestToggle(),
            wilcoxon=TestToggle(alternative="two-sided"),
            friedman=TestToggle(posthoc="nemenyi"),
        )
        scores = {
            "a": np.array([0.90, 0.91, 0.89, 0.92, 0.90]),
            "b": np.array([0.70, 0.71, 0.69, 0.72, 0.70]),
            "c": np.array([0.50, 0.51, 0.49, 0.52, 0.50]),
        }
        result = compare_all_models(scores, config)

        assert "friedman" in result
        assert len(result["pairwise"]) == 3
        assert all(
            entry["p_value_corrected"] >= entry["p_value"] for entry in result["pairwise"].values()
        )
