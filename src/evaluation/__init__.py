"""Avaliação rigorosa: métricas com incerteza, testes e análise por fatias.

Princípio do pacote: nenhuma métrica é reportada isolada, e nenhuma afirmação
de superioridade é feita sem teste de significância acompanhado de tamanho de
efeito.

Modules
-------
metrics
    Métricas de classificação, métricas por classe e intervalo de confiança
    por bootstrap.
calibration
    Brier score, ECE e curva de confiabilidade.
statistics
    McNemar, Wilcoxon, Friedman/Nemenyi, correção de Holm e delta de Cliff.
slices
    Avaliação por subgrupos comportamentais e detecção de disparidade.
ablation
    Ablation Study leave-one-out e only-one sobre os grupos de atributos.
evaluator
    :class:`Evaluator` — ponto único de avaliação, igual para todos os modelos.
reports
    Geração dos relatórios em JSON, CSV e Markdown.
"""

from evaluation.ablation import AblationResult, run_ablation, summarize_ablation
from evaluation.calibration import evaluate_calibration
from evaluation.evaluator import EvaluationResult, Evaluator
from evaluation.metrics import (
    bootstrap_confidence_interval,
    compute_confusion_matrix,
    compute_metrics,
    compute_per_class_metrics,
    format_metric_with_ci,
)
from evaluation.reports import build_markdown_report, save_reports
from evaluation.slices import evaluate_all_slices, evaluate_by_slice
from evaluation.statistics import (
    TestResult,
    cliffs_delta,
    compare_all_models,
    friedman_test,
    holm_correction,
    mcnemar_test,
    wilcoxon_test,
)

__all__ = [
    "AblationResult",
    "EvaluationResult",
    "Evaluator",
    "TestResult",
    "bootstrap_confidence_interval",
    "build_markdown_report",
    "cliffs_delta",
    "compare_all_models",
    "compute_confusion_matrix",
    "compute_metrics",
    "compute_per_class_metrics",
    "evaluate_all_slices",
    "evaluate_by_slice",
    "evaluate_calibration",
    "format_metric_with_ci",
    "friedman_test",
    "holm_correction",
    "mcnemar_test",
    "run_ablation",
    "save_reports",
    "summarize_ablation",
    "wilcoxon_test",
]
