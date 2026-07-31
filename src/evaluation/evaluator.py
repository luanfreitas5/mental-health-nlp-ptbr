"""Avaliador principal: reúne métricas, incerteza, calibração e fatias.

Um único ponto de entrada por modelo, para que todos recebam exatamente o
mesmo tratamento. Qualquer diferença de procedimento entre modelos
invalidaria a comparação — e diferenças assim aparecem naturalmente quando
cada família tem seu próprio caminho de avaliação.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from config.logging import get_logger
from config.settings import Config
from constants.labels import CLASS_ORDER
from evaluation.calibration import evaluate_calibration
from evaluation.metrics import (
    bootstrap_confidence_interval,
    compute_confusion_matrix,
    compute_metrics,
    compute_per_class_metrics,
    format_metric_with_ci,
)
from evaluation.slices import evaluate_all_slices
from models.base import BaseUserClassifier, UserDataset

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Resultado completo da avaliação de um modelo.

    Attributes
    ----------
    model_name : str
        Nome do modelo avaliado.
    metrics : dict
        Métricas agregadas.
    per_class : dict
        Métricas por classe.
    confidence_interval : dict
        Intervalo de confiança da métrica principal.
    confusion_matrix : list
        Matriz de confusão (como lista, para serialização).
    calibration : dict
        Resultado da avaliação de calibração.
    slices : dict
        Desempenho por fatia.
    predictions : dict
        Predições por usuário, para análise de erro e testes pareados.
    """

    model_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confidence_interval: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[float]] = field(default_factory=list)
    calibration: dict[str, Any] = field(default_factory=dict)
    slices: dict[str, Any] = field(default_factory=dict)
    predictions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializa o resultado para JSON.

        Returns
        -------
        dict
            Resultado completo, pronto para gravação.

        Examples
        --------
        >>> resultado.to_dict()["model_name"]  # doctest: +SKIP
        'hybrid_xgboost'
        """
        return {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "per_class": self.per_class,
            "confidence_interval": self.confidence_interval,
            "confusion_matrix": self.confusion_matrix,
            "calibration": self.calibration,
            "slices": self.slices,
        }

    def headline(self, metric: str) -> str:
        """Resume o resultado numa linha, com intervalo de confiança.

        Parameters
        ----------
        metric : str
            Métrica principal.

        Returns
        -------
        str
            Ex.: ``"hybrid_xgboost | f1_macro = 0,7612 [0,7210; 0,7990]"``.

        Examples
        --------
        >>> resultado.headline("f1_macro")  # doctest: +SKIP
        'hybrid_xgboost | f1_macro = 0,7612 [0,7210; 0,7990]'
        """
        if self.confidence_interval:
            return (
                f"{self.model_name} | {metric} = {format_metric_with_ci(self.confidence_interval)}"
            )
        value = self.metrics.get(metric, float("nan"))
        return f"{self.model_name} | {metric} = {value:.4f}".replace(".", ",")


class Evaluator:
    """Avaliador padronizado de modelos no nível do usuário.

    Parameters
    ----------
    config : Config
        Configuração completa do projeto.

    Examples
    --------
    >>> avaliador = Evaluator(config)
    >>> resultado = avaliador.evaluate(modelo, teste)  # doctest: +SKIP
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.classes = list(CLASS_ORDER)

    def evaluate(
        self,
        model: BaseUserClassifier,
        dataset: UserDataset,
        profile: pl.DataFrame | None = None,
    ) -> EvaluationResult:
        """Avalia um modelo no conjunto de teste.

        Parameters
        ----------
        model : BaseUserClassifier
            Modelo já treinado.
        dataset : UserDataset
            Conjunto de teste, com rótulos.
        profile : pl.DataFrame, optional
            Colunas descritivas dos usuários de teste, para a avaliação por
            fatias.

        Returns
        -------
        EvaluationResult
            Resultado completo.

        Raises
        ------
        ValueError
            Se o conjunto não tiver rótulos.

        Examples
        --------
        >>> avaliador.evaluate(modelo, teste, perfil)  # doctest: +SKIP
        """
        if not dataset.has_labels:
            raise ValueError(
                f"Não é possível avaliar '{model.name}' sem rótulos no conjunto de teste."
            )

        y_true = np.asarray(dataset.labels)
        y_proba = model.predict_proba(dataset)
        y_pred = y_proba.argmax(axis=1)

        settings = self.config.evaluation
        primary = settings.metrics.primary

        result = EvaluationResult(model_name=model.name)
        result.metrics = compute_metrics(y_true, y_pred, y_proba, n_classes=len(self.classes))
        result.per_class = compute_per_class_metrics(y_true, y_pred, y_proba, self.classes)
        result.confusion_matrix = compute_confusion_matrix(
            y_true, y_pred, len(self.classes)
        ).tolist()

        result.confidence_interval = bootstrap_confidence_interval(
            y_true,
            y_pred,
            metric=primary,
            n_bootstrap=settings.uncertainty.n_bootstrap,
            confidence_level=settings.uncertainty.confidence_level,
            random_state=settings.uncertainty.random_state,
        )

        if settings.calibration.enabled:
            result.calibration = evaluate_calibration(
                y_true,
                y_proba,
                n_bins=settings.calibration.n_bins,
                n_classes=len(self.classes),
            )

        if settings.slices.enabled and profile is not None:
            result.slices = evaluate_all_slices(
                y_true, y_pred, profile, settings.slices, metric=primary
            )

        result.predictions = {
            "user_ids": dataset.user_ids,
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "y_proba": y_proba.tolist(),
        }

        logger.info(result.headline(primary))
        self._log_highlights(result)
        return result

    def _log_highlights(self, result: EvaluationResult) -> None:
        """Registra as métricas destacadas em ``evaluation.metrics.highlight``."""
        for highlight in self.config.evaluation.metrics.highlight:
            for metric_name in ("recall", "pr_auc", "precision", "f1"):
                prefix = f"{metric_name}_"
                if not highlight.startswith(prefix):
                    continue
                class_name = highlight.removeprefix(prefix)
                value = result.per_class.get(class_name, {}).get(metric_name)
                if value is not None:
                    logger.info("Destaque | %s (%s) = %.4f", metric_name, class_name, value)
                break

    def compare(
        self,
        results: dict[str, EvaluationResult],
    ) -> pl.DataFrame:
        """Monta a tabela comparativa entre modelos.

        Parameters
        ----------
        results : dict of str to EvaluationResult
            Resultados por modelo.

        Returns
        -------
        pl.DataFrame
            Uma linha por modelo, ordenada pela métrica principal.

        Examples
        --------
        >>> avaliador.compare(resultados)  # doctest: +SKIP
        """
        primary = self.config.evaluation.metrics.primary
        records: list[dict[str, Any]] = []

        for name, result in results.items():
            record: dict[str, Any] = {"modelo": name}
            record.update(
                {
                    metric: result.metrics.get(metric)
                    for metric in self.config.evaluation.metrics.compute
                    if metric in result.metrics
                }
            )
            if result.confidence_interval:
                record[f"{primary}_ic_inferior"] = result.confidence_interval["lower"]
                record[f"{primary}_ic_superior"] = result.confidence_interval["upper"]
            if result.calibration:
                record["brier_score"] = result.calibration.get("brier_score")
                record["ece"] = result.calibration.get("expected_calibration_error")

            for class_name in self.classes:
                per_class = result.per_class.get(class_name, {})
                record[f"recall_{class_name}"] = per_class.get("recall")

            records.append(record)

        if not records:
            return pl.DataFrame()

        return pl.DataFrame(records).sort(primary, descending=True, nulls_last=True)
