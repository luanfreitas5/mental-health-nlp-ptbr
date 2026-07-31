"""Avaliação da calibração das probabilidades previstas.

Calibração importa porque as probabilidades do modelo têm uso operacional:
priorizar quem a equipe humana avalia primeiro. Um modelo que ordena bem
(ROC-AUC alto) mas afirma "90% de risco" em casos que só se confirmam 40% das
vezes induz decisões erradas de alocação — e a métrica de ordenação não revela
esse problema.

Duas medidas complementares:

* **Brier score** — erro quadrático médio das probabilidades. Combina
  calibração e poder discriminativo num único número.
* **ECE (Expected Calibration Error)** — discrepância média entre confiança
  declarada e acerto observado, ponderada por *bin*. Mede calibração
  isoladamente.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from config.logging import get_logger
from constants.labels import CLASS_ORDER

logger = get_logger(__name__)


def compute_brier_score(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    """Calcula o Brier score multiclasse.

    Parameters
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros (índices inteiros).
    y_proba : np.ndarray
        Probabilidades ``(n, n_classes)``.
    n_classes : int
        Número de classes.

    Returns
    -------
    float
        Brier score (quanto menor, melhor).

    Examples
    --------
    >>> round(compute_brier_score(np.array([0]), np.array([[1.0, 0.0]]), 2), 3)
    0.0
    """
    binarized = np.zeros((len(y_true), n_classes))
    binarized[np.arange(len(y_true)), y_true] = 1
    return float(np.mean(np.sum((y_proba - binarized) ** 2, axis=1)))


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Calcula o ECE e os dados da curva de confiabilidade.

    Parameters
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros.
    y_proba : np.ndarray
        Probabilidades previstas.
    n_bins : int, optional
        Número de faixas de confiança, by default 10.

    Returns
    -------
    dict
        ``ece``, ``mce`` (erro máximo) e ``bins`` com confiança média,
        acurácia observada e contagem por faixa.

    Examples
    --------
    >>> resultado = compute_expected_calibration_error(
    ...     np.array([0, 1]), np.array([[0.9, 0.1], [0.2, 0.8]])
    ... )
    >>> resultado["ece"] >= 0
    True
    """
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)

    ece = 0.0
    max_error = 0.0
    bins: list[dict[str, float]] = []

    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        # O primeiro bin inclui a borda inferior; os demais só a superior,
        # para que cada amostra caia em exatamente um bin.
        in_bin = (confidences > lower) & (confidences <= upper)
        if index == 0:
            in_bin |= confidences == lower

        count = int(in_bin.sum())
        if count == 0:
            bins.append(
                {
                    "bin_lower": float(lower),
                    "bin_upper": float(upper),
                    "confidence": 0.0,
                    "accuracy": 0.0,
                    "count": 0.0,
                }
            )
            continue

        bin_confidence = float(confidences[in_bin].mean())
        bin_accuracy = float(accuracies[in_bin].mean())
        gap = abs(bin_confidence - bin_accuracy)

        ece += (count / total) * gap
        max_error = max(max_error, gap)

        bins.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "confidence": bin_confidence,
                "accuracy": bin_accuracy,
                "count": float(count),
            }
        )

    return {"ece": ece, "mce": max_error, "bins": bins}


def evaluate_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
    n_classes: int = len(CLASS_ORDER),
) -> dict[str, Any]:
    """Avalia a calibração de um modelo.

    Parameters
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros.
    y_proba : np.ndarray
        Probabilidades previstas.
    n_bins : int, optional
        Número de faixas da curva de confiabilidade, by default 10.
    n_classes : int, optional
        Número de classes, by default 3.

    Returns
    -------
    dict
        ``brier_score``, ``expected_calibration_error``,
        ``maximum_calibration_error``, ``reliability_bins`` e uma
        interpretação em pt-BR.

    Examples
    --------
    >>> resultado = evaluate_calibration(
    ...     np.array([0, 1]), np.array([[0.9, 0.1], [0.3, 0.7]]), n_classes=2
    ... )
    >>> "brier_score" in resultado
    True
    """
    brier = compute_brier_score(y_true, y_proba, n_classes)
    calibration = compute_expected_calibration_error(y_true, y_proba, n_bins)

    ece = calibration["ece"]
    if ece < 0.05:
        verdict = "bem calibrado"
    elif ece < 0.15:
        verdict = "calibração moderada"
    else:
        verdict = "mal calibrado — considere calibração posterior (Platt ou isotônica)"

    mean_confidence = float(y_proba.max(axis=1).mean())
    mean_accuracy = float((y_proba.argmax(axis=1) == y_true).mean())
    bias = "otimista" if mean_confidence > mean_accuracy else "conservador"

    logger.info("Calibração: ECE=%.4f, Brier=%.4f (%s).", ece, brier, verdict)

    return {
        "brier_score": brier,
        "expected_calibration_error": ece,
        "maximum_calibration_error": calibration["mce"],
        "mean_confidence": mean_confidence,
        "mean_accuracy": mean_accuracy,
        "reliability_bins": calibration["bins"],
        "interpretation": (
            f"O modelo está {verdict} (ECE={ece:.4f}). A confiança média é "
            f"{mean_confidence:.3f} contra acurácia de {mean_accuracy:.3f}, "
            f"ou seja, o modelo é {bias}."
        ),
    }
