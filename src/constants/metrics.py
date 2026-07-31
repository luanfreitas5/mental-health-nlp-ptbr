"""Nomes canônicos das métricas e sua orientação de otimização.

:data:`METRIC_DIRECTION` evita o bug clássico de comparar modelos assumindo
que "maior é melhor" para toda métrica — o Brier score e o erro de calibração
são melhores quanto **menores**.
"""

from __future__ import annotations

from typing import Final

# --- Classificação ----------------------------------------------------------
ACCURACY: Final[str] = "accuracy"
PRECISION_MACRO: Final[str] = "precision_macro"
RECALL_MACRO: Final[str] = "recall_macro"
F1_MACRO: Final[str] = "f1_macro"
F1_WEIGHTED: Final[str] = "f1_weighted"
ROC_AUC_OVR: Final[str] = "roc_auc_ovr"
PR_AUC_MACRO: Final[str] = "pr_auc_macro"
MCC: Final[str] = "mcc"
BALANCED_ACCURACY: Final[str] = "balanced_accuracy"

# --- Calibração -------------------------------------------------------------
BRIER_SCORE: Final[str] = "brier_score"
EXPECTED_CALIBRATION_ERROR: Final[str] = "expected_calibration_error"

# --- Sufixos por classe -----------------------------------------------------
PER_CLASS_PRECISION: Final[str] = "precision"
PER_CLASS_RECALL: Final[str] = "recall"
PER_CLASS_F1: Final[str] = "f1"
PER_CLASS_SUPPORT: Final[str] = "support"
PER_CLASS_PR_AUC: Final[str] = "pr_auc"

#: Métricas calculadas por padrão para todo modelo.
DEFAULT_METRICS: Final[tuple[str, ...]] = (
    ACCURACY,
    PRECISION_MACRO,
    RECALL_MACRO,
    F1_MACRO,
    F1_WEIGHTED,
    ROC_AUC_OVR,
    PR_AUC_MACRO,
    MCC,
)

#: Orientação de cada métrica: ``1`` = maior é melhor, ``-1`` = menor é melhor.
METRIC_DIRECTION: Final[dict[str, int]] = {
    ACCURACY: 1,
    BALANCED_ACCURACY: 1,
    PRECISION_MACRO: 1,
    RECALL_MACRO: 1,
    F1_MACRO: 1,
    F1_WEIGHTED: 1,
    ROC_AUC_OVR: 1,
    PR_AUC_MACRO: 1,
    MCC: 1,
    BRIER_SCORE: -1,
    EXPECTED_CALIBRATION_ERROR: -1,
}

#: Métricas que exigem probabilidades (``predict_proba``), não apenas rótulos.
PROBABILITY_METRICS: Final[frozenset[str]] = frozenset(
    {ROC_AUC_OVR, PR_AUC_MACRO, BRIER_SCORE, EXPECTED_CALIBRATION_ERROR}
)

#: Nomes das métricas para exibição em tabelas e figuras (pt-BR).
METRIC_DISPLAY_NAMES: Final[dict[str, str]] = {
    ACCURACY: "Acurácia",
    BALANCED_ACCURACY: "Acurácia balanceada",
    PRECISION_MACRO: "Precisão (macro)",
    RECALL_MACRO: "Revocação (macro)",
    F1_MACRO: "F1 (macro)",
    F1_WEIGHTED: "F1 (ponderado)",
    ROC_AUC_OVR: "ROC-AUC (OvR)",
    PR_AUC_MACRO: "PR-AUC (macro)",
    MCC: "MCC",
    BRIER_SCORE: "Brier score",
    EXPECTED_CALIBRATION_ERROR: "Erro de calibração esperado (ECE)",
}

#: Mapa métrica do projeto -> string de *scoring* do scikit-learn.
SKLEARN_SCORING: Final[dict[str, str]] = {
    ACCURACY: "accuracy",
    BALANCED_ACCURACY: "balanced_accuracy",
    PRECISION_MACRO: "precision_macro",
    RECALL_MACRO: "recall_macro",
    F1_MACRO: "f1_macro",
    F1_WEIGHTED: "f1_weighted",
    ROC_AUC_OVR: "roc_auc_ovr",
    MCC: "matthews_corrcoef",
}


def is_higher_better(metric: str) -> bool:
    """Informa se a métrica deve ser maximizada.

    Parameters
    ----------
    metric : str
        Nome canônico da métrica.

    Returns
    -------
    bool
        ``True`` se maior é melhor, ``False`` se menor é melhor.

    Raises
    ------
    KeyError
        Se a métrica não estiver registrada em :data:`METRIC_DIRECTION`.

    Examples
    --------
    >>> is_higher_better("f1_macro")
    True
    >>> is_higher_better("brier_score")
    False
    """
    if metric not in METRIC_DIRECTION:
        raise KeyError(f"Métrica desconhecida: '{metric}'. Registradas: {sorted(METRIC_DIRECTION)}")
    return METRIC_DIRECTION[metric] > 0
