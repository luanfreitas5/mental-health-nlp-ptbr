"""Cálculo de métricas de classificação com quantificação de incerteza.

Regra do projeto: nenhuma métrica é reportada como número solto. Um F1 de
0,74 sobre 200 usuários de teste tem um intervalo de confiança largo o
bastante para ser indistinguível de 0,70 — e afirmar superioridade a partir
do ponto isolado seria uma conclusão que os dados não sustentam.

O intervalo é estimado por bootstrap percentílico sobre o conjunto de teste,
que não assume normalidade — premissa que não se sustenta para F1 e MCC,
métricas limitadas e assimétricas.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.logging import get_logger
from constants.labels import CLASS_ORDER
from constants.metrics import (
    ACCURACY,
    BALANCED_ACCURACY,
    F1_MACRO,
    F1_WEIGHTED,
    MCC,
    PR_AUC_MACRO,
    PRECISION_MACRO,
    RECALL_MACRO,
    ROC_AUC_OVR,
)

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    *,
    n_classes: int = len(CLASS_ORDER),
) -> dict[str, float]:
    """Calcula o conjunto padrão de métricas de classificação.

    Parameters
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros (índices inteiros).
    y_pred : np.ndarray
        Rótulos previstos.
    y_proba : np.ndarray, optional
        Probabilidades ``(n, n_classes)``; necessárias para ROC-AUC e PR-AUC.
    n_classes : int, optional
        Número de classes, by default 3.

    Returns
    -------
    dict of str to float
        Métricas agregadas. As que dependem de probabilidade ficam ausentes
        quando ``y_proba`` não é fornecido — melhor omitir do que reportar
        zero, que seria lido como desempenho ruim.

    Examples
    --------
    >>> metricas = compute_metrics(np.array([0, 1, 2]), np.array([0, 1, 1]))
    >>> round(metricas["accuracy"], 3)
    0.667
    """
    # `zero_division=0` é aceito em tempo de execução (float | "warn"), mas os
    # stubs do scikit-learn tipam o parâmetro só como `str` — falso positivo.
    metrics: dict[str, float] = {
        ACCURACY: float(accuracy_score(y_true, y_pred)),
        BALANCED_ACCURACY: float(balanced_accuracy_score(y_true, y_pred)),
        PRECISION_MACRO: float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)  # pyright: ignore[reportArgumentType]
        ),
        RECALL_MACRO: float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)  # pyright: ignore[reportArgumentType]
        ),
        F1_MACRO: float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)  # pyright: ignore[reportArgumentType]
        ),
        F1_WEIGHTED: float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)  # pyright: ignore[reportArgumentType]
        ),
        MCC: float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_proba is None:
        return metrics

    present = np.unique(y_true)
    if len(present) < 2:
        logger.warning("Apenas uma classe presente no conjunto: ROC-AUC e PR-AUC omitidos.")
        return metrics

    binarized = np.zeros((len(y_true), n_classes))
    binarized[np.arange(len(y_true)), y_true] = 1

    try:
        metrics[ROC_AUC_OVR] = float(
            roc_auc_score(binarized, y_proba, average="macro", multi_class="ovr")
        )
    except ValueError as error:
        logger.warning("ROC-AUC não pôde ser calculado: %s", error)

    try:
        metrics[PR_AUC_MACRO] = float(average_precision_score(binarized, y_proba, average="macro"))
    except ValueError as error:
        logger.warning("PR-AUC não pôde ser calculado: %s", error)

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    classes: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Calcula precisão, revocação, F1, suporte e PR-AUC por classe.

    A métrica agregada esconde o que mais importa aqui: um F1-macro razoável
    é perfeitamente compatível com revocação baixíssima em
    ``ideacao_suicida``, que é a classe de maior custo clínico associado a
    falso negativo.

    Parameters
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros.
    y_pred : np.ndarray
        Rótulos previstos.
    y_proba : np.ndarray, optional
        Probabilidades, usadas para o PR-AUC por classe.
    classes : list of str, optional
        Nomes das classes, by default :data:`constants.labels.CLASS_ORDER`.

    Returns
    -------
    dict
        ``{classe: {"precision", "recall", "f1", "support", "pr_auc"}}``.

    Examples
    --------
    >>> resultado = compute_per_class_metrics(np.array([0, 1]), np.array([0, 0]))
    >>> resultado["controle"]["recall"]
    1.0
    """
    names = classes or list(CLASS_ORDER)
    labels = list(range(len(names)))

    # Com `average=None` (o padrão), a função devolve um array por classe;
    # os stubs do scikit-learn não conseguem expressar essa sobrecarga e
    # inferem o escalar agregado da variante com `average` definido.
    precision, recall, f1, support = cast(
        "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]",
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
    )

    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        entry = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": float(support[index]),
        }
        if y_proba is not None and y_proba.shape[1] > index:
            binary_true = (y_true == index).astype(int)
            if binary_true.sum() > 0:
                entry["pr_auc"] = float(average_precision_score(binary_true, y_proba[:, index]))
        result[name] = entry

    return result


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = F1_MACRO,
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Estima o intervalo de confiança de uma métrica por bootstrap.

    Reamostra o conjunto de teste com reposição, preservando o pareamento
    entre verdade e predição — reamostrar os dois independentemente destruiria
    a correspondência e produziria um intervalo sem significado.

    Parameters
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros.
    y_pred : np.ndarray
        Rótulos previstos.
    metric : str, optional
        Métrica a estimar, by default ``f1_macro``.
    n_bootstrap : int, optional
        Número de reamostragens, by default 1000.
    confidence_level : float, optional
        Nível de confiança, by default 0.95.
    random_state : int, optional
        Semente, by default 42.

    Returns
    -------
    dict of str to float
        ``point``, ``lower``, ``upper``, ``std`` e ``margin``.

    Examples
    --------
    >>> intervalo = bootstrap_confidence_interval(
    ...     np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]), n_bootstrap=50
    ... )
    >>> intervalo["lower"] <= intervalo["point"] <= intervalo["upper"]
    True
    """
    # `zero_division=0` é aceito em tempo de execução, mas os stubs do
    # scikit-learn tipam o parâmetro só como `str` — falso positivo.
    scorers = {
        ACCURACY: accuracy_score,
        BALANCED_ACCURACY: balanced_accuracy_score,
        F1_MACRO: lambda a, b: f1_score(
            a,
            b,
            average="macro",
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
        F1_WEIGHTED: lambda a, b: f1_score(
            a,
            b,
            average="weighted",
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
        PRECISION_MACRO: lambda a, b: precision_score(
            a,
            b,
            average="macro",
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
        RECALL_MACRO: lambda a, b: recall_score(
            a,
            b,
            average="macro",
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
        MCC: matthews_corrcoef,
    }

    if metric not in scorers:
        raise KeyError(
            f"Métrica '{metric}' não suportada no bootstrap. Disponíveis: {sorted(scorers)}"
        )

    scorer = scorers[metric]
    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)

    scores = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, n_samples, size=n_samples)
        scores[index] = scorer(y_true[sample], y_pred[sample])

    alpha = (1.0 - confidence_level) / 2.0
    point = float(scorer(y_true, y_pred))
    lower = float(np.percentile(scores, 100 * alpha))
    upper = float(np.percentile(scores, 100 * (1 - alpha)))

    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "std": float(scores.std()),
        "margin": (upper - lower) / 2.0,
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = len(CLASS_ORDER),
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Calcula a matriz de confusão.

    Parameters
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros.
    y_pred : np.ndarray
        Rótulos previstos.
    n_classes : int, optional
        Número de classes, by default 3.
    normalize : bool, optional
        Normaliza por linha (revocação por classe), by default False.

    Returns
    -------
    np.ndarray
        Matriz ``(n_classes, n_classes)``.

    Examples
    --------
    >>> compute_confusion_matrix(np.array([0, 1]), np.array([0, 0]), n_classes=2).tolist()
    [[1, 0], [1, 0]]
    """
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    if not normalize:
        return matrix

    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums > 0)


def format_metric_with_ci(interval: dict[str, float], precision: int = 4) -> str:
    """Formata uma métrica com seu intervalo de confiança, em pt-BR.

    Parameters
    ----------
    interval : dict
        Saída de :func:`bootstrap_confidence_interval`.
    precision : int, optional
        Casas decimais, by default 4.

    Returns
    -------
    str
        Ex.: ``"0,7412 [0,7010; 0,7803]"``.

    Examples
    --------
    >>> format_metric_with_ci({"point": 0.5, "lower": 0.4, "upper": 0.6}, precision=2)
    '0,50 [0,40; 0,60]'
    """
    point = f"{interval['point']:.{precision}f}".replace(".", ",")
    lower = f"{interval['lower']:.{precision}f}".replace(".", ",")
    upper = f"{interval['upper']:.{precision}f}".replace(".", ",")
    return f"{point} [{lower}; {upper}]"


def summarize(metrics: dict[str, Any]) -> str:
    """Resume um dicionário de métricas numa linha legível.

    Parameters
    ----------
    metrics : dict
        Métricas calculadas.

    Returns
    -------
    str
        Resumo em pt-BR, com valores arredondados.

    Examples
    --------
    >>> summarize({"f1_macro": 0.7412, "accuracy": 0.8})
    'f1_macro=0,7412 | accuracy=0,8000'
    """
    parts = [
        f"{name}={value:.4f}".replace(".", ",")
        for name, value in metrics.items()
        if isinstance(value, int | float)
    ]
    return " | ".join(parts)
