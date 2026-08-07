"""Figuras de avaliação: confusão, ROC, precisão-revocação e calibração."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from config.logging import get_logger
from constants.labels import CLASS_ORDER
from constants.metrics import METRIC_DISPLAY_NAMES
from visualization.theme import (
    FIGURE_SIZES,
    SEQUENTIAL_PALETTE,
    get_class_labels,
    get_class_palette,
)

logger = get_logger(__name__)


def _prepare_confusion_display(
    data: np.ndarray, normalize: bool
) -> tuple[np.ndarray, np.ndarray, str]:
    """Prepara os valores exibidos e as anotações de texto da matriz de confusão."""
    if not normalize:
        return data, data.astype(int).astype(str), ""

    row_sums = data.sum(axis=1, keepdims=True)
    display = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums > 0)
    annotations = np.array(
        [
            [
                f"{display[i, j]:.2f}\n({int(data[i, j])})".replace(".", ",")
                for j in range(data.shape[1])
            ]
            for i in range(data.shape[0])
        ]
    )
    return display, annotations, ""


def plot_confusion_matrix(
    matrix: np.ndarray | list[list[float]],
    *,
    normalize: bool = True,
    title: str = "Matriz de Confusão",
) -> Any:
    """Plota a matriz de confusão.

    Normalizada por linha por padrão: as classes são desbalanceadas, e as
    contagens absolutas fariam a classe majoritária dominar visualmente,
    escondendo justamente o desempenho nas classes minoritárias — que são as
    de maior custo clínico.

    Parameters
    ----------
    matrix : np.ndarray or list of list of float
        Matriz de confusão com contagens.
    normalize : bool, optional
        Normaliza por linha (revocação por classe), by default True.
    title : str, optional
        Título da figura.

    Returns
    -------
    matplotlib.figure.Figure
        Figura da matriz.

    Examples
    --------
    >>> plot_confusion_matrix([[10, 2], [3, 8]])  # doctest: +SKIP
    """
    data = np.asarray(matrix, dtype=float)
    display, annotations, fmt = _prepare_confusion_display(data, normalize)
    labels = get_class_labels()[: data.shape[0]]

    figure, axis = plt.subplots(figsize=FIGURE_SIZES["square"])
    sns.heatmap(
        display,
        annot=annotations,
        fmt=fmt,
        cmap=SEQUENTIAL_PALETTE,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Proporção" if normalize else "Contagem"},
        vmin=0,
        vmax=1 if normalize else None,
        ax=axis,
    )

    axis.set_title(title)
    axis.set_xlabel("Classe prevista")
    axis.set_ylabel("Classe verdadeira")
    figure.tight_layout()
    return figure


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, model_name: str = "") -> Any:
    """Plota as curvas ROC um-contra-o-resto.

    Parameters
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros (índices inteiros).
    y_proba : np.ndarray
        Probabilidades ``(n, n_classes)``.
    model_name : str, optional
        Nome do modelo, incluído no título.

    Returns
    -------
    matplotlib.figure.Figure
        Figura com uma curva por classe.

    Examples
    --------
    >>> plot_roc_curves(y_true, y_proba, "hybrid_xgboost")  # doctest: +SKIP
    """
    figure, axis = plt.subplots(figsize=FIGURE_SIZES["square"])
    colors = get_class_palette()
    labels = get_class_labels()

    for index, class_name in enumerate(CLASS_ORDER[: y_proba.shape[1]]):
        binary_true = (y_true == index).astype(int)
        if binary_true.sum() == 0:
            logger.warning("Classe '%s' ausente no teste: curva ROC não plotada.", class_name)
            continue

        fpr, tpr, _ = roc_curve(binary_true, y_proba[:, index])
        axis.plot(
            fpr,
            tpr,
            color=colors[index],
            linewidth=2,
            label=f"{labels[index]} (AUC = {auc(fpr, tpr):.3f})".replace(".", ","),
        )

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Aleatório")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_title(f"Curvas ROC{f' — {model_name}' if model_name else ''}")
    axis.set_xlabel("Taxa de falsos positivos")
    axis.set_ylabel("Taxa de verdadeiros positivos")
    axis.legend(loc="lower right")
    figure.tight_layout()
    return figure


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "",
) -> Any:
    """Plota as curvas de precisão-revocação por classe.

    Mais informativa que a ROC neste projeto: com classes minoritárias, a
    taxa de falsos positivos da ROC fica artificialmente baixa porque o
    denominador (a classe negativa) é enorme, e a curva parece boa mesmo
    quando o desempenho na classe rara é fraco.

    Parameters
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros.
    y_proba : np.ndarray
        Probabilidades previstas.
    model_name : str, optional
        Nome do modelo.

    Returns
    -------
    matplotlib.figure.Figure
        Figura com uma curva por classe e as linhas de base.

    Examples
    --------
    >>> plot_precision_recall_curves(y_true, y_proba)  # doctest: +SKIP
    """
    figure, axis = plt.subplots(figsize=FIGURE_SIZES["square"])
    colors = get_class_palette()
    labels = get_class_labels()

    for index, class_name in enumerate(CLASS_ORDER[: y_proba.shape[1]]):
        binary_true = (y_true == index).astype(int)
        if binary_true.sum() == 0:
            logger.warning("Classe '%s' ausente no teste: curva PR não plotada.", class_name)
            continue

        precision, recall, _ = precision_recall_curve(binary_true, y_proba[:, index])
        axis.plot(
            recall,
            precision,
            color=colors[index],
            linewidth=2,
            label=f"{labels[index]} (PR-AUC = {auc(recall, precision):.3f})".replace(".", ","),
        )

        # Linha de base = prevalência da classe: é o que um classificador
        # aleatório atingiria, e sem ela a curva não tem referência.
        baseline = binary_true.mean()
        axis.axhline(baseline, color=colors[index], linestyle=":", linewidth=1, alpha=0.6)

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_title(f"Curvas Precisão-Revocação{f' — {model_name}' if model_name else ''}")
    axis.set_xlabel("Revocação")
    axis.set_ylabel("Precisão")
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def _filter_reliability_bins(calibration: dict[str, Any]) -> list[dict[str, Any]]:
    """Filtra os bins de confiabilidade com ao menos uma amostra."""
    return [item for item in calibration.get("reliability_bins", []) if item["count"] > 0]


def _plot_reliability_points(axis: Any, bins: list[dict[str, Any]]) -> None:
    """Desenha os pontos observados de confiabilidade, se houver bins com dados."""
    if not bins:
        return

    confidences = [item["confidence"] for item in bins]
    accuracies = [item["accuracy"] for item in bins]
    sizes = np.array([item["count"] for item in bins], dtype=float)

    axis.plot(confidences, accuracies, marker="o", color="#C44E52", linewidth=2, label="Observado")
    # O tamanho do marcador expõe quantas amostras sustentam cada ponto —
    # sem isso, um bin com 3 amostras parece tão confiável quanto um com 300.
    axis.scatter(
        confidences,
        accuracies,
        s=40 + 260 * sizes / sizes.max(),
        color="#C44E52",
        alpha=0.3,
    )


def plot_reliability_curve(calibration: dict[str, Any], model_name: str = "") -> Any:
    """Plota a curva de confiabilidade (calibração).

    Parameters
    ----------
    calibration : dict
        Saída de :func:`evaluation.calibration.evaluate_calibration`.
    model_name : str, optional
        Nome do modelo.

    Returns
    -------
    matplotlib.figure.Figure
        Curva observada contra a diagonal da calibração perfeita.

    Examples
    --------
    >>> plot_reliability_curve(resultado_calibracao)  # doctest: +SKIP
    """
    bins = _filter_reliability_bins(calibration)

    figure, axis = plt.subplots(figsize=FIGURE_SIZES["square"])
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibração perfeita")

    _plot_reliability_points(axis, bins)

    ece = calibration.get("expected_calibration_error", 0.0)
    axis.set_title(
        f"Curva de Confiabilidade{f' — {model_name}' if model_name else ''} "
        f"(ECE = {ece:.4f})".replace(".", ",")
    )
    axis.set_xlabel("Confiança média prevista")
    axis.set_ylabel("Acurácia observada")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="upper left")
    figure.tight_layout()
    return figure


def _add_confidence_error_bars(
    axis: Any, data: Any, metric: str, lower_column: str, upper_column: str
) -> None:
    """Adiciona barras de erro com o intervalo de confiança, se disponível na tabela."""
    if lower_column not in data.columns or upper_column not in data.columns:
        return

    errors = np.vstack(
        [
            (data[metric] - data[lower_column]).to_numpy(),
            (data[upper_column] - data[metric]).to_numpy(),
        ]
    )
    axis.errorbar(
        data[metric],
        range(len(data)),
        xerr=np.abs(errors),
        fmt="none",
        ecolor="#333333",
        capsize=4,
        linewidth=1.2,
    )


def _annotate_bar_values(axis: Any, values: Any) -> None:
    """Anota o valor numérico ao lado de cada barra horizontal."""
    for position, value in enumerate(values):
        if value is not None:
            axis.text(
                value + 0.01, position, f"{value:.4f}".replace(".", ","), va="center", fontsize=9
            )


def plot_model_comparison(comparison: pl.DataFrame, metric: str = "f1_macro") -> Any:
    """Plota a comparação entre modelos, com intervalo de confiança.

    Parameters
    ----------
    comparison : pl.DataFrame
        Saída de :meth:`evaluation.evaluator.Evaluator.compare`.
    metric : str, optional
        Métrica a comparar, by default ``f1_macro``.

    Returns
    -------
    matplotlib.figure.Figure
        Barras horizontais ordenadas, com barras de erro quando disponíveis.

    Raises
    ------
    KeyError
        Se a métrica não estiver na tabela.

    Examples
    --------
    >>> plot_model_comparison(comparacao)  # doctest: +SKIP
    """
    if metric not in comparison.columns:
        raise KeyError(f"Métrica '{metric}' ausente na tabela de comparação.")

    data = comparison.sort(metric, descending=False, nulls_last=False).to_pandas()

    figure, axis = plt.subplots(figsize=FIGURE_SIZES["medium"])
    axis.barh(data["modelo"], data[metric], color="#4C72B0", alpha=0.85)

    lower_column, upper_column = f"{metric}_ic_inferior", f"{metric}_ic_superior"
    _add_confidence_error_bars(axis, data, metric, lower_column, upper_column)
    _annotate_bar_values(axis, data[metric])

    axis.set_xlim(0, 1.05)
    axis.set_title(
        f"Comparação entre Modelos — {METRIC_DISPLAY_NAMES.get(metric, metric)} (IC 95%)"
    )
    axis.set_xlabel(METRIC_DISPLAY_NAMES.get(metric, metric))
    axis.set_ylabel("")
    figure.tight_layout()
    return figure


def _usable_slices(slices: dict[str, Any]) -> dict[str, Any]:
    """Filtra as definições de fatia que possuem dados de fato."""
    return {name: data for name, data in slices.items() if data.get("slices")}


def _plot_slice_panel(axis: Any, name: str, data: dict[str, Any], metric: str) -> None:
    """Plota o painel de desempenho de uma única definição de fatia."""
    entries = data["slices"]
    names = list(entries)
    values = [entries[key].get(metric, 0.0) for key in names]
    counts = [int(entries[key]["n"]) for key in names]

    axis.bar(names, values, color="#55A868", alpha=0.85)
    for position, (value, count) in enumerate(zip(values, counts, strict=True)):
        axis.text(
            position,
            value + 0.02,
            f"{value:.3f}\n(n={count})".replace(".", ","),
            ha="center",
            fontsize=8,
        )

    if data.get("exceeds_threshold"):
        axis.set_facecolor("#FFF5F5")

    axis.set_ylim(0, 1.15)
    axis.set_title(f"Fatia: {name}")
    axis.set_ylabel(METRIC_DISPLAY_NAMES.get(metric, metric))
    axis.tick_params(axis="x", rotation=20)


def plot_slice_performance(slices: dict[str, Any], metric: str = "f1_macro") -> Any | None:
    """Plota o desempenho por fatia, destacando disparidades.

    Parameters
    ----------
    slices : dict
        Saída de :func:`evaluation.slices.evaluate_all_slices`.
    metric : str, optional
        Métrica a comparar, by default ``f1_macro``.

    Returns
    -------
    matplotlib.figure.Figure or None
        Painel por definição de fatia, ou ``None`` se não houver dados.

    Examples
    --------
    >>> plot_slice_performance(resultado.slices)  # doctest: +SKIP
    """
    usable = _usable_slices(slices)
    if not usable:
        return None

    figure, axes = plt.subplots(1, len(usable), figsize=(6 * len(usable), 4.5), squeeze=False)

    for index, (name, data) in enumerate(usable.items()):
        _plot_slice_panel(axes[0][index], name, data, metric)

    figure.suptitle("Desempenho por Subgrupo Comportamental", fontweight="bold")
    figure.tight_layout()
    return figure
