"""Figuras de interpretabilidade: SHAP, importância e Ablation Study."""

from __future__ import annotations

import operator
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.patches import Rectangle

from config.logging import get_logger
from constants.metrics import METRIC_DISPLAY_NAMES
from visualization.theme import CATEGORICAL_PALETTE, FIGURE_SIZES

logger = get_logger(__name__)

#: Cor de cada grupo de atributos, reutilizada em todas as figuras do tema.
GROUP_COLORS: dict[str, str] = {
    "linguistic": "#4C72B0",
    "emotional": "#DD8452",
    "semantic": "#55A868",
    "temporal": "#C44E52",
    "behavioral": "#8172B3",
    "psychological": "#937860",
    "outro": "#8C8C8C",
}


def plot_feature_importance(importance: pl.DataFrame, top_n: int = 25) -> Any | None:
    """Plota os atributos mais importantes, coloridos por grupo.

    Parameters
    ----------
    importance : pl.DataFrame
        Saída de
        :func:`interpretability.importance.compute_permutation_importance`.
    top_n : int, optional
        Número de atributos exibidos, by default 25.

    Returns
    -------
    matplotlib.figure.Figure or None
        Barras horizontais com barra de erro, ou ``None`` se não houver dados.

    Examples
    --------
    >>> plot_feature_importance(importancia)  # doctest: +SKIP
    """
    if importance.is_empty():
        return None

    data = importance.head(top_n).reverse().to_pandas()
    groups = data["grupo"] if "grupo" in data.columns else []
    colors = [GROUP_COLORS.get(group, "#8C8C8C") for group in groups]

    figure, axis = plt.subplots(figsize=(10, max(5, 0.32 * len(data))))
    axis.barh(data["atributo"], data["importancia"], color=colors or "#4C72B0", alpha=0.9)

    if "desvio" in data.columns:
        axis.errorbar(
            data["importancia"],
            range(len(data)),
            xerr=data["desvio"],
            fmt="none",
            ecolor="#333333",
            capsize=3,
            linewidth=1,
        )

    if "grupo" in data.columns:
        present = sorted(set(data["grupo"]))
        handles = [
            Rectangle((0, 0), 1, 1, color=GROUP_COLORS.get(group, "#8C8C8C")) for group in present
        ]
        axis.legend(handles, present, title="Grupo de atributos", loc="lower right")

    axis.set_title(f"Importância por Permutação — {top_n} Atributos Mais Relevantes")
    axis.set_xlabel("Queda na métrica ao permutar o atributo")
    axis.set_ylabel("")
    figure.tight_layout()
    return figure


def plot_group_importance(grouped: pl.DataFrame) -> Any | None:
    """Plota a importância agregada por grupo de atributos.

    Parameters
    ----------
    grouped : pl.DataFrame
        Saída de
        :func:`interpretability.importance.aggregate_importance_by_group`.

    Returns
    -------
    matplotlib.figure.Figure or None
        Barras com a participação percentual anotada.

    Examples
    --------
    >>> plot_group_importance(agrupado)  # doctest: +SKIP
    """
    if grouped.is_empty():
        return None

    data = grouped.to_pandas()
    colors = [GROUP_COLORS.get(group, "#8C8C8C") for group in data["grupo"]]

    figure, axis = plt.subplots(figsize=FIGURE_SIZES["medium"])
    bars = axis.bar(data["grupo"], data["importancia_total"], color=colors, alpha=0.9)

    if "participacao_pct" in data.columns:
        for bar, percentage, count in zip(
            bars, data["participacao_pct"], data["n_atributos"], strict=True
        ):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{percentage:.1f}%\n({count} atrib.)".replace(".", ","),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axis.set_title("Importância Total por Grupo de Atributos")
    axis.set_xlabel("Grupo")
    axis.set_ylabel("Importância acumulada")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    return figure


def plot_shap_summary(summary: pl.DataFrame) -> Any | None:
    """Plota o resumo dos valores SHAP por atributo.

    Parameters
    ----------
    summary : pl.DataFrame
        Saída de :func:`interpretability.shap_values.summarize_shap`.

    Returns
    -------
    matplotlib.figure.Figure or None
        Barras horizontais, ou ``None`` se não houver dados.

    Examples
    --------
    >>> plot_shap_summary(resumo_shap)  # doctest: +SKIP
    """
    if summary.is_empty():
        return None

    data = summary.reverse().to_pandas()
    groups = data["grupo"] if "grupo" in data.columns else []
    colors = [GROUP_COLORS.get(group, "#8C8C8C") for group in groups]

    figure, axis = plt.subplots(figsize=(10, max(5, 0.32 * len(data))))
    axis.barh(data["atributo"], data["shap_medio_absoluto"], color=colors or "#4C72B0", alpha=0.9)

    axis.set_title("Contribuição Média dos Atributos (SHAP)")
    axis.set_xlabel("|SHAP| médio")
    axis.set_ylabel("")
    figure.text(
        0.01,
        0.01,
        "SHAP explica o comportamento do modelo, não a causalidade do fenômeno.",
        fontsize=8,
        style="italic",
        color="#555555",
    )
    figure.tight_layout()
    return figure


def plot_ablation(ablation: pl.DataFrame, metric: str = "f1_macro") -> Any | None:
    """Plota a contribuição marginal de cada grupo no Ablation Study.

    Parameters
    ----------
    ablation : pl.DataFrame
        Saída de :func:`evaluation.ablation.summarize_ablation`.
    metric : str, optional
        Métrica principal, usada nos rótulos, by default ``f1_macro``.

    Returns
    -------
    matplotlib.figure.Figure or None
        Dois painéis: contribuição marginal e desempenho isolado por grupo.

    Examples
    --------
    >>> plot_ablation(resumo_ablacao)  # doctest: +SKIP
    """
    if ablation.is_empty():
        return None

    data = ablation.to_pandas()
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["wide"])

    # Painel 1 — contribuição marginal (leave-one-out).
    colors = [GROUP_COLORS.get(group, "#8C8C8C") for group in data["grupo"]]
    axes[0].barh(data["grupo"], data["contribuicao_marginal"], color=colors, alpha=0.9)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title("Contribuição marginal (leave-one-out)")
    axes[0].set_xlabel(f"Queda em {METRIC_DISPLAY_NAMES.get(metric, metric)} ao remover o grupo")
    axes[0].invert_yaxis()

    # Painel 2 — desempenho de cada grupo isolado.
    if "score_apenas_grupo" in data.columns and bool(data["score_apenas_grupo"].notna().any()):
        axes[1].barh(
            data["grupo"],
            data["score_apenas_grupo"].fillna(0),
            color=colors,
            alpha=0.9,
        )
        axes[1].set_title("Desempenho com o grupo isolado")
        axes[1].set_xlabel(METRIC_DISPLAY_NAMES.get(metric, metric))
        axes[1].set_xlim(0, 1)
        axes[1].invert_yaxis()
    else:
        axes[1].set_visible(False)

    figure.suptitle("Ablation Study — Contribuição dos Grupos de Atributos", fontweight="bold")
    figure.tight_layout()
    return figure


def plot_critical_difference(
    mean_ranks: dict[str, float],
    critical_difference: float,
) -> Any | None:
    """Plota o diagrama de diferença crítica (pós-teste de Nemenyi).

    Modelos cuja distância de ranking é menor que a diferença crítica não são
    estatisticamente distinguíveis — a leitura correta é "empate", e não
    "o primeiro é melhor".

    Parameters
    ----------
    mean_ranks : dict of str to float
        Ranking médio de cada modelo (1 = melhor).
    critical_difference : float
        Diferença crítica calculada.

    Returns
    -------
    matplotlib.figure.Figure or None
        Diagrama, ou ``None`` se houver menos de dois modelos.

    Examples
    --------
    >>> plot_critical_difference({"a": 1.2, "b": 2.4}, 0.9)  # doctest: +SKIP
    """
    if len(mean_ranks) < 2:
        return None

    ordered = sorted(mean_ranks.items(), key=operator.itemgetter(1))
    names = [name for name, _ in ordered]
    ranks = [rank for _, rank in ordered]

    figure, axis = plt.subplots(figsize=FIGURE_SIZES["medium"])
    axis.scatter(ranks, range(len(names)), s=90, color=CATEGORICAL_PALETTE[0], zorder=3)

    for position, (name, rank) in enumerate(ordered):
        axis.text(rank + 0.05, position, f"{name} ({rank:.2f})".replace(".", ","), va="center")
        axis.hlines(
            position,
            rank - critical_difference / 2,
            rank + critical_difference / 2,
            color="#C44E52",
            alpha=0.35,
            linewidth=6,
            zorder=2,
        )

    axis.set_yticks([])
    axis.set_xlabel("Ranking médio (1 = melhor)")
    axis.set_title(
        f"Diagrama de Diferença Crítica (Nemenyi, DC = {critical_difference:.3f})".replace(".", ",")
    )
    axis.set_xlim(0.5, max(ranks) + 1.2)
    figure.text(
        0.01,
        0.01,
        "Barras sobrepostas indicam modelos sem diferença estatisticamente significativa.",
        fontsize=8,
        style="italic",
        color="#555555",
    )
    figure.tight_layout()
    return figure
