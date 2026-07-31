"""Avaliação por fatias (subgrupos) do conjunto de teste.

A métrica agregada esconde falhas concentradas. Um F1-macro de 0,78 é
compatível com desempenho perfeito em usuários de histórico longo e
desempenho aleatório em usuários de histórico curto — que são justamente os
casos mais difíceis e, num cenário de triagem, potencialmente os mais
urgentes.

**Sobre fairness:** o projeto não coleta atributos demográficos (sexo, idade,
raça, região), por minimização de dados sob a LGPD. Uma auditoria de justiça
demográfica exigiria coletar exatamente a informação sensível que se decidiu
não coletar. A alternativa adotada — e a limitação — está documentada no
model card: as fatias são **comportamentais** (volume de publicação, janela
de observação, atividade noturna), que é o que os dados disponíveis permitem
avaliar honestamente.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
import polars as pl

from config.logging import get_logger
from config.settings import SlicesSection
from evaluation.metrics import compute_metrics

logger = get_logger(__name__)


def assign_slices(
    values: np.ndarray,
    bins: list[float],
    labels: list[str],
) -> np.ndarray:
    """Atribui cada observação a uma faixa.

    Parameters
    ----------
    values : np.ndarray
        Valores da coluna que define a fatia.
    bins : list of float
        Bordas das faixas (``len(labels) + 1`` elementos).
    labels : list of str
        Nome de cada faixa.

    Returns
    -------
    np.ndarray
        Nome da faixa de cada observação.

    Examples
    --------
    >>> assign_slices(np.array([5, 50]), [0, 10, 100], ["baixo", "alto"]).tolist()
    ['baixo', 'alto']
    """
    indices = np.digitize(values, bins[1:-1], right=True)
    return np.array([labels[min(index, len(labels) - 1)] for index in indices])


def evaluate_by_slice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    slice_values: np.ndarray,
    config: SlicesSection,
    metric: str = "f1_macro",
) -> dict[str, Any]:
    """Avalia o desempenho dentro de cada fatia.

    Parameters
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros.
    y_pred : np.ndarray
        Rótulos previstos.
    slice_values : np.ndarray
        Fatia de cada observação.
    config : SlicesSection
        Seção ``slices`` de ``configs/evaluation.yaml``.
    metric : str, optional
        Métrica usada na comparação entre fatias, by default ``f1_macro``.

    Returns
    -------
    dict
        Métricas por fatia, maior lacuna observada e alerta quando a lacuna
        excede ``max_acceptable_gap``.

    Examples
    --------
    >>> evaluate_by_slice(y_true, y_pred, fatias, config.evaluation.slices)  # doctest: +SKIP
    """
    results: dict[str, Any] = {"slices": {}, "skipped": {}}

    for name in sorted(set(slice_values.tolist())):
        mask = slice_values == name
        count = int(mask.sum())

        # Fatias pequenas produzem métricas com variância altíssima; reportá-las
        # ao lado das demais convidaria a conclusões sobre ruído.
        if count < config.min_samples_per_slice:
            results["skipped"][name] = {
                "n": count,
                "reason": f"abaixo do mínimo de {config.min_samples_per_slice} amostras",
            }
            continue

        metrics = compute_metrics(y_true[mask], y_pred[mask])
        results["slices"][name] = {"n": count, **metrics}

    scores = {
        name: values[metric] for name, values in results["slices"].items() if metric in values
    }
    if len(scores) < 2:
        results["gap"] = 0.0
        return results

    best_name, best_score = max(scores.items(), key=operator.itemgetter(1))
    worst_name, worst_score = min(scores.items(), key=operator.itemgetter(1))
    gap = best_score - worst_score

    results["gap"] = float(gap)
    results["best_slice"] = {"name": best_name, metric: best_score}
    results["worst_slice"] = {"name": worst_name, metric: worst_score}
    results["exceeds_threshold"] = bool(gap > config.max_acceptable_gap)

    if results["exceeds_threshold"]:
        logger.warning(
            "Disparidade entre fatias: %s=%.4f (melhor: %s) vs. %.4f (pior: %s); "
            "lacuna de %.4f acima do limite de %.4f.",
            metric,
            best_score,
            best_name,
            worst_score,
            worst_name,
            gap,
            config.max_acceptable_gap,
        )

    return results


def evaluate_all_slices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    profile: pl.DataFrame,
    config: SlicesSection,
    metric: str = "f1_macro",
) -> dict[str, Any]:
    """Avalia o modelo em todas as fatias declaradas na configuração.

    Parameters
    ----------
    y_true : np.ndarray
        Rótulos verdadeiros, na ordem das linhas de ``profile``.
    y_pred : np.ndarray
        Rótulos previstos, na mesma ordem.
    profile : pl.DataFrame
        Colunas descritivas do perfil dos usuários de teste.
    config : SlicesSection
        Seção ``slices`` de ``configs/evaluation.yaml``.
    metric : str, optional
        Métrica de comparação, by default ``f1_macro``.

    Returns
    -------
    dict
        Resultado por definição de fatia.

    Examples
    --------
    >>> evaluate_all_slices(y_true, y_pred, perfil, config.evaluation.slices)  # doctest: +SKIP
    """
    if not config.enabled:
        return {}

    results: dict[str, Any] = {}
    for name, definition in config.definitions.items():
        if definition.column not in profile.columns:
            logger.warning(
                "Fatia '%s' ignorada: a coluna '%s' não está na matriz de teste.",
                name,
                definition.column,
            )
            continue

        values = profile[definition.column].fill_null(0).to_numpy()
        assigned = assign_slices(values, definition.bins, definition.labels)
        results[name] = evaluate_by_slice(y_true, y_pred, assigned, config, metric)

    return results
