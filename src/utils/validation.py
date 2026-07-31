"""Verificações defensivas sobre DataFrames e arrays.

Complementam os contratos pandera (:mod:`schemas`): o pandera valida a
fronteira entre estágios do pipeline, estas funções validam pré-condições
locais de uma função. Ambas existem pelo mesmo motivo — falhar cedo, com
mensagem acionável, em vez de propagar ``KeyError`` ou ``NaN`` silencioso.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from config.logging import get_logger
from exceptions.data import EmptyDatasetError, InsufficientDataError

logger = get_logger(__name__)


def require_columns(frame: pl.DataFrame, columns: Sequence[str], context: str = "") -> None:
    """Garante que todas as colunas exigidas existam no DataFrame.

    Parameters
    ----------
    frame : pl.DataFrame
        DataFrame a verificar.
    columns : sequence of str
        Colunas obrigatórias.
    context : str, optional
        Descrição da etapa, incluída na mensagem de erro.

    Raises
    ------
    KeyError
        Se alguma coluna estiver ausente.

    Examples
    --------
    >>> require_columns(pl.DataFrame({"a": [1]}), ["a"])
    """
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        prefix = f"{context}: " if context else ""
        raise KeyError(f"{prefix}colunas ausentes {missing}. Disponíveis: {sorted(frame.columns)}")


def require_non_empty(frame: pl.DataFrame, context: str = "") -> None:
    """Garante que o DataFrame tenha ao menos uma linha.

    Parameters
    ----------
    frame : pl.DataFrame
        DataFrame a verificar.
    context : str, optional
        Descrição da etapa, incluída na mensagem de erro.

    Raises
    ------
    EmptyDatasetError
        Se o DataFrame estiver vazio.

    Examples
    --------
    >>> require_non_empty(pl.DataFrame({"a": [1]}))
    """
    if frame.height == 0:
        prefix = f"{context}: " if context else ""
        raise EmptyDatasetError(
            f"{prefix}DataFrame vazio. Verifique os filtros aplicados na etapa anterior."
        )


def require_min_rows(frame: pl.DataFrame, minimum: int, context: str = "") -> None:
    """Garante um número mínimo de linhas.

    Parameters
    ----------
    frame : pl.DataFrame
        DataFrame a verificar.
    minimum : int
        Número mínimo de linhas.
    context : str, optional
        Descrição da etapa.

    Raises
    ------
    InsufficientDataError
        Se houver menos linhas que o mínimo.

    Examples
    --------
    >>> require_min_rows(pl.DataFrame({"a": [1, 2]}), 2)
    """
    if frame.height < minimum:
        prefix = f"{context}: " if context else ""
        raise InsufficientDataError(
            f"{prefix}são necessárias ao menos {minimum} linhas, mas há {frame.height}."
        )


def check_class_balance(
    labels: Sequence[str],
    max_ratio: float,
    *,
    raise_on_violation: bool = False,
) -> dict[str, int]:
    """Verifica o balanceamento entre classes.

    Parameters
    ----------
    labels : sequence of str
        Rótulos observados.
    max_ratio : float
        Razão máxima aceitável entre a classe mais e a menos frequente.
    raise_on_violation : bool, optional
        Levanta exceção em vez de apenas avisar, by default False.

    Returns
    -------
    dict of str to int
        Contagem por classe.

    Raises
    ------
    ClassImbalanceError
        Se ``raise_on_violation`` e a razão exceder ``max_ratio``.

    Examples
    --------
    >>> check_class_balance(["a", "a", "b"], max_ratio=3.0)
    {'a': 2, 'b': 1}
    """
    from collections import Counter

    from exceptions.data import ClassImbalanceError

    counts = dict(Counter(labels))
    if not counts:
        return counts

    largest, smallest = max(counts.values()), min(counts.values())
    ratio = largest / smallest if smallest else float("inf")

    if ratio > max_ratio:
        message = (
            f"Desbalanceamento de {ratio:.2f}x entre classes (limite: {max_ratio:.2f}x). "
            f"Contagens: {counts}."
        )
        if raise_on_violation:
            raise ClassImbalanceError(message)
        logger.warning(message)

    return counts


def check_no_group_leakage(
    train_groups: Sequence[str],
    test_groups: Sequence[str],
) -> None:
    """Garante que nenhum usuário apareça em treino e em teste ao mesmo tempo.

    É a verificação de vazamento mais importante do projeto: com tweets do
    mesmo usuário nas duas partições, o modelo aprenderia a reconhecer a
    pessoa (estilo, vocabulário, temas recorrentes) em vez do sinal clínico,
    e a métrica de teste ficaria inflada de forma indetectável.

    Parameters
    ----------
    train_groups : sequence of str
        Identificadores de usuário no treino.
    test_groups : sequence of str
        Identificadores de usuário no teste.

    Raises
    ------
    ValueError
        Se houver interseção entre as partições.

    Examples
    --------
    >>> check_no_group_leakage(["u1", "u2"], ["u3"])
    """
    overlap = set(train_groups) & set(test_groups)
    if overlap:
        raise ValueError(
            f"Vazamento entre partições: {len(overlap)} usuário(s) presentes em treino e teste. "
            "Use particionamento agrupado por user_id."
        )


def check_finite(array: np.ndarray, name: str = "array") -> None:
    """Garante ausência de ``NaN`` e infinitos em um array numérico.

    Parameters
    ----------
    array : np.ndarray
        Array a verificar.
    name : str, optional
        Nome usado na mensagem de erro, by default ``"array"``.

    Raises
    ------
    ValueError
        Se houver valores não finitos.

    Examples
    --------
    >>> check_finite(np.array([1.0, 2.0]))
    """
    if not np.all(np.isfinite(array)):
        n_nan = int(np.isnan(array).sum())
        n_inf = int(np.isinf(array).sum())
        raise ValueError(
            f"'{name}' contém valores não finitos: {n_nan} NaN e {n_inf} infinitos. "
            "Verifique a imputação em features.aggregation.missing_strategy."
        )


def summarize_missing(frame: pl.DataFrame) -> pl.DataFrame:
    """Resume a taxa de valores ausentes por coluna.

    Parameters
    ----------
    frame : pl.DataFrame
        DataFrame a inspecionar.

    Returns
    -------
    pl.DataFrame
        Colunas ``column``, ``n_missing`` e ``missing_rate``, em ordem
        decrescente de ausência.

    Examples
    --------
    >>> resumo = summarize_missing(pl.DataFrame({"a": [1, None], "b": [1, 2]}))
    >>> resumo.row(0)[0]
    'a'
    """
    height = max(frame.height, 1)
    records = [
        {
            "column": column,
            "n_missing": frame[column].null_count(),
            "missing_rate": frame[column].null_count() / height,
        }
        for column in frame.columns
    ]
    return pl.DataFrame(records).sort("missing_rate", descending=True)
