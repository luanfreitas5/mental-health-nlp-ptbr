"""Barras de progresso ``rich`` padronizadas.

Toda etapa longa do projeto (coleta, inferência de encoder, chamadas ao LLM,
treinamento) usa a mesma barra, com as colunas exigidas pelo CLAUDE.md. A
barra compartilha o :data:`config.logging.CONSOLE` com o logging: usar dois
consoles distintos faria as mensagens de log rasgarem a barra no terminal.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from config.logging import CONSOLE

T = TypeVar("T")


def build_progress(*, transient: bool = False) -> Progress:
    """Cria uma barra de progresso com as colunas padrão do projeto.

    Parameters
    ----------
    transient : bool, optional
        Remove a barra do terminal ao concluir, by default False (mantém o
        registro visual das etapas já executadas).

    Returns
    -------
    Progress
        Barra configurada com *spinner*, contagem, percentual e tempos.

    Examples
    --------
    >>> with build_progress() as progresso:
    ...     tarefa = progresso.add_task("Processando", total=3)
    ...     for _ in range(3):
    ...         progresso.advance(tarefa)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
        transient=transient,
    )


def track(
    items: Iterable[T],
    description: str,
    total: int | None = None,
    *,
    transient: bool = False,
) -> Iterator[T]:
    """Itera sobre uma coleção exibindo a barra de progresso padrão.

    Parameters
    ----------
    items : Iterable
        Coleção a percorrer.
    description : str
        Descrição exibida na barra (em pt-BR).
    total : int, optional
        Número total de itens; inferido de ``len(items)`` quando possível.
    transient : bool, optional
        Remove a barra ao concluir, by default False.

    Yields
    ------
    Any
        Cada item da coleção.

    Examples
    --------
    >>> list(track([1, 2, 3], "Somando"))
    [1, 2, 3]
    """
    if total is None:
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = None

    with build_progress(transient=transient) as progress:
        task = progress.add_task(description, total=total)
        for item in items:
            yield item
            progress.advance(task)
