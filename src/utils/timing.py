"""Medição de tempo de execução.

Usado para instrumentar as etapas do pipeline: a duração de cada estágio é
registrada no log e no MLflow, o que torna possível justificar qualquer
otimização com números de antes e depois, em vez de intuição.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar

from config.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def format_duration(seconds: float) -> str:
    """Formata uma duração em texto legível em pt-BR.

    Parameters
    ----------
    seconds : float
        Duração em segundos.

    Returns
    -------
    str
        Duração formatada (ex.: ``"1min 5,3s"``).

    Examples
    --------
    >>> format_duration(0.42)
    '0,42s'
    >>> format_duration(65.3)
    '1min 5,3s'
    >>> format_duration(3725)
    '1h 2min 5s'
    """
    if seconds < 1:
        return f"{seconds:.2f}s".replace(".", ",")
    if seconds < 60:
        return f"{seconds:.1f}s".replace(".", ",")
    if seconds < 3600:
        minutes, remainder = divmod(seconds, 60)
        return f"{int(minutes)}min {remainder:.1f}s".replace(".", ",")
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}min {int(secs)}s"


class Timer:
    """Cronômetro reutilizável.

    Attributes
    ----------
    elapsed : float
        Tempo decorrido em segundos (0 antes do primeiro uso).

    Examples
    --------
    >>> cronometro = Timer()
    >>> with cronometro:
    ...     _ = sum(range(1000))
    >>> cronometro.elapsed >= 0
    True
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float | None = None

    def __enter__(self) -> Timer:
        """Inicia a contagem."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Encerra a contagem e armazena o tempo decorrido."""
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start
            self._start = None

    @property
    def formatted(self) -> str:
        """Tempo decorrido em formato legível."""
        return format_duration(self.elapsed)


@contextmanager
def log_duration(description: str, level: int = 20) -> Iterator[Timer]:
    """Mede e registra a duração de um bloco de código.

    Parameters
    ----------
    description : str
        Descrição da operação, em pt-BR.
    level : int, optional
        Nível de log, by default ``logging.INFO`` (20).

    Yields
    ------
    Timer
        Cronômetro do bloco, consultável após a saída do contexto.

    Examples
    --------
    >>> with log_duration("Somando números"):
    ...     _ = sum(range(100))
    """
    timer = Timer()
    logger.log(level, "Iniciando: %s", description)
    with timer:
        yield timer
    logger.log(level, "Concluído: %s em %s.", description, timer.formatted)


def timed(func: F) -> F:
    """Decora uma função registrando sua duração no log.

    Parameters
    ----------
    func : Callable
        Função a instrumentar.

    Returns
    -------
    Callable
        Função equivalente, com registro de duração.

    Examples
    --------
    >>> @timed
    ... def calculate_sum(n: int) -> int:
    ...     return sum(range(n))
    >>> calculate_sum(10)
    45
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        timer = Timer()
        with timer:
            result = func(*args, **kwargs)
        logger.debug("%s executada em %s.", func.__qualname__, timer.formatted)
        return result

    return wrapper  # type: ignore[return-value]
