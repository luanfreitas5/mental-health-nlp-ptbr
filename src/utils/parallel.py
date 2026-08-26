"""Paralelização por processos do laço de processamento por usuário.

As etapas 2 (``preprocess``) e 6 (``features``) processam um usuário por vez
e são CPU-bound (limpeza/normalização de texto, tokenização, agregações
polars) — cada usuário é independente dos demais, então o laço pode ser
distribuído entre processos em vez de rodar sequencialmente numa única
thread. Threads não ajudariam aqui: ao contrário do polars (que libera o GIL
na maior parte do trabalho pesado), boa parte deste código é Python puro
(regex, spaCy, laços de validação), que fica preso ao GIL.

Cada worker lê e grava a partição do próprio usuário diretamente em disco —
o resultado nunca cruza a fronteira entre processos, só um pequeno valor de
retorno (ou ``None``) por usuário.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import TypeVar

from config.logging import get_logger
from utils.progress import build_progress

logger = get_logger(__name__)

R = TypeVar("R")


def resolve_worker_count(requested: int | None) -> int:
    """Resolve quantos processos usar a partir da opção ``--workers``.

    Parameters
    ----------
    requested : int, optional
        Valor explícito de ``--workers``; ``None`` usa todos os núcleos
        detectados.

    Returns
    -------
    int
        Número de processos a usar (nunca menor que 1).

    Examples
    --------
    >>> resolve_worker_count(4)
    4
    >>> resolve_worker_count(0)
    1
    """
    if requested is not None:
        return max(1, requested)
    return max(1, os.cpu_count() or 1)


def run_user_pool(
    jobs: dict[str, Callable[[], R]],
    *,
    description: str,
    max_workers: int,
) -> list[R]:
    """Executa um job por usuário em processos paralelos.

    Cada job é um "thunk" (função sem argumentos, ex.: ``functools.partial``
    já com todos os dados daquele usuário vinculados) em vez de uma função
    compartilhada recebendo ``user_id`` — assim, cada worker recebe só a
    fatia de dados do próprio usuário (ex.: sua linha de metadata), nunca a
    coleção inteira dos usuários pendentes.

    Parameters
    ----------
    jobs : dict of str to Callable[[], R]
        Um job por usuário pendente, indexado por ``user_id``. Cada job
        precisa ser "picklable" — no Windows (método ``spawn``), isso exige
        que seja ``functools.partial`` de uma função de nível de módulo
        (nunca um método ligado, uma função aninhada ou uma lambda).
    description : str
        Texto exibido na barra de progresso.
    max_workers : int
        Número de processos paralelos (ver :func:`resolve_worker_count`).

    Returns
    -------
    list
        Um resultado por usuário processado com sucesso (jobs que retornam
        ``None`` são omitidos), na ordem de conclusão — a ordem não importa
        aqui, pois cada usuário já foi gravado em sua própria partição
        dentro do worker.

    Raises
    ------
    Exception
        Repropaga a primeira exceção levantada por um worker, após aguardar
        o encerramento dos processos já em execução — mesmo comportamento de
        interrupção imediata do laço sequencial original.

    Examples
    --------
    >>> run_user_pool({}, description="x", max_workers=1)
    []
    """
    if not jobs:
        return []

    logger.info(
        "%s: distribuindo %d usuário(s) entre %d processo(s).",
        description,
        len(jobs),
        max_workers,
    )

    results: list[R] = []
    with build_progress() as progress:
        task = progress.add_task(description, total=len(jobs))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(job): user_id for user_id, job in jobs.items()}
            for future in as_completed(futures):
                user_id = futures[future]
                try:
                    result = future.result()
                except Exception:
                    logger.exception("Falha ao processar o usuário '%s'.", user_id)
                    raise
                if result is not None:
                    results.append(result)
                progress.advance(task)

    return results


def run_thread_pool(
    jobs: dict[str, Callable[[], R]],
    *,
    description: str,
    max_workers: int,
    catch: tuple[type[Exception], ...] = (Exception,),
    show_progress: bool = True,
) -> dict[str, R]:
    """Executa jobs independentes em threads, isolando falhas por job.

    Complementa :func:`run_user_pool` para o caso de treino/validação/avaliação
    de modelos: scikit-learn, XGBoost e PyTorch liberam o GIL durante a parte
    pesada (código nativo em C/C++/CUDA), então threads bastam para sobrepor
    o trabalho — ao contrário do laço por usuário (Python puro, preso ao
    GIL), aqui processos só trariam a sobrecarga de serializar datasets
    grandes (atributos, textos, sequências de embeddings) entre processos e
    de abrir um contexto CUDA por processo na mesma GPU.

    Diferente de :func:`run_user_pool`, uma falha num job não interrompe os
    demais: o job correspondente fica ausente do dicionário retornado, e a
    exceção é registrada em log — o mesmo padrão de isolamento já usado nos
    laços sequenciais de treino, validação cruzada e avaliação que este
    utilitário substitui.

    Parameters
    ----------
    jobs : dict of str to Callable[[], R]
        Um job por chave (ex.: nome do modelo, índice do fold). Cada job é um
        "thunk" (``functools.partial`` ou closure já com os dados vinculados).
    description : str
        Texto exibido na barra de progresso.
    max_workers : int
        Número de threads paralelas (ver :func:`resolve_worker_count`).
    catch : tuple of type[Exception], optional
        Exceções tratadas como falha isolada de um job (logadas e omitidas do
        retorno); qualquer exceção fora dessa tupla propaga normalmente.
    show_progress : bool, optional
        Exibe uma barra de progresso própria, by default True. Use ``False``
        ao chamar esta função de dentro de um job que já roda sob outra
        barra de progresso (ex.: paralelismo aninhado) — a barra ``rich``
        compartilha um único console (:data:`config.logging.CONSOLE`) e duas
        barras ativas ao mesmo tempo levantam ``rich.errors.LiveError``.

    Returns
    -------
    dict of str to R
        Um resultado por job concluído com sucesso, indexado pela mesma
        chave de ``jobs``. Jobs que falharam (dentro de ``catch``) ficam
        ausentes.

    Examples
    --------
    >>> run_thread_pool({}, description="x", max_workers=1)
    {}
    """
    if not jobs:
        return {}

    logger.info(
        "%s: distribuindo %d tarefa(s) entre %d thread(s).",
        description,
        len(jobs),
        max_workers,
    )

    results: dict[str, R] = {}

    def _dispatch(advance: Callable[[], None]) -> None:
        """Submete todos os jobs e coleta os resultados, isolando falhas."""
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {executor.submit(job): key for key, job in jobs.items()}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except catch:
                    logger.exception("Falha ao executar '%s' (%s).", key, description)
                advance()

    if show_progress:
        with build_progress() as progress:
            task = progress.add_task(description, total=len(jobs))
            _dispatch(lambda: progress.advance(task))
    else:
        _dispatch(lambda: None)

    return results
