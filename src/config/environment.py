"""Reprodutibilidade e detecção de ambiente.

``random_state`` sozinho não é reprodutibilidade: sem fixar ``PYTHONHASHSEED``,
a ordem de iteração de conjuntos muda entre processos; sem fixar as sementes
do PyTorch e desativar os kernels não determinísticos da cuDNN, dois
treinamentos idênticos divergem na terceira casa decimal.

Examples
--------
>>> from config.environment import seed_everything, resolve_device
>>> seed_everything(42)
>>> resolve_device("auto") in {"cpu", "cuda"}
True
"""

from __future__ import annotations

import os
import platform
import random
import sys
from typing import Any, Literal

import numpy as np

from config.logging import get_logger
from constants.defaults import RANDOM_SEED

logger = get_logger(__name__)


def seed_everything(seed: int = RANDOM_SEED, *, deterministic_torch: bool = True) -> None:
    """Fixa todas as fontes de aleatoriedade do processo.

    Parameters
    ----------
    seed : int, optional
        Semente global, by default :data:`constants.defaults.RANDOM_SEED`.
    deterministic_torch : bool, optional
        Ativa os algoritmos determinísticos do PyTorch, by default True.
        Custa desempenho, mas é o que torna o experimento replicável — e
        replicabilidade é requisito da dissertação, não otimização.

    Notes
    -----
    ``PYTHONHASHSEED`` só afeta processos iniciados **depois** da atribuição;
    para o processo atual, o efeito prático é sobre bibliotecas que leem a
    variável em tempo de execução. Para garantia total, exporte a variável
    antes de chamar o Python (o ``Makefile`` já faz isso no alvo ``train``).

    Examples
    --------
    >>> seed_everything(123)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    # Semeia o RNG global legado (não um Generator local): scikit-learn e
    # outras dependências chamam `np.random.*` internamente e só respeitam
    # essa semente global.
    np.random.seed(seed)  # noqa: NPY002

    try:
        import torch
    except ImportError:
        logger.debug("PyTorch ausente: sementes de deep learning não foram fixadas.")
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # `warn_only`: alguns kernels não têm implementação determinística;
        # preferimos um aviso a uma exceção que interromperia o treinamento.
        torch.use_deterministic_algorithms(True, warn_only=True)
    logger.debug("Sementes fixadas em %d (torch determinístico=%s).", seed, deterministic_torch)


def resolve_device(preference: Literal["auto", "cpu", "cuda"] = "auto") -> str:
    """Resolve o dispositivo de execução dos modelos neurais.

    Parameters
    ----------
    preference : {'auto', 'cpu', 'cuda'}, optional
        ``auto`` usa GPU se disponível, by default ``'auto'``.

    Returns
    -------
    str
        ``'cuda'`` ou ``'cpu'``.

    Examples
    --------
    >>> resolve_device("cpu")
    'cpu'
    """
    if preference == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        if preference == "cuda":
            logger.warning("CUDA solicitada, mas o PyTorch não está instalado. Usando CPU.")
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if preference == "cuda":
        logger.warning("CUDA solicitada, mas nenhuma GPU foi detectada. Usando CPU.")
    return "cpu"


def describe_environment() -> dict[str, Any]:
    """Coleta metadados do ambiente para registro no MLflow e no model card.

    Sem isso, um resultado não é reproduzível: a mesma semente em versões
    diferentes de biblioteca produz números diferentes.

    Returns
    -------
    dict
        Versões de Python, sistema operacional e bibliotecas relevantes.

    Examples
    --------
    >>> "python_version" in describe_environment()
    True
    """
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
    }

    for module_name, key in (
        ("polars", "polars_version"),
        ("pandas", "pandas_version"),
        ("sklearn", "sklearn_version"),
        ("xgboost", "xgboost_version"),
        ("torch", "torch_version"),
        ("transformers", "transformers_version"),
    ):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        info[key] = getattr(module, "__version__", "desconhecida")

    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda  # pyright: ignore[reportAttributeAccessIssue]
    except ImportError:
        info["cuda_available"] = False

    return info


def log_environment() -> None:
    """Registra o ambiente no log, em nível ``INFO``.

    Examples
    --------
    >>> log_environment()
    """
    for key, value in describe_environment().items():
        logger.info("Ambiente | %s: %s", key, value)
