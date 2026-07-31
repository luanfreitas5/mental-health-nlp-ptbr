"""Persistência de modelos treinados, com metadados de rastreabilidade.

Um arquivo ``.joblib`` sozinho não é reproduzível: seis meses depois não há
como saber com qual versão do código, com quais dados ou com qual
configuração ele foi produzido. Por isso todo modelo é salvo junto de um
JSON com versão do projeto, SHA do git, hiperparâmetros e ambiente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from config.environment import describe_environment
from config.logging import get_logger
from config.version import describe_version
from exceptions.model import ModelPersistenceError
from models.base import BaseUserClassifier
from utils.files import read_json, write_json

logger = get_logger(__name__)


def save_model(
    model: BaseUserClassifier,
    directory: Path,
    *,
    metrics: dict[str, float] | None = None,
    dataset_hash: str | None = None,
) -> Path:
    """Persiste um modelo treinado e seus metadados.

    Parameters
    ----------
    model : BaseUserClassifier
        Modelo já treinado.
    directory : Path
        Diretório de destino.
    metrics : dict, optional
        Métricas de teste, gravadas junto dos metadados.
    dataset_hash : str, optional
        Hash do dataset usado no treino — fecha a tríade
        código + ambiente + dados.

    Returns
    -------
    Path
        Caminho do arquivo ``.joblib`` gravado.

    Raises
    ------
    ModelPersistenceError
        Se o modelo não estiver treinado ou a gravação falhar.

    Examples
    --------
    >>> save_model(modelo, Path("models/artifacts"))  # doctest: +SKIP
    """
    if not model.is_fitted:
        raise ModelPersistenceError(
            f"O modelo '{model.name}' não foi treinado: não há o que salvar."
        )

    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / f"{model.name}.joblib"

    try:
        joblib.dump(model, model_path)
    except (OSError, TypeError) as error:
        raise ModelPersistenceError(
            f"Falha ao salvar o modelo '{model.name}' em {model_path}: {error}"
        ) from error

    metadata: dict[str, Any] = {
        **describe_version(),
        "model": model.describe(),
        "environment": describe_environment(),
    }
    if metrics:
        metadata["metrics"] = metrics
    if dataset_hash:
        metadata["dataset_hash"] = dataset_hash

    write_json(target_dir / f"{model.name}_metadata.json", metadata)
    logger.info("Modelo '%s' salvo em %s.", model.name, model_path)
    return model_path


def load_model(path: Path) -> BaseUserClassifier:
    """Carrega um modelo persistido.

    Parameters
    ----------
    path : Path
        Caminho do arquivo ``.joblib``.

    Returns
    -------
    BaseUserClassifier
        Modelo pronto para inferência.

    Raises
    ------
    ModelPersistenceError
        Se o arquivo não existir ou não puder ser desserializado.

    Examples
    --------
    >>> load_model(Path("models/artifacts/hybrid_xgboost.joblib"))  # doctest: +SKIP
    """
    target = Path(path)
    if not target.is_file():
        raise ModelPersistenceError(f"Modelo não encontrado: {target}")

    try:
        model = joblib.load(target)
    except (OSError, ValueError, AttributeError) as error:
        raise ModelPersistenceError(
            f"Falha ao carregar {target}: {error}. O modelo pode ter sido salvo com uma "
            "versão incompatível das bibliotecas — confira o metadata.json ao lado."
        ) from error

    logger.info("Modelo carregado de %s.", target)
    return model


def load_metadata(path: Path) -> dict[str, Any]:
    """Carrega os metadados de um modelo persistido.

    Parameters
    ----------
    path : Path
        Caminho do ``.joblib`` ou do ``_metadata.json``.

    Returns
    -------
    dict
        Metadados (dicionário vazio se ausentes).

    Examples
    --------
    >>> load_metadata(Path("models/artifacts/xgboost.joblib"))  # doctest: +SKIP
    {'version': '0.1.0', ...}
    """
    target = Path(path)
    if target.suffix == ".joblib":
        target = target.with_name(f"{target.stem}_metadata.json")

    if not target.is_file():
        logger.warning("Metadados não encontrados em %s.", target)
        return {}
    return read_json(target)
