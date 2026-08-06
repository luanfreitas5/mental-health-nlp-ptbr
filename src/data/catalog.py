"""Catálogo dos artefatos de dados e manifesto de reprodutibilidade.

O catálogo responde a duas perguntas operacionais que, sem ele, exigiriam
inspecionar diretórios na mão: *quais etapas já rodaram?* e *os dados mudaram
desde a última execução?*

O manifesto (hash SHA-256 de cada artefato) é gravado junto do dataset
processado e registrado no MLflow. É o que fecha a tríade que torna um
experimento reproduzível: **código** (SHA do git) + **ambiente** (uv.lock) +
**dados** (este manifesto).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.logging import get_logger
from config.paths import ProjectPaths
from config.version import describe_version
from utils.files import read_json, write_json
from utils.hashing import build_manifest

logger = get_logger(__name__)


def _artifact_exists(path: Path) -> bool:
    """Verifica se um artefato existe, seja arquivo único ou diretório particionado."""
    if path.is_dir():
        return any(path.glob("*.parquet"))
    return path.is_file()


def _artifact_size_mb(path: Path) -> float:
    """Calcula o tamanho de um artefato em MB, somando os arquivos quando particionado."""
    if path.is_dir():
        total = sum(file.stat().st_size for file in path.glob("*.parquet"))
        return round(total / (1024 * 1024), 2)
    return round(path.stat().st_size / (1024 * 1024), 2)


@dataclass(frozen=True)
class ArtifactStatus:
    """Situação de um artefato de dados.

    Attributes
    ----------
    name : str
        Nome lógico do artefato.
    path : Path
        Caminho em disco.
    exists : bool
        Se o artefato já foi produzido.
    stage : str
        Etapa do pipeline responsável por criá-lo.
    size_mb : float
        Tamanho em megabytes (0 se ausente).
    """

    name: str
    path: Path
    exists: bool
    stage: str
    size_mb: float


def build_catalog(paths: ProjectPaths) -> dict[str, ArtifactStatus]:
    """Monta o catálogo dos artefatos esperados do pipeline.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos do projeto.

    Returns
    -------
    dict of str to ArtifactStatus
        Nome lógico -> situação do artefato.

    Examples
    --------
    >>> catalogo = build_catalog(get_paths())
    >>> "user_features" in catalogo
    True
    """
    declared: dict[str, tuple[Path, str]] = {
        "seed_tweets": (paths.data.seed_tweets / "seed_tweets.parquet", "collect"),
        "user_metadata": (paths.data.user_metadata, "collect"),
        "tweets_clean": (paths.data.tweets_clean, "preprocess"),
        "tweets_labeled": (paths.data.tweets_labeled, "label"),
        "psychological_scores": (paths.data.psychological_scores, "psych"),
        "user_labels": (paths.data.user_labels, "label"),
        "user_features": (paths.data.user_features, "features"),
        "splits": (paths.data.splits, "split"),
    }

    catalog: dict[str, ArtifactStatus] = {}
    for name, (path, stage) in declared.items():
        exists = _artifact_exists(path)
        catalog[name] = ArtifactStatus(
            name=name,
            path=path,
            exists=exists,
            stage=stage,
            size_mb=_artifact_size_mb(path) if exists else 0.0,
        )
    return catalog


def log_catalog(paths: ProjectPaths) -> None:
    """Registra a situação de cada artefato no log.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos do projeto.

    Examples
    --------
    >>> log_catalog(get_paths())
    """
    for status in build_catalog(paths).values():
        marker = "OK" if status.exists else "--"
        logger.info(
            "[%s] %-22s etapa=%-10s %6.2f MB",
            marker,
            status.name,
            status.stage,
            status.size_mb,
        )


def write_dataset_manifest(paths: ProjectPaths, extra: dict[str, Any] | None = None) -> Path:
    """Grava o manifesto do dataset com o hash de cada artefato.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos do projeto.
    extra : dict, optional
        Metadados adicionais (contagens, configuração relevante da execução).

    Returns
    -------
    Path
        Caminho do manifesto gravado.

    Examples
    --------
    >>> write_dataset_manifest(get_paths(), {"n_users": 900})  # doctest: +SKIP
    """
    catalog = build_catalog(paths)
    artifacts = {name: status.path for name, status in catalog.items() if status.exists}

    manifest: dict[str, Any] = {
        **describe_version(),
        "artifacts": build_manifest(artifacts),
    }
    if extra:
        manifest["metadata"] = extra

    target = write_json(paths.data.dataset_manifest, manifest)
    logger.info("Manifesto do dataset gravado em %s.", target)
    return target


def compare_manifest(paths: ProjectPaths) -> dict[str, str]:
    """Compara os artefatos atuais com o manifesto gravado.

    Detecta a falha mais traiçoeira da reprodutibilidade: os dados mudaram,
    mas o código e a configuração não — e a diferença de resultado seria
    atribuída, erroneamente, a uma mudança de método.

    Parameters
    ----------
    paths : ProjectPaths
        Caminhos do projeto.

    Returns
    -------
    dict of str to str
        Artefato -> ``"inalterado"``, ``"alterado"``, ``"novo"`` ou
        ``"removido"``. Vazio se ainda não houver manifesto.

    Examples
    --------
    >>> compare_manifest(get_paths())  # doctest: +SKIP
    {'user_features': 'inalterado'}
    """
    manifest_path = paths.data.dataset_manifest
    if not manifest_path.is_file():
        logger.info("Nenhum manifesto anterior encontrado em %s.", manifest_path)
        return {}

    previous = read_json(manifest_path).get("artifacts", {})
    catalog = build_catalog(paths)
    current = build_manifest(
        {name: status.path for name, status in catalog.items() if status.exists}
    )

    changes: dict[str, str] = {}
    for name in sorted(set(previous) | set(current)):
        old = previous.get(name, {}).get("sha256")
        new = current.get(name, {}).get("sha256")
        if old and new:
            changes[name] = "inalterado" if old == new else "alterado"
        elif new:
            changes[name] = "novo"
        else:
            changes[name] = "removido"

    modified = [name for name, state in changes.items() if state == "alterado"]
    if modified:
        logger.warning("Artefatos alterados desde o último manifesto: %s.", modified)

    return changes
