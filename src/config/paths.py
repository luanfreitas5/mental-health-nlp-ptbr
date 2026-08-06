"""Centraliza todos os caminhos do projeto a partir de ``configs/paths.yaml``.

Nenhum caminho é escrito diretamente no código: tudo passa por aqui e é
resolvido para ``pathlib.Path`` absoluto em relação à raiz do repositório.
Isso mantém o projeto funcional independentemente do diretório de trabalho
(execução via ``make``, via VSCode, via CI ou dentro do container).

Examples
--------
>>> from config.paths import get_paths
>>> paths = get_paths()
>>> paths.data.processed.name
'processed'
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from exceptions.configuration import ConfigFileNotFoundError, ConfigParsingError

# Raiz do repositório: src/config/paths.py -> src/config -> src -> <raiz>
ROOT: Path = Path(__file__).resolve().parents[2]

#: Caminho padrão do arquivo de caminhos.
DEFAULT_PATHS_FILE: Path = ROOT / "configs" / "paths.yaml"


def resolve_path(value: str | Path) -> Path:
    """Converte um caminho relativo do YAML em caminho absoluto sob a raiz.

    Parameters
    ----------
    value : str or Path
        Caminho relativo à raiz do repositório (ex.: ``"data/raw"``). Caminhos
        já absolutos são devolvidos inalterados.

    Returns
    -------
    Path
        Caminho absoluto correspondente.

    Examples
    --------
    >>> resolve_path("data/raw").is_absolute()
    True
    """
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


class _PathModel(BaseModel):
    """Modelo base que resolve automaticamente todos os campos de caminho."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataPaths(_PathModel):
    """Caminhos dos dados, organizados por estágio do pipeline."""

    root: Path
    raw: Path
    external: Path
    interim: Path
    processed: Path
    seed_tweets: Path
    user_histories: Path
    user_metadata: Path
    tweets_clean: Path
    tweets_labeled: Path
    psychological_scores: Path
    embeddings: Path
    user_features_raw: Path
    user_features: Path
    user_labels: Path
    splits: Path
    dataset_manifest: Path


class QueryPaths(_PathModel):
    """Caminhos dos arquivos ``.txt`` de palavras-chave e hashtags."""

    root: Path
    depression_keywords: Path
    depression_hashtags: Path
    suicidal_keywords: Path
    suicidal_hashtags: Path
    control_hashtags: Path


class LexiconPaths(_PathModel):
    """Caminhos dos léxicos psicolinguísticos."""

    root: Path
    death: Path
    loneliness: Path
    hopelessness: Path
    isolation: Path
    negative_emotion: Path
    insomnia: Path
    stopwords: Path


class ModelPaths(_PathModel):
    """Caminhos de modelos treinados, checkpoints e artefatos."""

    root: Path
    checkpoints: Path
    artifacts: Path
    registry: Path


class ReportPaths(_PathModel):
    """Caminhos dos relatórios, figuras e documentos de IA responsável."""

    root: Path
    figures: Path
    tables: Path
    metrics: Path
    statistics: Path
    ablation: Path
    interpretability: Path
    model_cards: Path
    datasheets: Path


class LogPaths(_PathModel):
    """Caminhos dos logs."""

    root: Path


class ConfigPaths(_PathModel):
    """Caminhos dos arquivos de configuração."""

    root: Path


class ProjectPaths(_PathModel):
    """Agregador de todos os caminhos do projeto.

    Attributes
    ----------
    root : Path
        Raiz do repositório.
    data, queries, lexicons, models, reports, logs, configs
        Agrupamentos temáticos de caminhos.
    """

    root: Path = ROOT
    data: DataPaths
    queries: QueryPaths
    lexicons: LexiconPaths
    models: ModelPaths
    reports: ReportPaths
    logs: LogPaths
    configs: ConfigPaths

    def iter_directories(self) -> list[Path]:
        """Lista os diretórios que o pipeline precisa ter criados.

        Arquivos (caminhos com sufixo, como ``.parquet``) são convertidos no
        diretório que os contém.

        Returns
        -------
        list of Path
            Diretórios únicos, sem duplicatas.
        """
        directories: list[Path] = []
        for group in (
            self.data,
            self.models,
            self.reports,
            self.logs,
            self.configs,
        ):
            for value in group.model_dump().values():
                path = Path(value)
                directories.append(path.parent if path.suffix else path)
        return sorted(set(directories))

    def ensure_directories(self) -> None:
        """Cria todos os diretórios do projeto, se ainda não existirem.

        Examples
        --------
        >>> get_paths().ensure_directories()
        """
        for directory in self.iter_directories():
            directory.mkdir(parents=True, exist_ok=True)


def _resolve_section(section: dict[str, Any]) -> dict[str, Path]:
    """Resolve todos os valores de uma seção do YAML para caminhos absolutos."""
    return {key: resolve_path(value) for key, value in section.items()}


@lru_cache(maxsize=1)
def get_paths(paths_file: Path | None = None) -> ProjectPaths:
    """Carrega e valida ``configs/paths.yaml``.

    O resultado é memoizado: o arquivo é lido uma única vez por processo.

    Parameters
    ----------
    paths_file : Path, optional
        Arquivo alternativo de caminhos, by default ``configs/paths.yaml``.

    Returns
    -------
    ProjectPaths
        Estrutura validada com todos os caminhos absolutos.

    Raises
    ------
    ConfigFileNotFoundError
        Se o arquivo de caminhos não existir.
    ConfigParsingError
        Se o YAML for inválido ou não seguir o schema esperado.

    Examples
    --------
    >>> paths = get_paths()
    >>> paths.lexicons.death.suffix
    '.txt'
    """
    target = Path(paths_file) if paths_file else DEFAULT_PATHS_FILE
    if not target.is_file():
        raise ConfigFileNotFoundError(
            f"Arquivo de caminhos não encontrado: {target}. "
            "Verifique se 'configs/paths.yaml' existe na raiz do projeto."
        )

    try:
        raw: dict[str, Any] = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise ConfigParsingError(f"YAML inválido em {target}: {error}") from error

    try:
        return ProjectPaths(
            root=ROOT,
            data=DataPaths(**_resolve_section(raw["data"])),
            queries=QueryPaths(**_resolve_section(raw["queries"])),
            lexicons=LexiconPaths(**_resolve_section(raw["lexicons"])),
            models=ModelPaths(**_resolve_section(raw["models"])),
            reports=ReportPaths(**_resolve_section(raw["reports"])),
            logs=LogPaths(**_resolve_section(raw["logs"])),
            configs=ConfigPaths(**_resolve_section(raw["configs"])),
        )
    except KeyError as error:
        raise ConfigParsingError(
            f"Seção obrigatória ausente em {target}: {error}. "
            "Compare com o modelo em configs/paths.yaml."
        ) from error
    except (TypeError, ValueError) as error:
        raise ConfigParsingError(f"Estrutura inválida em {target}: {error}") from error
