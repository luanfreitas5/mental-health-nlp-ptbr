"""Gerenciamento de configuração, caminhos, logging e reprodutibilidade.

Toda configuração do projeto entra por este pacote e é validada com Pydantic
no *startup*. Nenhum outro módulo lê YAML ou variável de ambiente diretamente.

Modules
-------
paths
    :func:`get_paths` — todos os caminhos do projeto como ``pathlib.Path``.
settings
    :func:`load_config` — YAMLs de ``configs/`` validados com Pydantic.
logging
    :func:`configure_logging` e :func:`get_logger` — console ``rich`` + arquivo
    rotativo, com redação automática de PII.
environment
    :func:`seed_everything`, :func:`resolve_device` e :func:`describe_environment`.
version
    :func:`get_version`, :func:`get_git_sha` e :func:`build_run_id`.
"""

from config.environment import describe_environment, resolve_device, seed_everything
from config.logging import configure_logging, get_logger
from config.paths import ROOT, ProjectPaths, get_paths
from config.settings import Config, load_config
from config.version import build_run_id, get_git_sha, get_version

__all__ = [
    "ROOT",
    "Config",
    "ProjectPaths",
    "build_run_id",
    "configure_logging",
    "describe_environment",
    "get_git_sha",
    "get_logger",
    "get_paths",
    "get_version",
    "load_config",
    "resolve_device",
    "seed_everything",
]
