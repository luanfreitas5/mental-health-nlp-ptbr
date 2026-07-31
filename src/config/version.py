"""Versionamento do projeto e identificação da execução.

A versão vem de uma única fonte de verdade: ``[project].version`` do
``pyproject.toml``, gerenciado pelo ``commitizen`` (SemVer + Conventional
Commits). Duplicá-la em código levaria, inevitavelmente, a divergência.
"""

from __future__ import annotations

import subprocess  # nosec B404 — usado apenas para ler o SHA do git, sem entrada do usuário
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

try:  # Python >= 3.11 traz o parser de TOML na biblioteca padrão
    import tomllib
except ModuleNotFoundError:  # pragma: no cover — caminho exercido só no 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from config.paths import ROOT

#: Fallback caso o ``pyproject.toml`` não possa ser lido (ex.: wheel instalada).
FALLBACK_VERSION: str = "0.0.0"


@lru_cache(maxsize=1)
def get_version() -> str:
    """Lê a versão do projeto a partir do ``pyproject.toml``.

    Returns
    -------
    str
        Versão semântica (ex.: ``"0.1.0"``).

    Examples
    --------
    >>> get_version().count(".")
    2
    """
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return FALLBACK_VERSION

    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return FALLBACK_VERSION

    return str(data.get("project", {}).get("version", FALLBACK_VERSION))


@lru_cache(maxsize=1)
def get_git_sha(short: bool = True) -> str:
    """Retorna o SHA do commit atual, para rastrear qual código gerou um modelo.

    Parameters
    ----------
    short : bool, optional
        Retorna a forma abreviada (7 caracteres), by default True.

    Returns
    -------
    str
        SHA do commit, ou ``"desconhecido"`` fora de um repositório git.

    Examples
    --------
    >>> isinstance(get_git_sha(), str)
    True
    """
    command = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]

    try:
        result = subprocess.run(  # nosec B603 B607 — comando fixo, sem shell
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "desconhecido"

    return result.stdout.strip() or "desconhecido"


def build_run_id(prefix: str = "run") -> str:
    """Compõe um identificador único e legível para uma execução.

    Combina carimbo temporal e SHA do git: o carimbo ordena as execuções, o
    SHA identifica exatamente qual código as produziu.

    Parameters
    ----------
    prefix : str, optional
        Prefixo do identificador, by default ``"run"``.

    Returns
    -------
    str
        Identificador no formato ``<prefixo>_<AAAAMMDD-HHMMSS>_<sha>``.

    Examples
    --------
    >>> build_run_id("train").startswith("train_")
    True
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{timestamp}_{get_git_sha()}"


def describe_version() -> dict[str, str]:
    """Reúne os metadados de versão para o MLflow e o model card.

    Returns
    -------
    dict of str to str
        Versão do projeto, SHA do git e carimbo temporal UTC.

    Examples
    --------
    >>> sorted(describe_version())
    ['generated_at', 'git_sha', 'version']
    """
    return {
        "version": get_version(),
        "git_sha": get_git_sha(short=False),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def read_pyproject(path: Path | None = None) -> dict[str, object]:
    """Lê o ``pyproject.toml`` completo.

    Parameters
    ----------
    path : Path, optional
        Caminho alternativo, by default ``<raiz>/pyproject.toml``.

    Returns
    -------
    dict
        Conteúdo do arquivo (vazio se ausente ou inválido).

    Examples
    --------
    >>> "project" in read_pyproject()
    True
    """
    target = path or (ROOT / "pyproject.toml")
    if not target.is_file():
        return {}
    try:
        return tomllib.loads(target.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
