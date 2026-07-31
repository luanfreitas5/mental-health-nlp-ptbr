"""Leitura e escrita de arquivos auxiliares (termos, JSON, texto).

A escrita é **atômica**: o conteúdo vai para um arquivo temporário no mesmo
diretório e só então é renomeado sobre o destino. Uma interrupção no meio de
uma execução longa deixa o arquivo anterior intacto, em vez de um JSON
truncado que quebraria a etapa seguinte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config.logging import get_logger

logger = get_logger(__name__)


def read_terms_file(path: Path) -> list[str]:
    """Lê um arquivo de termos (léxico, palavras-chave ou hashtags).

    Um termo por linha. Uma linha é comentário quando começa com ``#``
    seguido de espaço, ou quando é apenas ``#`` — a exceção existe porque as
    hashtags (``#depressao``) também começam com ``#`` e são conteúdo, não
    comentário. Linhas vazias são ignoradas e duplicatas são removidas
    preservando a ordem original.

    Parameters
    ----------
    path : Path
        Caminho do arquivo ``.txt``.

    Returns
    -------
    list of str
        Termos, sem espaços nas extremidades.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.

    Examples
    --------
    >>> read_terms_file(Path("configs/lexicons/death.txt"))[:1]
    ['morte']
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"Arquivo de termos não encontrado: {target}. "
            "Confira os caminhos em configs/paths.yaml."
        )

    terms: list[str] = []
    seen: set[str] = set()
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "#" or line.startswith("# "):
            continue
        if line not in seen:
            seen.add(line)
            terms.append(line)

    logger.debug("Lidos %d termos de %s.", len(terms), target.name)
    return terms


def write_text(path: Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Escreve texto de forma atômica.

    Parameters
    ----------
    path : Path
        Caminho de destino.
    content : str
        Conteúdo a gravar.
    encoding : str, optional
        Codificação, by default ``"utf-8"``.

    Returns
    -------
    Path
        Caminho gravado.

    Examples
    --------
    >>> write_text(Path("reports/exemplo.md"), "# Título")  # doctest: +SKIP
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(content, encoding=encoding, newline="\n")
    temporary.replace(target)
    return target


class NumpyJSONEncoder(json.JSONEncoder):
    """Serializa tipos NumPy, que o ``json`` padrão não conhece.

    Sem isso, gravar métricas falha com ``TypeError: Object of type float32 is
    not JSON serializable`` — situação garantida, já que toda métrica vem do
    scikit-learn como escalar NumPy.
    """

    def default(self, o: Any) -> Any:
        """Converte tipos NumPy para equivalentes nativos do Python."""
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(path: Path, payload: Any, *, indent: int = 2) -> Path:
    """Grava um objeto em JSON de forma atômica, com suporte a tipos NumPy.

    Parameters
    ----------
    path : Path
        Caminho de destino.
    payload : Any
        Objeto serializável.
    indent : int, optional
        Indentação, by default 2.

    Returns
    -------
    Path
        Caminho gravado.

    Examples
    --------
    >>> write_json(Path("reports/metrics/exemplo.json"), {"f1": 0.8})  # doctest: +SKIP
    """
    content = json.dumps(payload, cls=NumpyJSONEncoder, ensure_ascii=False, indent=indent)
    return write_text(Path(path), content + "\n")


def read_json(path: Path) -> Any:
    """Lê um arquivo JSON.

    Parameters
    ----------
    path : Path
        Caminho do arquivo.

    Returns
    -------
    Any
        Conteúdo desserializado.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    json.JSONDecodeError
        Se o conteúdo não for JSON válido.

    Examples
    --------
    >>> read_json(Path("reports/metrics/exemplo.json"))  # doctest: +SKIP
    {'f1': 0.8}
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Arquivo JSON não encontrado: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def ensure_directory(path: Path) -> Path:
    """Garante a existência de um diretório.

    Parameters
    ----------
    path : Path
        Diretório a criar.

    Returns
    -------
    Path
        O próprio diretório.

    Examples
    --------
    >>> ensure_directory(Path("reports/figures")).is_dir()
    True
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def list_files(directory: Path, pattern: str = "*") -> list[Path]:
    """Lista arquivos de um diretório em ordem determinística.

    A ordenação importa: uma listagem em ordem do sistema de arquivos varia
    entre máquinas e quebraria a reprodutibilidade de qualquer etapa que
    concatene arquivos.

    Parameters
    ----------
    directory : Path
        Diretório a inspecionar.
    pattern : str, optional
        Padrão glob, by default ``"*"``.

    Returns
    -------
    list of Path
        Arquivos ordenados por nome (lista vazia se o diretório não existir).

    Examples
    --------
    >>> list_files(Path("configs/lexicons"), "*.txt")  # doctest: +SKIP
    [PosixPath('configs/lexicons/death.txt'), ...]
    """
    target = Path(directory)
    if not target.is_dir():
        logger.debug("Diretório inexistente ao listar arquivos: %s", target)
        return []
    return sorted(item for item in target.glob(pattern) if item.is_file())
