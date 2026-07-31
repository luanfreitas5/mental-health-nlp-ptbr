"""Leitura de artefatos de dados, com mensagens de erro acionáveis.

Quando um artefato não existe, o erro diz **qual etapa** deveria tê-lo
produzido. Num pipeline de dez estágios, "arquivo não encontrado" sozinho
obriga a reconstruir mentalmente o grafo de dependências toda vez.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from config.logging import get_logger
from exceptions.data import DatasetNotFoundError
from utils.files import list_files

logger = get_logger(__name__)

#: Artefato -> etapa que o produz (usado nas mensagens de erro).
_PRODUCER_STAGE: dict[str, str] = {
    "tweets_clean.parquet": "preprocess",
    "tweets_labeled.parquet": "label",
    "psychological_scores.parquet": "psych",
    "user_features.parquet": "features",
    "user_labels.parquet": "label",
    "splits.parquet": "split",
    "users_metadata.parquet": "collect",
}


def read_parquet(path: Path, columns: list[str] | None = None) -> pl.DataFrame:
    """Lê um Parquet.

    Parameters
    ----------
    path : Path
        Caminho do arquivo.
    columns : list of str, optional
        Subconjunto de colunas a carregar — ler só o necessário reduz
        drasticamente o uso de memória em matrizes com embeddings.

    Returns
    -------
    pl.DataFrame
        Conteúdo do arquivo.

    Raises
    ------
    DatasetNotFoundError
        Se o arquivo não existir, indicando a etapa que deveria criá-lo.

    Examples
    --------
    >>> read_parquet(Path("data/processed/user_features.parquet"))  # doctest: +SKIP
    """
    target = Path(path)
    if not target.is_file():
        stage = _PRODUCER_STAGE.get(target.name)
        hint = f" Execute antes: 'make {stage}' (etapa '{stage}')." if stage else ""
        raise DatasetNotFoundError(f"Artefato não encontrado: {target}.{hint}")

    frame = pl.read_parquet(target, columns=columns)
    logger.debug("Lido %s (%d linhas, %d colunas).", target.name, frame.height, frame.width)
    return frame


def scan_parquet(path: Path) -> pl.LazyFrame:
    """Abre um Parquet em modo preguiçoso (``LazyFrame``).

    Preferível a :func:`read_parquet` quando a etapa seguinte filtra ou
    agrega: o otimizador do polars empurra filtros e projeções para a leitura,
    e só materializa o que sobra.

    Parameters
    ----------
    path : Path
        Caminho do arquivo.

    Returns
    -------
    pl.LazyFrame
        Plano de consulta preguiçoso.

    Raises
    ------
    DatasetNotFoundError
        Se o arquivo não existir.

    Examples
    --------
    >>> scan_parquet(Path("data/interim/tweets_clean.parquet"))  # doctest: +SKIP
    """
    target = Path(path)
    if not target.is_file():
        stage = _PRODUCER_STAGE.get(target.name)
        hint = f" Execute antes a etapa '{stage}'." if stage else ""
        raise DatasetNotFoundError(f"Artefato não encontrado: {target}.{hint}")
    return pl.scan_parquet(target)


def read_user_histories(
    directory: Path,
    *,
    user_ids: list[str] | None = None,
) -> pl.DataFrame:
    """Lê e concatena os históricos por usuário gravados na coleta.

    Parameters
    ----------
    directory : Path
        Diretório com um ``.parquet`` por usuário.
    user_ids : list of str, optional
        Restringe a leitura a determinados usuários.

    Returns
    -------
    pl.DataFrame
        Histórico consolidado, ordenado por ``user_id`` e ``created_at``.

    Raises
    ------
    DatasetNotFoundError
        Se o diretório não contiver nenhum arquivo.

    Examples
    --------
    >>> read_user_histories(Path("data/raw/user_histories"))  # doctest: +SKIP
    """
    files = list_files(Path(directory), "*.parquet")
    if user_ids is not None:
        wanted = set(user_ids)
        files = [file for file in files if file.stem in wanted]

    if not files:
        raise DatasetNotFoundError(
            f"Nenhum histórico encontrado em {directory}. Execute antes a etapa 'collect'."
        )

    frames = [pl.read_parquet(file) for file in files]
    combined = pl.concat(frames, how="vertical_relaxed")

    logger.info("Lidos %d históricos (%d tweets no total).", len(files), combined.height)
    return combined.sort(["user_id", "created_at"])


def count_users(directory: Path) -> int:
    """Conta quantos históricos de usuário já foram coletados.

    Usado para retomar uma coleta interrompida sem baixar tudo de novo.

    Parameters
    ----------
    directory : Path
        Diretório dos históricos.

    Returns
    -------
    int
        Número de arquivos ``.parquet`` presentes.

    Examples
    --------
    >>> count_users(Path("data/raw/user_histories")) >= 0
    True
    """
    return len(list_files(Path(directory), "*.parquet"))


def list_collected_users(directory: Path) -> set[str]:
    """Lista os usuários cujo histórico já foi coletado.

    Parameters
    ----------
    directory : Path
        Diretório dos históricos (um arquivo por ``user_id``).

    Returns
    -------
    set of str
        Identificadores pseudonimizados já presentes em disco.

    Examples
    --------
    >>> isinstance(list_collected_users(Path("data/raw/user_histories")), set)
    True
    """
    return {file.stem for file in list_files(Path(directory), "*.parquet")}
