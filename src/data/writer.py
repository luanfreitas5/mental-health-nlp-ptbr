"""Escrita de artefatos de dados em Parquet.

Parquet com compressão ``zstd`` em vez de CSV: preserva o tipo das colunas
(uma data que volta como string quebra todas as features temporais), ocupa
uma fração do espaço e permite leitura preguiçosa por coluna, que é o que
torna viável processar históricos de milhares de usuários.

A escrita é atômica — arquivo temporário seguido de ``Path.replace`` — para
que uma interrupção não deixe um Parquet truncado que a etapa seguinte leria
como corrompido.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from config.logging import get_logger
from constants.defaults import PARQUET_COMPRESSION
from utils.hashing import hash_dataframe

logger = get_logger(__name__)


def write_parquet(
    frame: pl.DataFrame,
    path: Path,
    *,
    compression: str = PARQUET_COMPRESSION,
    log_hash: bool = True,
) -> Path:
    """Grava um DataFrame em Parquet de forma atômica.

    Parameters
    ----------
    frame : pl.DataFrame
        DataFrame a persistir.
    path : Path
        Caminho de destino (``.parquet``).
    compression : str, optional
        Codec de compressão, by default ``"zstd"``.
    log_hash : bool, optional
        Registra o hash do conteúdo no log, by default True. É o que permite
        confirmar, depois, que duas execuções produziram os mesmos dados.

    Returns
    -------
    Path
        Caminho gravado.

    Examples
    --------
    >>> write_parquet(pl.DataFrame({"a": [1]}), Path("data/interim/x.parquet"))  # doctest: +SKIP
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    temporary = target.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression=compression)  # type: ignore[arg-type]
    temporary.replace(target)

    message = "Gravado %s (%d linhas, %d colunas)"
    if log_hash:
        logger.info(
            f"{message} | sha256=%s",
            target.name,
            frame.height,
            frame.width,
            hash_dataframe(frame)[:16],
        )
    else:
        logger.info(message, target.name, frame.height, frame.width)

    return target


def write_partitioned(
    frame: pl.DataFrame,
    directory: Path,
    partition_column: str,
    *,
    compression: str = PARQUET_COMPRESSION,
    clear: bool = False,
) -> list[Path]:
    """Grava um Parquet por valor distinto de uma coluna.

    Usado na coleta: um arquivo por usuário permite retomar uma coleta
    interrompida sem reprocessar quem já foi baixado. Também usado para
    persistir artefatos grandes (ex.: ``tweets_clean``, ``tweets_labeled``)
    em vários arquivos menores, em vez de um único Parquet monolítico caro
    de carregar por inteiro nas etapas seguintes.

    Parameters
    ----------
    frame : pl.DataFrame
        DataFrame a particionar.
    directory : Path
        Diretório de destino.
    partition_column : str
        Coluna que define as partições (normalmente ``user_id``).
    compression : str, optional
        Codec de compressão, by default ``"zstd"``.
    clear : bool, optional
        Remove os ``.parquet`` já existentes no diretório antes de escrever,
        by default False. Necessário quando a etapa reescreve o artefato do
        zero a cada execução (ex.: preprocess, label) — sem isso, um usuário
        removido nesta execução deixaria seu arquivo antigo para trás, e a
        etapa seguinte o leria como se ainda fizesse parte do dataset. Na
        coleta (retomável), o padrão ``False`` é o que preserva o progresso.

    Returns
    -------
    list of Path
        Caminhos gravados, em ordem determinística.

    Raises
    ------
    KeyError
        Se a coluna de partição não existir.

    Examples
    --------
    >>> write_partitioned(frame, Path("data/raw/user_histories"), "user_id")  # doctest: +SKIP
    """
    if partition_column not in frame.columns:
        raise KeyError(
            f"Coluna de partição '{partition_column}' ausente. Disponíveis: {frame.columns}"
        )

    target_dir = Path(directory)
    if clear and target_dir.is_dir():
        for stale in target_dir.glob("*.parquet"):
            stale.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for (value,), partition in frame.partition_by(
        partition_column, as_dict=True, maintain_order=True
    ).items():
        path = target_dir / f"{value}.parquet"
        written.append(write_parquet(partition, path, compression=compression, log_hash=False))

    logger.info("Gravadas %d partições em %s.", len(written), target_dir)
    return written


def write_user_partition(frame: pl.DataFrame, directory: Path, user_id: str) -> Path:
    """Grava o resultado de um único usuário, imediatamente após seu processamento.

    É a contraparte, usuário a usuário, de :func:`write_partitioned`: em vez de
    processar todos os usuários em memória e só então gravar o diretório
    inteiro, cada etapa que itera por usuário chama esta função logo após
    processar cada um. Se a execução for interrompida, os usuários já
    gravados não precisam ser reprocessados na próxima chamada — basta
    verificar quem já existe no diretório (ver :func:`list_collected_users` e
    :func:`select_pending_users`).

    Parameters
    ----------
    frame : pl.DataFrame
        Resultado de um único usuário (pode ter zero linhas, quando o
        usuário foi processado mas não produziu saída — o arquivo ainda é
        gravado, para que ele conte como "já processado" e não seja
        reprocessado na próxima execução).
    directory : Path
        Diretório particionado por usuário.
    user_id : str
        Identificador pseudonimizado do usuário.

    Returns
    -------
    Path
        Caminho gravado (``<directory>/<user_id>.parquet``).

    Examples
    --------
    >>> destino = Path("data/interim/x")
    >>> write_user_partition(pl.DataFrame({"user_id": ["u_a"]}), destino, "u_a")  # doctest: +SKIP
    """
    return write_parquet(frame, Path(directory) / f"{user_id}.parquet", log_hash=False)


def append_parquet(frame: pl.DataFrame, path: Path) -> Path:
    """Concatena um DataFrame a um Parquet existente.

    Parameters
    ----------
    frame : pl.DataFrame
        Novas linhas.
    path : Path
        Arquivo de destino (criado se não existir).

    Returns
    -------
    Path
        Caminho gravado.

    Raises
    ------
    ValueError
        Se o esquema do novo bloco não for compatível com o do arquivo.

    Examples
    --------
    >>> append_parquet(novos, Path("data/interim/tweets_clean.parquet"))  # doctest: +SKIP
    """
    target = Path(path)
    if not target.is_file():
        return write_parquet(frame, target)

    existing = pl.read_parquet(target)
    if set(existing.columns) != set(frame.columns):
        raise ValueError(
            f"Esquemas incompatíveis ao concatenar em {target.name}. "
            f"Existente: {sorted(existing.columns)}; novo: {sorted(frame.columns)}."
        )

    combined = pl.concat([existing, frame.select(existing.columns)], how="vertical")
    return write_parquet(combined, target)
