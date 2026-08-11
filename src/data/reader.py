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

    before = combined.height
    combined = combined.unique(subset=["tweet_id"], keep="first", maintain_order=True)
    removed = before - combined.height
    if removed:
        # Uma mesma conta pode ter sido exportada duas vezes sob screen_names
        # diferentes (ex.: troca de @) — cada exportação vira um arquivo
        # próprio, mas os tweets pseudonimizados colidem no `tweet_id`. Sem
        # esse dedupe, `RawTweetSchema` rejeitaria a entrada inteira do
        # preprocess por violação de unicidade.
        logger.warning(
            "Removidos %d tweets duplicados entre arquivos de histórico "
            "(mesmo tweet_id em usuários/arquivos diferentes).",
            removed,
        )

    logger.info("Lidos %d históricos (%d tweets no total).", len(files), combined.height)
    return combined.sort(["user_id", "created_at"])


def read_user_history(directory: Path, user_id: str) -> pl.DataFrame:
    """Lê o histórico bruto de um único usuário, sem concatenar os demais.

    Contraparte de :func:`read_user_histories` restrita a um usuário: aplica
    o mesmo dedupe por ``tweet_id`` e a mesma ordenação por ``created_at``,
    mas lendo só o arquivo do usuário pedido — usada pelo pré-processamento,
    que processa um usuário por vez e não precisa (nem deve) materializar o
    histórico inteiro para extrair a fatia de um só.

    Parameters
    ----------
    directory : Path
        Diretório com um ``.parquet`` por usuário (histórico bruto da coleta).
    user_id : str
        Identificador pseudonimizado do usuário.

    Returns
    -------
    pl.DataFrame
        Histórico do usuário, deduplicado por ``tweet_id`` e ordenado por
        ``created_at``; DataFrame vazio se o arquivo não existir.

    Examples
    --------
    >>> read_user_history(Path("data/raw/user_histories"), "u_a")  # doctest: +SKIP
    """
    path = Path(directory) / f"{user_id}.parquet"
    if not path.is_file():
        return pl.DataFrame()

    frame = pl.read_parquet(path)
    before = frame.height
    frame = frame.unique(subset=["tweet_id"], keep="first", maintain_order=True)
    removed = before - frame.height
    if removed:
        logger.warning(
            "Removidos %d tweets duplicados no histórico do usuário (mesmo tweet_id).",
            removed,
        )

    return frame.sort("created_at") if "created_at" in frame.columns else frame


def read_partitioned(directory: Path, *, stage: str | None = None) -> pl.DataFrame:
    """Lê e concatena os arquivos de um artefato particionado em diretório.

    Usado para artefatos grandes gravados em vários arquivos (um por usuário)
    em vez de um único Parquet monolítico — evita que uma etapa precise
    carregar um arquivo gigante de uma só vez.

    Parameters
    ----------
    directory : Path
        Diretório com um ``.parquet`` por partição.
    stage : str, optional
        Nome da etapa que deveria ter produzido o artefato, usado apenas
        para compor a mensagem de erro quando o diretório está vazio.

    Returns
    -------
    pl.DataFrame
        Conteúdo consolidado, ordenado por ``user_id`` (e ``created_at``,
        quando presente).

    Raises
    ------
    DatasetNotFoundError
        Se o diretório não existir ou não contiver nenhum arquivo.

    Examples
    --------
    >>> read_partitioned(Path("data/interim/tweets_clean"), stage="preprocess")  # doctest: +SKIP
    """
    files = list_files(Path(directory), "*.parquet")
    if not files:
        hint = f" Execute antes a etapa '{stage}'." if stage else ""
        raise DatasetNotFoundError(f"Artefato particionado não encontrado: {directory}.{hint}")

    frames = [pl.read_parquet(file) for file in files]
    combined = pl.concat(frames, how="vertical_relaxed")

    sort_columns = [column for column in ("user_id", "created_at") if column in combined.columns]
    if sort_columns:
        combined = combined.sort(sort_columns)

    logger.info(
        "Lidos %d arquivos particionados (%d linhas, %d colunas) de %s.",
        len(files),
        combined.height,
        combined.width,
        directory,
    )
    return combined


def read_user_partition(directory: Path, user_id: str) -> pl.DataFrame:
    """Lê o arquivo de um único usuário num artefato particionado, sem tocar os demais.

    É a contraparte de leitura de :func:`data.writer.write_user_partition`: em
    vez de carregar o diretório inteiro em memória e depois filtrar por
    usuário, cada etapa que processa um usuário por vez lê diretamente o
    arquivo dele — o que mantém o uso de memória e de I/O proporcional a um
    usuário, não à população inteira.

    Parameters
    ----------
    directory : Path
        Diretório particionado por usuário (um arquivo por ``user_id``).
    user_id : str
        Identificador pseudonimizado do usuário.

    Returns
    -------
    pl.DataFrame
        Conteúdo do usuário, ou um DataFrame vazio se o arquivo não existir
        (usuário sem dados nesta etapa).

    Examples
    --------
    >>> read_user_partition(Path("data/interim/tweets_clean"), "u_a")  # doctest: +SKIP
    """
    path = Path(directory) / f"{user_id}.parquet"
    if not path.is_file():
        return pl.DataFrame()

    frame = pl.read_parquet(path)
    logger.debug("Lido %s (%d linhas, %d colunas).", path.name, frame.height, frame.width)
    return frame


def count_partitioned_rows(directory: Path) -> int:
    """Conta as linhas de um artefato particionado sem concatenar os arquivos em memória.

    Usada para compor resumos de execução (ex.: total de tweets processados)
    a partir de um diretório particionado por usuário, sem pagar o custo de
    materializar a população inteira só para contar linhas.

    Parameters
    ----------
    directory : Path
        Diretório particionado por usuário.

    Returns
    -------
    int
        Soma de linhas de todos os arquivos ``.parquet`` do diretório
        (``0`` se o diretório estiver vazio ou não existir).

    Examples
    --------
    >>> count_partitioned_rows(Path("data/interim/tweets_clean"))  # doctest: +SKIP
    """
    files = list_files(Path(directory), "*.parquet")
    if not files:
        return 0
    return sum(int(pl.scan_parquet(file).select(pl.len()).collect().item()) for file in files)


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
    """Lista os usuários já processados num diretório particionado por usuário.

    Usada originalmente para retomar a coleta, mas serve a qualquer etapa que
    grava um ``.parquet`` por usuário (``preprocess``, ``label``, ``psych``) —
    o nome do arquivo (sem extensão) é o próprio ``user_id``.

    Parameters
    ----------
    directory : Path
        Diretório particionado por usuário (um arquivo por ``user_id``).

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


def select_pending_users(
    available: set[str],
    already_processed: set[str],
    limit: int | None = None,
) -> list[str]:
    """Calcula quais usuários processar nesta execução.

    Centraliza a lógica de retomada usada por toda etapa que processa um
    usuário por vez: ignora quem já tem resultado em disco e, se
    ``--limit-users`` estiver definido, processa só os próximos ``limit``
    pendentes — o que permite interromper e retomar sem reprocessar ninguém.

    Parameters
    ----------
    available : set of str
        Usuários disponíveis na entrada da etapa.
    already_processed : set of str
        Usuários que já têm resultado gravado (serão ignorados).
    limit : int, optional
        Número máximo de usuários pendentes a processar nesta execução;
        ``None`` processa todos os pendentes.

    Returns
    -------
    list of str
        Usuários pendentes, em ordem determinística, já limitados.

    Examples
    --------
    >>> select_pending_users({"u_a", "u_b", "u_c"}, {"u_a"}, limit=1)
    ['u_b']
    """
    pending = sorted(available - already_processed)
    if limit is not None:
        pending = pending[:limit]
    return pending
