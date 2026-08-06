"""Pseudonimização e *hashing* de dados.

Duas finalidades distintas:

* **Pseudonimização (LGPD).** Identificadores diretos — handle e id numérico
  do X/Twitter — são substituídos por um hash SHA-256 com *salt* logo na
  ingestão. Sem o salt, o hash seria trivialmente reversível: o espaço de
  handles é público e enumerável, então bastaria calcular o hash de cada
  handle conhecido para desfazer a proteção.
* **Rastreabilidade.** O hash do dataset entra no MLflow junto com o SHA do
  git, o que permite provar *qual* versão dos dados gerou *qual* modelo.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

#: Tamanho do bloco de leitura no hash de arquivo (1 MiB): arquivos de dados
#: chegam a vários GB e não cabem em memória de uma vez.
_CHUNK_SIZE: int = 1024 * 1024

#: Comprimento do identificador pseudonimizado. 16 hex = 64 bits: colisão
#: desprezível para milhares de usuários e muito mais legível em log e figura
#: do que os 64 caracteres do SHA-256 completo.
PSEUDONYM_LENGTH: int = 16


def hash_text(text: str, salt: str = "") -> str:
    """Calcula o SHA-256 de um texto, opcionalmente com *salt*.

    Parameters
    ----------
    text : str
        Texto de entrada.
    salt : str, optional
        Segredo prefixado ao texto, by default ``""``.

    Returns
    -------
    str
        Digest hexadecimal de 64 caracteres.

    Examples
    --------
    >>> hash_text("abc")[:8]
    'ba7816bf'
    """
    return hashlib.sha256(f"{salt}{text}".encode()).hexdigest()


def pseudonymize(identifier: str | int, salt: str, length: int = PSEUDONYM_LENGTH) -> str:
    """Pseudonimiza um identificador direto de forma determinística.

    Determinístico de propósito: o mesmo usuário precisa receber o mesmo
    pseudônimo em coletas diferentes, senão seria impossível montar o
    histórico longitudinal. Irreversível sem o salt.

    Parameters
    ----------
    identifier : str or int
        Identificador direto (handle ou id numérico).
    salt : str
        Segredo vindo do ``.env`` (``PSEUDONYMIZATION_SALT``).
    length : int, optional
        Número de caracteres hexadecimais mantidos, by default 16.

    Returns
    -------
    str
        Pseudônimo no formato ``u_<hex>``.

    Raises
    ------
    ValueError
        Se o salt estiver vazio — pseudonimizar sem salt não protege nada.

    Examples
    --------
    >>> pseudonymize("usuario_exemplo", salt="segredo")[:2]
    'u_'
    >>> pseudonymize("a", "s") == pseudonymize("a", "s")
    True
    """
    if not salt:
        raise ValueError(
            "Salt vazio: a pseudonimização seria reversível por força bruta sobre "
            "handles públicos. Defina PSEUDONYMIZATION_SALT no .env."
        )
    digest = hash_text(str(identifier), salt=salt)
    return f"u_{digest[:length]}"


def pseudonymize_column(
    frame: pl.DataFrame,
    column: str,
    salt: str,
    *,
    alias: str | None = None,
    drop_original: bool = True,
) -> pl.DataFrame:
    """Substitui uma coluna de identificadores diretos pelos pseudônimos.

    Parameters
    ----------
    frame : pl.DataFrame
        DataFrame de entrada.
    column : str
        Coluna com o identificador direto.
    salt : str
        Segredo de pseudonimização.
    alias : str, optional
        Nome da coluna resultante, by default o mesmo de ``column``.
    drop_original : bool, optional
        Remove a coluna original quando ``alias`` difere dela, by default True.

    Returns
    -------
    pl.DataFrame
        DataFrame com a coluna pseudonimizada.

    Raises
    ------
    KeyError
        Se a coluna não existir no DataFrame.

    Examples
    --------
    >>> frame = pl.DataFrame({"handle": ["a", "b"]})
    >>> resultado = pseudonymize_column(frame, "handle", salt="s", alias="user_id")
    >>> resultado.columns
    ['user_id']
    """
    if column not in frame.columns:
        raise KeyError(f"Coluna '{column}' ausente. Disponíveis: {frame.columns}")

    target = alias or column
    result = frame.with_columns(
        pl.col(column)
        .cast(pl.Utf8)
        .map_elements(lambda value: pseudonymize(value, salt), return_dtype=pl.Utf8)
        .alias(target)
    )
    if drop_original and target != column:
        result = result.drop(column)
    return result


def hash_file(path: Path, chunk_size: int = _CHUNK_SIZE) -> str:
    """Calcula o SHA-256 de um arquivo, lendo em blocos.

    Parameters
    ----------
    path : Path
        Caminho do arquivo.
    chunk_size : int, optional
        Tamanho do bloco de leitura em bytes, by default 1 MiB.

    Returns
    -------
    str
        Digest hexadecimal do conteúdo.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.

    Examples
    --------
    >>> hash_file(Path("pyproject.toml")) != ""
    True
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado para hash: {target}")

    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_directory(path: Path, chunk_size: int = _CHUNK_SIZE) -> str:
    """Calcula um hash estável do conteúdo de um diretório particionado.

    Combina o nome e o hash de cada arquivo ``.parquet``, em ordem
    determinística por nome — necessário porque artefatos grandes (ex.:
    ``tweets_clean``) são gravados em vários arquivos, um por usuário, em vez
    de um único Parquet monolítico.

    Parameters
    ----------
    path : Path
        Diretório com um ou mais arquivos ``.parquet``.
    chunk_size : int, optional
        Tamanho do bloco de leitura em bytes, by default 1 MiB.

    Returns
    -------
    str
        Digest hexadecimal combinado do conteúdo do diretório.

    Raises
    ------
    FileNotFoundError
        Se o diretório não existir ou não contiver nenhum arquivo ``.parquet``.

    Examples
    --------
    >>> hash_directory(Path("data/interim/tweets_clean")) != ""  # doctest: +SKIP
    True
    """
    target = Path(path)
    files = sorted(target.glob("*.parquet")) if target.is_dir() else []
    if not files:
        raise FileNotFoundError(f"Diretório vazio ou inexistente para hash: {target}")

    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        digest.update(hash_file(file, chunk_size).encode())
    return digest.hexdigest()


def hash_dataframe(frame: pl.DataFrame) -> str:
    """Calcula um hash estável do conteúdo de um DataFrame.

    Estável quanto à ordem das colunas (ordenadas antes do hash), mas
    **sensível** à ordem das linhas — reordenar linhas muda o hash, o que é
    desejável: a ordem afeta particionamento e resultados de modelos.

    Parameters
    ----------
    frame : pl.DataFrame
        DataFrame a resumir.

    Returns
    -------
    str
        Digest hexadecimal do conteúdo.

    Examples
    --------
    >>> a = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
    >>> b = pl.DataFrame({"y": [3, 4], "x": [1, 2]})
    >>> hash_dataframe(a) == hash_dataframe(b)
    True
    """
    ordered = frame.select(sorted(frame.columns))
    payload = str(ordered.schema).encode() + ordered.hash_rows().to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def build_manifest(artifacts: dict[str, Path]) -> dict[str, Any]:
    """Monta um manifesto com o hash de cada artefato de dados.

    Gravado junto do dataset processado, é o que permite detectar que os dados
    mudaram silenciosamente entre duas execuções.

    Parameters
    ----------
    artifacts : dict of str to Path
        Nome lógico -> caminho do arquivo.

    Returns
    -------
    dict
        Nome -> ``{"path", "sha256", "size_bytes"}`` (mais ``"n_files"`` para
        artefatos particionados em diretório). Artefatos ausentes recebem
        ``{"path", "status": "ausente"}``.

    Examples
    --------
    >>> manifest = build_manifest({"pyproject": Path("pyproject.toml")})
    >>> "sha256" in manifest["pyproject"]
    True
    """
    manifest: dict[str, Any] = {}
    for name, path in artifacts.items():
        target = Path(path)
        if target.is_dir():
            files = sorted(target.glob("*.parquet"))
            if not files:
                manifest[name] = {"path": str(target), "status": "ausente"}
                continue
            manifest[name] = {
                "path": str(target),
                "sha256": hash_directory(target),
                "size_bytes": sum(file.stat().st_size for file in files),
                "n_files": len(files),
            }
            continue
        if not target.is_file():
            manifest[name] = {"path": str(target), "status": "ausente"}
            continue
        manifest[name] = {
            "path": str(target),
            "sha256": hash_file(target),
            "size_bytes": target.stat().st_size,
        }
    return manifest


def hash_payload(payload: dict[str, Any]) -> str:
    """Calcula um hash estável de um dicionário serializável em JSON.

    Usado como chave do cache de respostas do LLM: o mesmo prompt, com os
    mesmos parâmetros, reaproveita a resposta em vez de gastar inferência.

    Parameters
    ----------
    payload : dict
        Dicionário serializável.

    Returns
    -------
    str
        Digest hexadecimal.

    Examples
    --------
    >>> hash_payload({"b": 1, "a": 2}) == hash_payload({"a": 2, "b": 1})
    True
    """
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
