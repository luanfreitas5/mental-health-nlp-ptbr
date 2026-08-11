"""Ingestão, leitura, escrita e particionamento dos dados.

Modules
-------
queries
    Constrói as consultas de busca a partir dos ``.txt`` de palavras-chave e
    hashtags (:func:`build_queries`, :class:`SearchQuery`).
collector
    :class:`TweetCollector` — coleta longitudinal via ``twscrape``, com
    pseudonimização na ingestão e retomada de coletas interrompidas.
reader
    Leitura de Parquet e dos históricos por usuário, com erros que apontam a
    etapa faltante.
writer
    Escrita atômica de Parquet, inclusive particionada por usuário.
splitter
    Partições treino/validação/teste e folds, agrupados por usuário para
    impedir vazamento.
catalog
    Situação dos artefatos e manifesto de hashes para reprodutibilidade.
"""

from data.catalog import build_catalog, compare_manifest, write_dataset_manifest
from data.collector import CandidateUser, TweetCollector
from data.queries import SearchQuery, build_queries
from data.reader import (
    count_partitioned_rows,
    read_parquet,
    read_user_histories,
    read_user_history,
    read_user_partition,
    scan_parquet,
)
from data.splitter import build_split_table, create_splits, filter_split
from data.writer import write_parquet, write_partitioned

__all__ = [
    "CandidateUser",
    "SearchQuery",
    "TweetCollector",
    "build_catalog",
    "build_queries",
    "build_split_table",
    "compare_manifest",
    "count_partitioned_rows",
    "create_splits",
    "filter_split",
    "read_parquet",
    "read_user_histories",
    "read_user_history",
    "read_user_partition",
    "scan_parquet",
    "write_dataset_manifest",
    "write_parquet",
    "write_partitioned",
]
