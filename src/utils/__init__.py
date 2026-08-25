"""Utilitários transversais compartilhados por todo o projeto.

Modules
-------
hashing
    Pseudonimização (LGPD), hash de arquivos/DataFrames e manifesto de dados.
files
    Leitura de arquivos de termos e escrita atômica de JSON/texto.
lexicons
    Carregamento e compilação dos léxicos psicolinguísticos.
parallel
    Distribuição do laço de processamento por usuário entre processos (etapas
    ``preprocess`` e ``features``, CPU-bound).
progress
    Barras de progresso ``rich`` padronizadas.
timing
    Cronômetro, decorador ``@timed`` e formatação de duração em pt-BR.
validation
    Verificações defensivas de DataFrame, balanceamento e vazamento entre grupos.
"""

from utils.files import read_json, read_terms_file, write_json, write_text
from utils.hashing import build_manifest, hash_dataframe, hash_file, pseudonymize
from utils.lexicons import Lexicon, load_lexicons, load_stopwords, normalize_term
from utils.parallel import resolve_worker_count, run_user_pool
from utils.progress import build_progress, track
from utils.timing import Timer, format_duration, log_duration, timed
from utils.validation import (
    check_class_balance,
    check_no_group_leakage,
    require_columns,
    require_non_empty,
)

__all__ = [
    "Lexicon",
    "Timer",
    "build_manifest",
    "build_progress",
    "check_class_balance",
    "check_no_group_leakage",
    "format_duration",
    "hash_dataframe",
    "hash_file",
    "load_lexicons",
    "load_stopwords",
    "log_duration",
    "normalize_term",
    "pseudonymize",
    "read_json",
    "read_terms_file",
    "require_columns",
    "require_non_empty",
    "resolve_worker_count",
    "run_user_pool",
    "timed",
    "track",
    "write_json",
    "write_text",
]
