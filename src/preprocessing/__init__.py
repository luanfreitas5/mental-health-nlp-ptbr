"""Limpeza, normalização e tokenização dos tweets.

O pacote mantém **dois** caminhos de texto que nunca se misturam:
``text_normalized`` (PII removida, semântica preservada — entrada dos
Transformers e do LLM) e ``text_clean`` (agressivamente reduzido — entrada do
TF-IDF, dos n-grams e dos léxicos).

Modules
-------
text
    Funções puras de normalização, limpeza e detecção residual de PII.
cleaning
    Deduplicação e filtros de qualidade, de atividade e de contas automatizadas.
tokenization
    :class:`Tokenizer` — spaCy com lematização e fallback por regex.
pipeline
    :func:`run_preprocessing` — encadeia tudo e valida os contratos.
"""

from preprocessing.cleaning import (
    deduplicate,
    filter_after_cleaning,
    filter_automated_accounts,
    filter_by_quality,
    filter_users_by_activity,
)
from preprocessing.pipeline import apply_text_processing, run_preprocessing
from preprocessing.text import clean_text, contains_pii, normalize_text, strip_accents, tokenize
from preprocessing.tokenization import Tokenizer, load_spacy_model

__all__ = [
    "Tokenizer",
    "apply_text_processing",
    "clean_text",
    "contains_pii",
    "deduplicate",
    "filter_after_cleaning",
    "filter_automated_accounts",
    "filter_by_quality",
    "filter_users_by_activity",
    "load_spacy_model",
    "normalize_text",
    "run_preprocessing",
    "strip_accents",
    "tokenize",
]
