"""Carregamento dos léxicos psicolinguísticos.

Os léxicos vivem em ``configs/lexicons/*.txt`` (e não em código) porque são
material de pesquisa: revisá-los é parte do método, e cada revisão precisa
ser rastreável no histórico do git sem tocar em módulo Python.

Os padrões são compilados uma única vez e memoizados — são aplicados a
milhões de tweets, e recompilar por chamada dominaria o tempo da etapa.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from config.logging import get_logger
from config.paths import get_paths
from constants.regex import build_term_pattern
from utils.files import read_terms_file

logger = get_logger(__name__)

#: Léxicos de risco disponíveis (a chave é o nome usado em features.yaml).
LEXICON_NAMES: tuple[str, ...] = (
    "death",
    "loneliness",
    "hopelessness",
    "isolation",
    "negative_emotion",
    "insomnia",
)


def normalize_term(term: str) -> str:
    """Normaliza um termo de léxico para comparação robusta.

    Aplica caixa baixa e remove acentos, de modo que ``"solidão"`` no léxico
    case com ``"solidao"`` escrito sem acento no tweet — variação ortográfica
    é a regra, não a exceção, em texto de rede social.

    Parameters
    ----------
    term : str
        Termo original.

    Returns
    -------
    str
        Termo normalizado.

    Examples
    --------
    >>> normalize_term("Solidão")
    'solidao'
    """
    lowered = term.strip().lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


@dataclass(frozen=True)
class Lexicon:
    """Um léxico com seus termos e o padrão compilado.

    Attributes
    ----------
    name : str
        Nome do léxico (ex.: ``"death"``).
    terms : list of str
        Termos originais, como escritos no arquivo.
    pattern : re.Pattern
        Padrão que casa qualquer termo, já normalizado.
    """

    name: str
    terms: list[str] = field(repr=False)
    pattern: re.Pattern[str] = field(repr=False)

    def count(self, text: str) -> int:
        """Conta ocorrências de termos do léxico em um texto.

        Parameters
        ----------
        text : str
            Texto a inspecionar (será normalizado internamente).

        Returns
        -------
        int
            Número de ocorrências.

        Examples
        --------
        >>> lexicons = load_lexicons()
        >>> lexicons["loneliness"].count("me sinto sozinho e sozinho de novo")
        2
        """
        return len(self.pattern.findall(normalize_term(text)))

    def contains(self, text: str) -> bool:
        """Informa se o texto contém ao menos um termo do léxico.

        Parameters
        ----------
        text : str
            Texto a inspecionar.

        Returns
        -------
        bool
            ``True`` se houver ao menos uma ocorrência.

        Examples
        --------
        >>> load_lexicons()["death"].contains("quero apenas dormir")
        False
        """
        return bool(self.pattern.search(normalize_term(text)))


@lru_cache(maxsize=1)
def load_lexicons() -> dict[str, Lexicon]:
    """Carrega todos os léxicos de risco declarados em ``configs/paths.yaml``.

    Returns
    -------
    dict of str to Lexicon
        Nome -> léxico compilado. Léxicos com arquivo ausente são registrados
        como aviso e omitidos, em vez de derrubar o pipeline: um léxico
        opcional faltando degrada as features, não invalida a execução.

    Examples
    --------
    >>> lexicons = load_lexicons()
    >>> "death" in lexicons
    True
    """
    paths = get_paths().lexicons
    lexicons: dict[str, Lexicon] = {}

    for name in LEXICON_NAMES:
        path = getattr(paths, name)
        try:
            raw_terms = read_terms_file(path)
        except FileNotFoundError:
            logger.warning("Léxico '%s' não encontrado em %s: será ignorado.", name, path)
            continue

        normalized = [normalize_term(term) for term in raw_terms if term.strip()]
        if not normalized:
            logger.warning("Léxico '%s' está vazio: será ignorado.", name)
            continue

        lexicons[name] = Lexicon(
            name=name,
            terms=raw_terms,
            pattern=build_term_pattern(normalized),
        )
        logger.debug("Léxico '%s' carregado com %d termos.", name, len(normalized))

    logger.info("Léxicos carregados: %s.", ", ".join(sorted(lexicons)) or "nenhum")
    return lexicons


@lru_cache(maxsize=1)
def load_stopwords() -> frozenset[str]:
    """Carrega as stopwords do português brasileiro.

    Returns
    -------
    frozenset of str
        Stopwords normalizadas (conjunto vazio se o arquivo não existir).

    Examples
    --------
    >>> "de" in load_stopwords()
    True
    """
    path = get_paths().lexicons.stopwords
    try:
        terms = read_terms_file(path)
    except FileNotFoundError:
        logger.warning("Arquivo de stopwords não encontrado em %s.", path)
        return frozenset()
    return frozenset(normalize_term(term) for term in terms)
