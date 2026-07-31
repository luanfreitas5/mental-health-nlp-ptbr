"""Padrões de expressão regular compartilhados.

Compilados uma única vez no import: são aplicados a milhões de tweets nas
etapas de limpeza e de redação de PII, e recompilar a cada chamada dominaria
o tempo de processamento.

Os padrões de PII (:data:`MENTION_PATTERN`, :data:`URL_PATTERN`,
:data:`EMAIL_PATTERN`, :data:`PHONE_PATTERN`) são usados em dois lugares
distintos: na normalização do texto e no filtro de log — a mesma definição
para as duas fronteiras, evitando que uma proteção fique defasada da outra.
"""

from __future__ import annotations

import re
from typing import Final

# --- PII --------------------------------------------------------------------

#: Menção a outro usuário (``@fulano``).
MENTION_PATTERN: Final[re.Pattern[str]] = re.compile(r"@[A-Za-z0-9_]{1,15}\b")

#: URL http(s) ou iniciada por ``www.``.
URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:https?://|www\.)[^\s<>\"]+",
    flags=re.IGNORECASE,
)

#: Endereço de e-mail.
EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

#: Telefone brasileiro, com ou sem DDD, separadores e código do país.
PHONE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:\+?55[\s-]?)?(?:\(?\d{2}\)?[\s-]?)?9?\d{4}[\s-]?\d{4}\b"
)

#: Mapa nome -> padrão, consumido por ``logging.redaction.patterns``.
PII_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "mention": MENTION_PATTERN,
    "url": URL_PATTERN,
    "email": EMAIL_PATTERN,
    "phone": PHONE_PATTERN,
}

# --- Estrutura do tweet -----------------------------------------------------

#: Hashtag (``#saudemental``); o grupo 1 captura o termo sem o ``#``.
HASHTAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"#(\w+)", flags=re.UNICODE)

#: Marcador de retweet no início do texto (``RT @fulano:``).
RETWEET_PATTERN: Final[re.Pattern[str]] = re.compile(r"^RT\s+@[A-Za-z0-9_]+:\s*")

# --- Normalização -----------------------------------------------------------

#: Sequência de espaços em branco (inclui quebras de linha).
WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")

#: Caracteres de controle (exceto tabulação e quebra de linha).
CONTROL_CHARS_PATTERN: Final[re.Pattern[str]] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: Três ou mais repetições do mesmo caractere (``muitooooo``); grupo 1 = caractere.
REPEATED_CHARS_PATTERN: Final[re.Pattern[str]] = re.compile(r"(.)\1{2,}", flags=re.UNICODE)

#: Números inteiros ou decimais isolados.
NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b\d+(?:[.,]\d+)?\b")

#: Pontuação e símbolos (mantém letras acentuadas, dígitos e espaço).
PUNCTUATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[^\w\s]|_",
    flags=re.UNICODE,
)

#: Emojis, símbolos pictográficos, transporte e bandeiras.
EMOJI_PATTERN: Final[re.Pattern[str]] = re.compile(
    "["
    "\U0001f300-\U0001f5ff"  # símbolos e pictogramas
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f680-\U0001f6ff"  # transporte e mapas
    "\U0001f700-\U0001f77f"  # alquímicos
    "\U0001f780-\U0001f7ff"  # geométricos estendidos
    "\U0001f800-\U0001f8ff"  # setas suplementares
    "\U0001f900-\U0001f9ff"  # suplementares e emoji
    "\U0001fa00-\U0001faff"  # símbolos estendidos
    "\U00002600-\U000026ff"  # diversos
    "\U00002700-\U000027bf"  # dingbats
    "\U0001f1e6-\U0001f1ff"  # bandeiras regionais
    "\U0000fe00-\U0000fe0f"  # seletores de variação
    "\U00002b00-\U00002bff"  # setas diversas
    "]+",
    flags=re.UNICODE,
)

# --- Tokenização ------------------------------------------------------------

#: Token do fallback por regex: palavra com letras acentuadas, hífen ou apóstrofo.
TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[^\W\d_]+(?:['\-][^\W\d_]+)*", flags=re.UNICODE
)

#: Fronteira de sentença simples (para contagem de sentenças).
SENTENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"[.!?]+(?:\s+|$)")


def build_term_pattern(terms: list[str]) -> re.Pattern[str]:
    """Compila um padrão que casa qualquer termo de um léxico.

    Os termos são ordenados do mais longo para o mais curto, de modo que a
    expressão mais específica vença (``"não quero mais viver"`` antes de
    ``"viver"``), e delimitados por fronteira de palavra para não casar
    dentro de outra palavra.

    Parameters
    ----------
    terms : list of str
        Termos ou expressões do léxico, já normalizados.

    Returns
    -------
    re.Pattern
        Padrão compilado, insensível a caixa.

    Raises
    ------
    ValueError
        Se a lista de termos estiver vazia.

    Examples
    --------
    >>> pattern = build_term_pattern(["quero morrer", "morrer"])
    >>> bool(pattern.search("hoje eu quero morrer de sono"))
    True
    """
    if not terms:
        raise ValueError("Não é possível compilar um padrão a partir de um léxico vazio.")

    ordered = sorted(set(terms), key=len, reverse=True)
    alternatives = "|".join(re.escape(term) for term in ordered)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", flags=re.IGNORECASE | re.UNICODE)
