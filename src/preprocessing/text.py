"""Normalização e limpeza de texto em dois caminhos independentes.

A distinção entre :func:`normalize_text` e :func:`clean_text` é deliberada e
central para o projeto:

* :func:`normalize_text` remove PII e ruído, mas **preserva** caixa,
  pontuação, emoji e negações. É a entrada do BERTimbau e do LLM, que
  extraem sinal justamente desses elementos — aplicar limpeza agressiva antes
  de um Transformer destrói informação que o modelo foi pré-treinado para usar.
* :func:`clean_text` reduz agressivamente (minúsculas, sem pontuação, sem
  stopwords). É a entrada do TF-IDF e das contagens lexicais, onde a
  variação superficial só aumenta a esparsidade.
"""

from __future__ import annotations

import unicodedata

from config.settings import CleaningSection, NormalizationSection
from constants.regex import (
    CONTROL_CHARS_PATTERN,
    EMAIL_PATTERN,
    EMOJI_PATTERN,
    HASHTAG_PATTERN,
    MENTION_PATTERN,
    NUMBER_PATTERN,
    PHONE_PATTERN,
    PUNCTUATION_PATTERN,
    REPEATED_CHARS_PATTERN,
    RETWEET_PATTERN,
    TOKEN_PATTERN,
    URL_PATTERN,
    WHITESPACE_PATTERN,
)


def strip_accents(text: str) -> str:
    """Remove os acentos de um texto.

    Parameters
    ----------
    text : str
        Texto de entrada.

    Returns
    -------
    str
        Texto sem sinais diacríticos.

    Examples
    --------
    >>> strip_accents("solidão")
    'solidao'
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def normalize_text(text: str, config: NormalizationSection) -> str:
    """Normaliza um tweet preservando a semântica (entrada de Transformers/LLM).

    A ordem das operações importa: e-mails são substituídos **antes** das
    menções, porque ``fulano@dominio.com`` contém um padrão que a regex de
    menção também casaria, e a substituição na ordem errada produziria
    ``fulano@user``.

    Parameters
    ----------
    text : str
        Texto bruto do tweet.
    config : NormalizationSection
        Seção ``normalization`` de ``configs/preprocessing.yaml``.

    Returns
    -------
    str
        Texto normalizado, sem PII.

    Examples
    --------
    >>> from config.settings import NormalizationSection
    >>> normalize_text("Oi @fulano veja http://x.com #saudemental", NormalizationSection())
    'Oi @user veja URL saudemental'
    """
    if not text:
        return ""

    result = unicodedata.normalize(config.unicode_form, text)
    result = RETWEET_PATTERN.sub("", result)

    if config.strip_control_chars:
        result = CONTROL_CHARS_PATTERN.sub(" ", result)

    # PII: e-mail antes de menção (ver docstring), telefone antes de número.
    if config.replace_emails is not None:
        result = EMAIL_PATTERN.sub(config.replace_emails, result)
    if config.replace_urls is not None:
        result = URL_PATTERN.sub(config.replace_urls, result)
    if config.replace_mentions is not None:
        result = MENTION_PATTERN.sub(config.replace_mentions, result)
    if config.replace_phone_numbers is not None:
        result = PHONE_PATTERN.sub(config.replace_phone_numbers, result)
    if config.replace_numbers is not None:
        result = NUMBER_PATTERN.sub(config.replace_numbers, result)

    if config.unpack_hashtags:
        result = HASHTAG_PATTERN.sub(r"\1", result)

    if config.collapse_repeated_chars:
        keep = config.collapse_repeated_chars
        result = REPEATED_CHARS_PATTERN.sub(lambda match: match.group(1) * keep, result)

    if config.demojize:
        result = EMOJI_PATTERN.sub(" ", result)

    if config.collapse_whitespace:
        result = WHITESPACE_PATTERN.sub(" ", result)

    return result.strip()


def clean_text(text: str, config: CleaningSection, stopwords: frozenset[str]) -> str:
    """Aplica a limpeza agressiva (entrada de TF-IDF, n-grams e léxicos).

    As stopwords da whitelist são preservadas mesmo estando na lista de
    remoção: pronomes de 1ª pessoa e negações são features centrais na
    literatura de detecção de depressão, e removê-los apagaria justamente o
    sinal que o projeto procura.

    Parameters
    ----------
    text : str
        Texto já normalizado.
    config : CleaningSection
        Seção ``cleaning`` de ``configs/preprocessing.yaml``.
    stopwords : frozenset of str
        Stopwords a remover.

    Returns
    -------
    str
        Texto limpo, com tokens separados por espaço.

    Examples
    --------
    >>> from config.settings import CleaningSection
    >>> clean_text("Eu não estou bem hoje!", CleaningSection(), frozenset({"estou"}))
    'eu não bem hoje'
    """
    if not text:
        return ""

    result = text.lower() if config.lowercase else text

    if config.remove_emojis:
        result = EMOJI_PATTERN.sub(" ", result)
    if config.remove_accents:
        result = strip_accents(result)
    if config.remove_punctuation:
        result = PUNCTUATION_PATTERN.sub(" ", result)

    whitelist = {term.lower() for term in config.stopwords_whitelist}
    tokens: list[str] = []
    for token in WHITESPACE_PATTERN.split(result):
        if not token or len(token) < config.min_token_length:
            continue
        # A lista de stopwords é comparada com e sem acento: em rede social,
        # "não" e "nao" aparecem com frequência semelhante.
        if (
            config.remove_stopwords
            and token not in whitelist
            and (token in stopwords or strip_accents(token) in stopwords)
        ):
            continue
        tokens.append(token)

    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    """Tokeniza um texto por expressão regular (fallback sem spaCy).

    Parameters
    ----------
    text : str
        Texto de entrada.

    Returns
    -------
    list of str
        Tokens alfabéticos, incluindo acentuados.

    Examples
    --------
    >>> tokenize("não estou bem, hoje!")
    ['não', 'estou', 'bem', 'hoje']
    """
    return TOKEN_PATTERN.findall(text)


def count_characters(text: str) -> int:
    """Conta os caracteres de um texto.

    Parameters
    ----------
    text : str
        Texto de entrada.

    Returns
    -------
    int
        Número de caracteres.

    Examples
    --------
    >>> count_characters("abc")
    3
    """
    return len(text)


def extract_hashtags(text: str) -> list[str]:
    """Extrai as hashtags de um texto.

    Parameters
    ----------
    text : str
        Texto bruto (antes do desempacotamento de hashtags).

    Returns
    -------
    list of str
        Termos das hashtags, sem o ``#`` e em minúsculas.

    Examples
    --------
    >>> extract_hashtags("dia difícil #Desabafo #saudemental")
    ['desabafo', 'saudemental']
    """
    return [match.lower() for match in HASHTAG_PATTERN.findall(text)]


#: Placeholders inseridos pela normalização. Não são PII — são a marca de que
#: a PII foi removida — e por isso não podem disparar a salvaguarda.
PII_PLACEHOLDERS: tuple[str, ...] = ("@user", "URL", "EMAIL", "TELEFONE")


def contains_pii(text: str) -> bool:
    """Verifica se restou alguma PII evidente no texto.

    Usado como salvaguarda antes de enviar conteúdo ao LLM
    (``llm.safeguards.require_pii_scrubbed_input``). Os placeholders da
    normalização são desconsiderados: ``@user`` casa com o padrão de menção,
    mas é justamente o sinal de que a menção original foi removida.

    Parameters
    ----------
    text : str
        Texto a inspecionar.

    Returns
    -------
    bool
        ``True`` se houver menção, URL, e-mail ou telefone reais.

    Examples
    --------
    >>> contains_pii("meu email é a@b.com")
    True
    >>> contains_pii("oi @user, veja URL")
    False
    >>> contains_pii("hoje foi um dia difícil")
    False
    """
    candidate = text
    for placeholder in PII_PLACEHOLDERS:
        candidate = candidate.replace(placeholder, " ")

    return any(
        pattern.search(candidate)
        for pattern in (EMAIL_PATTERN, URL_PATTERN, MENTION_PATTERN, PHONE_PATTERN)
    )
