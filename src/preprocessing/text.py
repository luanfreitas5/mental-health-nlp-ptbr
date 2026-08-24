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
from typing import Final

import polars as pl

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

#: Marcas diacríticas combinantes (bloco Unicode U+0300-U+036F: agudo, grave,
#: til, cedilha combinante, trema), usado pela versão vetorizada de
#: :func:`strip_accents` (:func:`_strip_accents_expr`) após decomposição NFD.
#: Contém os caracteres reais (não ``\uXXXX``), como :data:`constants.regex.
#: EMOJI_PATTERN`: o motor de regex Rust do polars não entende essa notação
#: de escape do Python.
_ACCENT_MARKS_PATTERN: Final[str] = "[̀-ͯ]"


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
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    # NFC recompõe sequências que a decomposição NFD separa sem serem
    # diacríticos latinos (ex.: silabas Hangul viram Jamo), o que violaria a
    # invariante de que remover acentos nunca aumenta o comprimento do texto.
    return unicodedata.normalize("NFC", stripped)


def _strip_accents_expr(expr: pl.Expr) -> pl.Expr:
    """Versão vetorizada de :func:`strip_accents` para uma expressão polars.

    Mesma lógica (NFD, remove marcas combinantes, recompõe NFC), reescrita
    como expressões nativas para rodar sobre a coluna inteira em uma
    passada, sem UDF por linha. Usada tanto na etapa de limpeza
    (:func:`clean_text_expr`, texto inteiro) quanto na comparação de
    stopwords sem acento por token (dentro de ``list.eval``).

    Parameters
    ----------
    expr : pl.Expr
        Expressão de texto (coluna inteira ou ``pl.element()`` de uma lista).

    Returns
    -------
    pl.Expr
        Expressão sem sinais diacríticos.
    """
    return (
        expr.str.normalize("NFD")
        .str.replace_all(_ACCENT_MARKS_PATTERN, "", literal=False)
        .str.normalize("NFC")
    )


def collapse_repeated_chars(text: str, keep: int) -> str:
    """Colapsa sequências de 3+ repetições do mesmo caractere para ``keep`` cópias.

    Isolada como função própria porque é a única etapa de
    :func:`normalize_text` sem equivalente vetorizado no polars: o padrão
    (:data:`constants.regex.REPEATED_CHARS_PATTERN`) usa uma referência
    retroativa (``(.)\\1{2,}``) que o motor de regex do polars (Rust
    ``regex``, sem suporte a backtracking) não implementa. Reusada pelo
    caminho escalar (:func:`_apply_normalization_options`) e pelo caminho
    vetorizado (``preprocessing.pipeline.apply_text_processing``, via
    ``map_elements`` restrito a esta única etapa).

    Parameters
    ----------
    text : str
        Texto de entrada.
    keep : int
        Número de cópias do caractere repetido a preservar.

    Returns
    -------
    str
        Texto com as repetições colapsadas.

    Examples
    --------
    >>> collapse_repeated_chars("muitooooo triste", keep=2)
    'muitoo triste'
    """
    return REPEATED_CHARS_PATTERN.sub(lambda match: match.group(1) * keep, text)


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

    result = _apply_pii_redaction(result, config)

    return _apply_normalization_options(result, config).strip()


def _apply_pii_redaction(text: str, config: NormalizationSection) -> str:
    """Substitui e-mails, URLs, menções, telefones e números por placeholders.

    A ordem importa: e-mail antes de menção (ver docstring de
    :func:`normalize_text`), telefone antes de número.

    Parameters
    ----------
    text : str
        Texto já normalizado em Unicode, sem retweet nem caracteres de controle.
    config : NormalizationSection
        Seção ``normalization`` de ``configs/preprocessing.yaml``.

    Returns
    -------
    str
        Texto com PII substituída pelos placeholders configurados.
    """
    result = _redact_contact_identifiers(text, config)
    return _redact_numeric_identifiers(result, config)


def _redact_contact_identifiers(text: str, config: NormalizationSection) -> str:
    """Substitui e-mails, URLs e menções por placeholders (e-mail antes de menção).

    Parameters
    ----------
    text : str
        Texto de entrada.
    config : NormalizationSection
        Seção ``normalization`` de ``configs/preprocessing.yaml``.

    Returns
    -------
    str
        Texto com e-mails, URLs e menções substituídos.
    """
    result = text
    if config.replace_emails is not None:
        result = EMAIL_PATTERN.sub(config.replace_emails, result)
    if config.replace_urls is not None:
        result = URL_PATTERN.sub(config.replace_urls, result)
    if config.replace_mentions is not None:
        result = MENTION_PATTERN.sub(config.replace_mentions, result)
    return result


def _redact_numeric_identifiers(text: str, config: NormalizationSection) -> str:
    """Substitui telefones e números por placeholders (telefone antes de número).

    Parameters
    ----------
    text : str
        Texto de entrada.
    config : NormalizationSection
        Seção ``normalization`` de ``configs/preprocessing.yaml``.

    Returns
    -------
    str
        Texto com telefones e números substituídos.
    """
    result = text
    if config.replace_phone_numbers is not None:
        result = PHONE_PATTERN.sub(config.replace_phone_numbers, result)
    if config.replace_numbers is not None:
        result = NUMBER_PATTERN.sub(config.replace_numbers, result)
    return result


def _apply_normalization_options(text: str, config: NormalizationSection) -> str:
    """Aplica desempacotamento de hashtags, colapso de repetições, emoji e espaços.

    Parameters
    ----------
    text : str
        Texto já com a PII redigida.
    config : NormalizationSection
        Seção ``normalization`` de ``configs/preprocessing.yaml``.

    Returns
    -------
    str
        Texto com as opções de normalização aplicadas.
    """
    result = text
    if config.unpack_hashtags:
        result = HASHTAG_PATTERN.sub(r"\1", result)

    if config.collapse_repeated_chars:
        result = collapse_repeated_chars(result, config.collapse_repeated_chars)

    if config.demojize:
        result = EMOJI_PATTERN.sub(" ", result)

    if config.collapse_whitespace:
        result = WHITESPACE_PATTERN.sub(" ", result)

    return result


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

    result = _strip_and_lowercase(text, config)
    tokens = _filter_tokens(result, config, stopwords)

    return " ".join(tokens)


def _strip_and_lowercase(text: str, config: CleaningSection) -> str:
    """Aplica minúsculas e remove emoji, acentos e pontuação, conforme a configuração.

    Parameters
    ----------
    text : str
        Texto já normalizado.
    config : CleaningSection
        Seção ``cleaning`` de ``configs/preprocessing.yaml``.

    Returns
    -------
    str
        Texto reduzido, antes da tokenização.
    """
    result = text.lower() if config.lowercase else text

    if config.remove_emojis:
        result = EMOJI_PATTERN.sub(" ", result)
    if config.remove_accents:
        result = strip_accents(result)
    if config.remove_punctuation:
        result = PUNCTUATION_PATTERN.sub(" ", result)

    return result


def _filter_tokens(text: str, config: CleaningSection, stopwords: frozenset[str]) -> list[str]:
    """Tokeniza por espaço e filtra tokens curtos e stopwords.

    A lista de stopwords é comparada com e sem acento: em rede social, "não"
    e "nao" aparecem com frequência semelhante. A whitelist é preservada
    mesmo estando na lista de remoção (ver docstring de :func:`clean_text`).

    Parameters
    ----------
    text : str
        Texto já reduzido por :func:`_strip_and_lowercase`.
    config : CleaningSection
        Seção ``cleaning`` de ``configs/preprocessing.yaml``.
    stopwords : frozenset of str
        Stopwords a remover.

    Returns
    -------
    list of str
        Tokens filtrados.
    """
    whitelist = {term.lower() for term in config.stopwords_whitelist}
    normalized_stopwords = frozenset(strip_accents(stopword) for stopword in stopwords)
    tokens: list[str] = []
    for token in WHITESPACE_PATTERN.split(text):
        if _is_token_too_short(token, config):
            continue
        if _is_removable_stopword(token, config, normalized_stopwords, whitelist):
            continue
        tokens.append(token)

    return tokens


def _is_token_too_short(token: str, config: CleaningSection) -> bool:
    """Verifica se o token é vazio ou menor que o comprimento mínimo configurado."""
    return not token or len(token) < config.min_token_length


def _is_removable_stopword(
    token: str,
    config: CleaningSection,
    normalized_stopwords: frozenset[str],
    whitelist: set[str],
) -> bool:
    """Verifica se o token é uma stopword removível (fora da whitelist).

    A comparação ignora acentos dos dois lados: em rede social, "não" e "nao"
    aparecem com frequência semelhante.

    Parameters
    ----------
    normalized_stopwords : frozenset of str
        Stopwords já sem acento (ver :func:`_filter_tokens`).
    """
    return bool(
        config.remove_stopwords
        and token not in whitelist
        and strip_accents(token) in normalized_stopwords
    )


# --- Versões vetorizadas (polars), usadas por preprocessing.pipeline -------
#
# Reescrevem normalize_text/clean_text como expressões pl.Expr.str.* /
# pl.Expr.list.eval em vez de map_elements (UDF Python por tweet). A única
# exceção é o colapso de repetições de caractere (ver
# :func:`collapse_repeated_chars`): o padrão regex correspondente usa
# referência retroativa, que o motor Rust do polars não suporta, então essa
# etapa isolada continua via map_elements em
# ``preprocessing.pipeline.apply_text_processing``.


def normalize_text_expr(expr: pl.Expr, config: NormalizationSection) -> pl.Expr:
    """Versão vetorizada de :func:`normalize_text`, até o desempacotamento de hashtags.

    Cobre unicode, remoção de retweet/caracteres de controle, redação de PII
    e desempacotamento de hashtags — tudo que antecede o colapso de
    repetições na ordem original de :func:`normalize_text`. O restante
    (colapso de repetições, demojização, colapso de espaços, strip) é
    aplicado por :func:`finish_normalize_text_expr`, depois do
    ``map_elements`` isolado para o colapso de repetições — ver
    ``preprocessing.pipeline.apply_text_processing`` para a ordem completa.

    Parameters
    ----------
    expr : pl.Expr
        Expressão da coluna de texto bruto.
    config : NormalizationSection
        Seção ``normalization`` de ``configs/preprocessing.yaml``.

    Returns
    -------
    pl.Expr
        Expressão do texto parcialmente normalizado.
    """
    result = expr.fill_null("").str.normalize(config.unicode_form)
    result = result.str.replace_all(RETWEET_PATTERN.pattern, "", literal=False)

    if config.strip_control_chars:
        result = result.str.replace_all(CONTROL_CHARS_PATTERN.pattern, " ", literal=False)

    result = _pii_redaction_expr(result, config)

    if config.unpack_hashtags:
        result = result.str.replace_all(HASHTAG_PATTERN.pattern, r"${1}", literal=False)

    return result


def _pii_redaction_expr(expr: pl.Expr, config: NormalizationSection) -> pl.Expr:
    """Versão vetorizada de :func:`_apply_pii_redaction`.

    E-mail antes de menção, telefone antes de número (mesma ordem).
    """
    result = expr
    if config.replace_emails is not None:
        result = result.str.replace_all(EMAIL_PATTERN.pattern, config.replace_emails, literal=False)
    if config.replace_urls is not None:
        result = result.str.replace_all(URL_PATTERN.pattern, config.replace_urls, literal=False)
    if config.replace_mentions is not None:
        result = result.str.replace_all(
            MENTION_PATTERN.pattern, config.replace_mentions, literal=False
        )
    if config.replace_phone_numbers is not None:
        result = result.str.replace_all(
            PHONE_PATTERN.pattern, config.replace_phone_numbers, literal=False
        )
    if config.replace_numbers is not None:
        result = result.str.replace_all(
            NUMBER_PATTERN.pattern, config.replace_numbers, literal=False
        )
    return result


def finish_normalize_text_expr(expr: pl.Expr, config: NormalizationSection) -> pl.Expr:
    """Conclui :func:`normalize_text_expr` após o colapso de repetições.

    Aplica demojização, colapso de espaços e o strip final — mesma ordem de
    :func:`_apply_normalization_options`, na parte que roda depois do
    ``map_elements`` de :func:`collapse_repeated_chars`.

    Parameters
    ----------
    expr : pl.Expr
        Expressão da coluna já com as repetições colapsadas.
    config : NormalizationSection
        Seção ``normalization`` de ``configs/preprocessing.yaml``.

    Returns
    -------
    pl.Expr
        Expressão do texto normalizado, pronta para gravação.
    """
    result = expr
    if config.demojize:
        result = result.str.replace_all(EMOJI_PATTERN.pattern, " ", literal=False)
    if config.collapse_whitespace:
        result = result.str.replace_all(WHITESPACE_PATTERN.pattern, " ", literal=False)
    return result.str.strip_chars()


def clean_text_expr(expr: pl.Expr, config: CleaningSection, stopwords: frozenset[str]) -> pl.Expr:
    """Versão vetorizada de :func:`clean_text`.

    Reescreve a redução léxica (minúsculas, emoji, acentos, pontuação) e a
    filtragem de tokens (comprimento mínimo, whitelist, stopwords sem
    acento) como expressões ``pl.Expr.str.*``/``pl.Expr.list.eval``, sem UDF
    por tweet.

    Parameters
    ----------
    expr : pl.Expr
        Expressão da coluna de texto já normalizado.
    config : CleaningSection
        Seção ``cleaning`` de ``configs/preprocessing.yaml``.
    stopwords : frozenset of str
        Stopwords a remover.

    Returns
    -------
    pl.Expr
        Expressão do texto limpo, com tokens separados por espaço.
    """
    result = expr.fill_null("")
    if config.lowercase:
        result = result.str.to_lowercase()
    if config.remove_emojis:
        result = result.str.replace_all(EMOJI_PATTERN.pattern, " ", literal=False)
    if config.remove_accents:
        result = _strip_accents_expr(result)
    if config.remove_punctuation:
        result = result.str.replace_all(PUNCTUATION_PATTERN.pattern, " ", literal=False)

    tokens = result.str.extract_all(r"\S+")
    keep = pl.element().str.len_chars() >= config.min_token_length

    if config.remove_stopwords and stopwords:
        whitelist = [term.lower() for term in config.stopwords_whitelist]
        normalized_stopwords = [strip_accents(stopword) for stopword in stopwords]
        removable = ~pl.element().is_in(whitelist) & _strip_accents_expr(pl.element()).is_in(
            normalized_stopwords
        )
        keep = keep & ~removable

    return tokens.list.eval(pl.element().filter(keep)).list.join(" ")


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
