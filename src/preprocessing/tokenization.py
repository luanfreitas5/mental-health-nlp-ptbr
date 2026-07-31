"""Tokenização e lematização com spaCy, e fallback por expressão regular.

O modelo ``pt_core_news_sm`` do spaCy exige um download separado
(``python -m spacy download pt_core_news_sm``), que não cabe em
``pyproject.toml``. Em vez de quebrar o pipeline quando ele falta, o módulo
cai num tokenizador por regex e **avisa** que a lematização foi desativada:
degradar a qualidade das features é aceitável, interromper uma execução de
horas por causa de uma dependência opcional não é.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from config.logging import get_logger
from config.settings import TokenizationSection
from preprocessing.text import tokenize as regex_tokenize

logger = get_logger(__name__)


@lru_cache(maxsize=2)
def load_spacy_model(model_name: str) -> Any | None:
    """Carrega um modelo do spaCy, devolvendo ``None`` se indisponível.

    Só os componentes necessários à lematização são mantidos: o *parser* e o
    NER custam a maior parte do tempo de processamento e não são usados aqui.

    Parameters
    ----------
    model_name : str
        Nome do modelo (ex.: ``"pt_core_news_sm"``).

    Returns
    -------
    spacy.Language or None
        Modelo carregado, ou ``None`` se o spaCy ou o modelo não estiverem
        instalados.

    Examples
    --------
    >>> load_spacy_model("modelo_inexistente") is None
    True
    """
    try:
        import spacy
    except ImportError:
        logger.warning(
            "spaCy não está instalado: a lematização será desativada e a tokenização "
            "usará o fallback por regex."
        )
        return None

    try:
        return spacy.load(model_name, disable=["parser", "ner", "textcat"])
    except OSError:
        logger.warning(
            "Modelo spaCy '%s' não encontrado. Rode 'python -m spacy download %s' para "
            "habilitar a lematização. Usando fallback por regex.",
            model_name,
            model_name,
        )
        return None


class Tokenizer:
    """Tokenizador com lematização opcional.

    Parameters
    ----------
    config : TokenizationSection
        Seção ``tokenization`` de ``configs/preprocessing.yaml``.

    Attributes
    ----------
    uses_spacy : bool
        ``True`` se o spaCy foi carregado com sucesso.

    Examples
    --------
    >>> from config.settings import TokenizationSection
    >>> tokenizer = Tokenizer(TokenizationSection(backend="regex"))
    >>> tokenizer.tokenize("não estou bem")
    ['não', 'estou', 'bem']
    """

    def __init__(self, config: TokenizationSection) -> None:
        self.config = config
        self._nlp = load_spacy_model(config.spacy_model) if config.backend == "spacy" else None

    @property
    def uses_spacy(self) -> bool:
        """Indica se o backend do spaCy está ativo."""
        return self._nlp is not None

    def tokenize(self, text: str) -> list[str]:
        """Tokeniza um único texto.

        Parameters
        ----------
        text : str
            Texto de entrada.

        Returns
        -------
        list of str
            Tokens (lematizados quando o spaCy está disponível e
            ``lemmatize`` está ativo).

        Examples
        --------
        >>> Tokenizer(TokenizationSection(backend="regex")).tokenize("dias difíceis")
        ['dias', 'difíceis']
        """
        if self._nlp is None:
            return regex_tokenize(text)

        document = self._nlp(text)
        if self.config.lemmatize:
            return [token.lemma_.lower() for token in document if not token.is_space]
        return [token.text for token in document if not token.is_space]

    def tokenize_batch(self, texts: list[str]) -> list[list[str]]:
        """Tokeniza uma coleção de textos.

        Com spaCy, usa ``nlp.pipe`` — processar em lote é uma ordem de
        grandeza mais rápido que chamar o modelo texto a texto, diferença que
        importa em milhões de tweets.

        Parameters
        ----------
        texts : list of str
            Textos a tokenizar.

        Returns
        -------
        list of list of str
            Tokens de cada texto, na mesma ordem da entrada.

        Examples
        --------
        >>> Tokenizer(TokenizationSection(backend="regex")).tokenize_batch(["a b", "c"])
        [['a', 'b'], ['c']]
        """
        if self._nlp is None:
            return [regex_tokenize(text) for text in texts]

        documents = self._nlp.pipe(
            texts,
            batch_size=self.config.batch_size,
            n_process=self.config.n_process,
        )

        if self.config.lemmatize:
            return [
                [token.lemma_.lower() for token in document if not token.is_space]
                for document in documents
            ]
        return [[token.text for token in document if not token.is_space] for document in documents]
