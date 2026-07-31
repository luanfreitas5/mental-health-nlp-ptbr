"""Vetorização de n-grams no nível do usuário (TF-IDF).

Os n-grams **não** entram na matriz de atributos pré-calculada, ao contrário
dos demais grupos. O motivo é evitar vazamento: o vocabulário e os pesos IDF
precisam ser estimados apenas na partição de treino. Se o ``TfidfVectorizer``
fosse ajustado sobre o dataset inteiro antes da divisão, termos que só
aparecem no teste influenciariam a representação — um vazamento sutil, que
não aparece em nenhuma checagem de código e infla a métrica.

Por isso este módulo expõe um transformador compatível com o scikit-learn,
usado **dentro** do ``Pipeline``, onde o ``fit`` vê apenas o treino.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from config.logging import get_logger
from config.settings import NgramsSection
from constants.columns import TEXT_CLEAN, USER_ID
from utils.validation import require_columns

logger = get_logger(__name__)


def build_user_documents(tweets: pl.DataFrame) -> pl.DataFrame:
    """Concatena os tweets de cada usuário num único documento.

    A abordagem centrada no usuário exige um documento por pessoa: o TF-IDF
    passa a descrever o vocabulário do *perfil*, e não de uma publicação
    isolada.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos, com ``user_id`` e ``text_clean``.

    Returns
    -------
    pl.DataFrame
        Colunas ``user_id`` e ``document``, ordenadas por ``user_id``.

    Examples
    --------
    >>> frame = pl.DataFrame({"user_id": ["u_a", "u_a"], "text_clean": ["oi", "tudo bem"]})
    >>> build_user_documents(frame)["document"][0]
    'oi tudo bem'
    """
    require_columns(tweets, [USER_ID, TEXT_CLEAN], context="documentos por usuário")

    return (
        tweets.group_by(USER_ID)
        .agg(pl.col(TEXT_CLEAN).str.join(" ").alias("document"))
        .sort(USER_ID)
    )


class UserNgramVectorizer(BaseEstimator, TransformerMixin):
    """Vetorizador TF-IDF de documentos por usuário, compatível com scikit-learn.

    Parameters
    ----------
    config : NgramsSection
        Seção ``linguistic.ngrams`` de ``configs/features.yaml``.

    Attributes
    ----------
    vectorizer_ : TfidfVectorizer or CountVectorizer
        Vetorizador ajustado (disponível após o ``fit``).

    Examples
    --------
    >>> from config.settings import NgramsSection
    >>> vectorizer = UserNgramVectorizer(NgramsSection(min_df=1))
    >>> matriz = vectorizer.fit_transform(["eu não estou bem", "dia bom hoje"])
    >>> matriz.shape[0]
    2
    """

    def __init__(self, config: NgramsSection) -> None:
        self.config = config
        self.vectorizer_: TfidfVectorizer | CountVectorizer | None = None

    def fit(self, X: list[str], y: object = None) -> UserNgramVectorizer:  # noqa: N803, ARG002
        """Ajusta o vocabulário e os pesos IDF **apenas** com os documentos vistos.

        Parameters
        ----------
        X : list of str
            Documentos de treino (um por usuário).
        y : object, optional
            Ignorado; presente para compatibilidade com a API do scikit-learn.

        Returns
        -------
        UserNgramVectorizer
            O próprio objeto, ajustado.
        """
        factory = TfidfVectorizer if self.config.vectorizer == "tfidf" else CountVectorizer
        kwargs = {
            "ngram_range": tuple(self.config.ngram_range),
            "max_features": self.config.max_features,
            "min_df": self.config.min_df,
            "max_df": self.config.max_df,
        }
        if self.config.vectorizer == "tfidf":
            kwargs["sublinear_tf"] = self.config.sublinear_tf

        self.vectorizer_ = factory(**kwargs)
        self.vectorizer_.fit(X)

        logger.info(
            "Vocabulário de n-grams ajustado: %d termos (ngram_range=%s).",
            len(self.vectorizer_.vocabulary_),
            tuple(self.config.ngram_range),
        )
        return self

    def transform(self, X: list[str]) -> np.ndarray:  # noqa: N803
        """Transforma documentos na matriz de n-grams.

        Parameters
        ----------
        X : list of str
            Documentos a transformar.

        Returns
        -------
        np.ndarray
            Matriz densa ``(n_documentos, n_termos)``.

        Raises
        ------
        RuntimeError
            Se ``fit`` ainda não tiver sido chamado.
        """
        if self.vectorizer_ is None:
            raise RuntimeError("UserNgramVectorizer.transform chamado antes de fit.")
        # Os stubs do scikit-learn não fixam o tipo de retorno de `.transform`
        # para a união `TfidfVectorizer | CountVectorizer`; em tempo de
        # execução é sempre uma matriz esparsa, que suporta `.todense()`.
        sparse_matrix = self.vectorizer_.transform(X)
        return np.asarray(sparse_matrix.todense())  # pyright: ignore[reportAttributeAccessIssue]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:  # noqa: ARG002
        """Retorna os nomes dos termos do vocabulário.

        Returns
        -------
        np.ndarray
            Nomes prefixados com ``ling_ngram_``, coerentes com a convenção
            de prefixos dos grupos de atributos.

        Raises
        ------
        RuntimeError
            Se ``fit`` ainda não tiver sido chamado.
        """
        if self.vectorizer_ is None:
            raise RuntimeError("get_feature_names_out chamado antes de fit.")
        return np.array([f"ling_ngram_{name}" for name in self.vectorizer_.get_feature_names_out()])
