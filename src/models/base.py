"""Contrato comum de todos os classificadores do projeto.

A comparação principal reúne famílias muito diferentes — árvores sobre
atributos tabulares, uma rede recorrente sobre sequências de embeddings, um
Transformer fine-tuned e um LLM por prompt. Sem uma interface única, o
avaliador precisaria de um ramo condicional por modelo, e qualquer diferença
de tratamento entre eles tornaria a comparação injusta.

:class:`UserDataset` carrega as várias representações do mesmo conjunto de
usuários (matriz tabular, textos, sequências de embeddings) para que cada
modelo consuma a sua sem que o pipeline precise saber quem consome o quê.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config.logging import get_logger
from constants.labels import CLASS_ORDER
from exceptions.model import ModelNotFittedError

logger = get_logger(__name__)


@dataclass
class UserDataset:
    """Conjunto de usuários em todas as representações usadas pelos modelos.

    Attributes
    ----------
    user_ids : list of str
        Identificadores pseudonimizados, na ordem das linhas.
    features : np.ndarray
        Matriz tabular ``(n_usuarios, n_atributos)``.
    feature_names : list of str
        Nomes das colunas de ``features``.
    labels : np.ndarray, optional
        Rótulos codificados como inteiros; ``None`` na predição.
    texts : dict, optional
        ``user_id -> lista de tweets`` (Transformers e LLM).
    sequences : dict, optional
        ``user_id -> matriz (n_tweets, dim)`` de embeddings (modelos recorrentes).
    """

    user_ids: list[str]
    features: np.ndarray
    feature_names: list[str]
    labels: np.ndarray | None = None
    texts: dict[str, list[str]] | None = None
    sequences: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        """Número de usuários no conjunto."""
        return len(self.user_ids)

    @property
    def has_labels(self) -> bool:
        """Indica se o conjunto possui rótulos."""
        return self.labels is not None

    def require_texts(self) -> dict[str, list[str]]:
        """Retorna os textos, falhando com mensagem clara se ausentes.

        Returns
        -------
        dict of str to list of str
            Tweets por usuário.

        Raises
        ------
        ValueError
            Se o conjunto não contiver textos.
        """
        if self.texts is None:
            raise ValueError(
                "Este modelo exige os textos dos usuários, mas o conjunto foi montado sem "
                "eles. Verifique a construção do UserDataset em pipelines.training."
            )
        return self.texts

    def require_sequences(self) -> dict[str, np.ndarray]:
        """Retorna as sequências de embeddings, falhando se ausentes.

        Returns
        -------
        dict of str to np.ndarray
            Sequência de vetores por usuário.

        Raises
        ------
        ValueError
            Se o conjunto não contiver sequências.
        """
        if self.sequences is None:
            raise ValueError(
                "Este modelo exige sequências de embeddings por usuário. Execute a etapa "
                "'embed' para gerá-las."
            )
        return self.sequences


@dataclass
class BaseUserClassifier(ABC):
    """Classe base de todos os classificadores no nível do usuário.

    Attributes
    ----------
    name : str
        Nome do modelo, usado em relatórios e no MLflow.
    params : dict
        Hiperparâmetros vindos de ``configs/model_params.yaml``.
    classes : list of str
        Ordem canônica das classes (fixa a ordem das colunas de probabilidade).
    is_fitted : bool
        Se o modelo já foi treinado.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    classes: list[str] = field(default_factory=lambda: list(CLASS_ORDER))
    is_fitted: bool = field(default=False, init=False)

    #: Recursos exigidos pelo modelo, verificados antes do treino.
    requires_text: bool = field(default=False, init=False)
    requires_sequences: bool = field(default=False, init=False)

    @abstractmethod
    def fit(self, dataset: UserDataset) -> BaseUserClassifier:
        """Treina o modelo.

        Parameters
        ----------
        dataset : UserDataset
            Conjunto de treino, com rótulos.

        Returns
        -------
        BaseUserClassifier
            O próprio modelo, treinado.
        """

    @abstractmethod
    def predict_proba(self, dataset: UserDataset) -> np.ndarray:
        """Prevê as probabilidades de cada classe.

        Parameters
        ----------
        dataset : UserDataset
            Conjunto a classificar.

        Returns
        -------
        np.ndarray
            Matriz ``(n_usuarios, n_classes)``, com colunas na ordem de
            :attr:`classes`.
        """

    def predict(self, dataset: UserDataset) -> np.ndarray:
        """Prevê a classe mais provável de cada usuário.

        Parameters
        ----------
        dataset : UserDataset
            Conjunto a classificar.

        Returns
        -------
        np.ndarray
            Índices das classes previstas.

        Examples
        --------
        >>> modelo.predict(conjunto_teste)  # doctest: +SKIP
        array([0, 2, 1])
        """
        return np.asarray(self.predict_proba(dataset)).argmax(axis=1)

    def check_fitted(self) -> None:
        """Garante que o modelo já foi treinado.

        Raises
        ------
        ModelNotFittedError
            Se ``fit`` ainda não tiver sido chamado.
        """
        if not self.is_fitted:
            raise ModelNotFittedError(
                f"O modelo '{self.name}' ainda não foi treinado. Chame fit() antes de prever."
            )

    def validate_dataset(self, dataset: UserDataset, *, require_labels: bool = True) -> None:
        """Valida se o conjunto atende aos requisitos do modelo.

        Parameters
        ----------
        dataset : UserDataset
            Conjunto a validar.
        require_labels : bool, optional
            Exige a presença de rótulos, by default True.

        Raises
        ------
        ValueError
            Se faltarem rótulos, textos ou sequências exigidos pelo modelo.
        """
        if require_labels and not dataset.has_labels:
            raise ValueError(f"O modelo '{self.name}' exige rótulos para treinar.")
        if self.requires_text:
            dataset.require_texts()
        if self.requires_sequences:
            dataset.require_sequences()

    def describe(self) -> dict[str, Any]:
        """Descreve o modelo para registro no MLflow e no model card.

        Returns
        -------
        dict
            Nome, tipo, classes e hiperparâmetros.

        Examples
        --------
        >>> modelo.describe()["name"]  # doctest: +SKIP
        'xgboost'
        """
        return {
            "name": self.name,
            "type": type(self).__name__,
            "classes": self.classes.copy(),
            "params": self.params.copy(),
        }
