"""Modelo híbrido — a principal contribuição metodológica da proposta.

Arquitetura::

    Tweets -> Transformer -> Embeddings ─┐
    Atributos emocionais ────────────────┤
    Atributos temporais ─────────────────┼─> Concatenação -> XGBoost -> Classe
    Atributos comportamentais ───────────┤
    Vetor psicológico (LLM) ─────────────┘

A diferença em relação ao :class:`~models.traditional.TabularClassifier` é o
tratamento assimétrico dos blocos de atributos: os embeddings passam por PCA
antes da concatenação, enquanto os atributos estruturados entram inteiros.

O motivo é o desequilíbrio de dimensionalidade. Com ~1.500 dimensões de
embedding contra algumas dezenas de atributos estruturados, as árvores
escolheriam quase sempre alguma dimensão semântica — não por ser mais
informativa, mas por haver muito mais candidatas. O PCA equilibra os blocos e
é o que permite ao Ablation Study medir a contribuição real de cada grupo
(H2, H3, H4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.logging import get_logger
from constants.columns import SEMANTIC_PREFIX
from models.base import BaseUserClassifier, UserDataset
from models.traditional import build_estimator

logger = get_logger(__name__)


def split_feature_blocks(feature_names: list[str]) -> tuple[list[int], list[int]]:
    """Separa os índices das colunas semânticas dos demais atributos.

    Parameters
    ----------
    feature_names : list of str
        Nomes das colunas da matriz.

    Returns
    -------
    tuple
        ``(índices_semânticos, índices_estruturados)``.

    Examples
    --------
    >>> split_feature_blocks(["sem_mean_000", "emo_polarity_mean"])
    ([0], [1])
    """
    semantic = [
        index for index, name in enumerate(feature_names) if name.startswith(SEMANTIC_PREFIX)
    ]
    structured = [
        index for index, name in enumerate(feature_names) if not name.startswith(SEMANTIC_PREFIX)
    ]
    return semantic, structured


@dataclass
class HybridClassifier(BaseUserClassifier):
    """Classificador híbrido: embeddings reduzidos + atributos estruturados.

    Parameters
    ----------
    name : str
        Nome do modelo.
    params : dict
        Hiperparâmetros da cabeça (``head``, além dos parâmetros do XGBoost).
    n_components : int, optional
        Dimensões mantidas pelo PCA sobre os embeddings, by default 64.

    Examples
    --------
    >>> modelo = HybridClassifier(name="hybrid_xgboost", params={"head": "xgboost"})
    >>> modelo.fit(treino)  # doctest: +SKIP
    """

    n_components: int = 64
    random_state: int = 42
    pipeline_: Pipeline | None = field(default=None, init=False, repr=False)
    feature_names_: list[str] = field(default_factory=list, init=False, repr=False)
    n_semantic_: int = field(default=0, init=False, repr=False)
    present_classes_: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int), init=False, repr=False
    )

    def _build_pipeline(self, feature_names: list[str], n_classes: int) -> Pipeline:
        """Monta o pré-processador por blocos seguido da cabeça de classificação."""
        semantic, structured = split_feature_blocks(feature_names)
        self.n_semantic_ = len(semantic)

        transformers: list[tuple[str, Any, list[int]]] = []

        if semantic:
            # n_components nunca pode exceder o número de colunas disponíveis;
            # o mínimo evita falha quando o grupo semântico está desativado.
            components = min(self.n_components, len(semantic))
            transformers.append(
                (
                    "semantic",
                    Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            ("pca", PCA(n_components=components, random_state=self.random_state)),
                        ]
                    ),
                    semantic,
                )
            )
            logger.info(
                "Bloco semântico: %d dimensões reduzidas para %d componentes.",
                len(semantic),
                components,
            )

        if structured:
            transformers.append(("structured", StandardScaler(), structured))
            logger.info("Bloco estruturado: %d atributos mantidos.", len(structured))

        head_params = {key: value for key, value in self.params.items() if key != "head"}
        head = build_estimator(str(self.params.get("head", "xgboost")), head_params, n_classes)

        return Pipeline(
            [
                ("blocks", ColumnTransformer(transformers, remainder="drop")),
                ("model", head),
            ]
        )

    def fit(self, dataset: UserDataset) -> HybridClassifier:
        """Treina o modelo híbrido.

        Parameters
        ----------
        dataset : UserDataset
            Conjunto de treino, com rótulos.

        Returns
        -------
        HybridClassifier
            O próprio modelo, treinado.

        Examples
        --------
        >>> modelo.fit(treino)  # doctest: +SKIP
        """
        self.validate_dataset(dataset)
        assert dataset.labels is not None

        # Mesmo racional do TabularClassifier: um fold pode não conter
        # nenhum usuário de uma classe, e o XGBoost exige rótulos contíguos
        # a partir de 0. Ver models.traditional.TabularClassifier.fit.
        self.feature_names_ = list(dataset.feature_names)
        self.present_classes_ = np.unique(dataset.labels)
        encoded_labels = np.searchsorted(self.present_classes_, dataset.labels)

        self.pipeline_ = self._build_pipeline(self.feature_names_, len(self.present_classes_))
        self.pipeline_.fit(dataset.features, encoded_labels)
        self.is_fitted = True

        logger.info(
            "Modelo híbrido '%s' treinado: %d usuários, %d atributos "
            "(%d semânticos + %d estruturados).",
            self.name,
            len(dataset),
            dataset.features.shape[1],
            self.n_semantic_,
            dataset.features.shape[1] - self.n_semantic_,
        )
        return self

    def predict_proba(self, dataset: UserDataset) -> np.ndarray:
        """Prevê as probabilidades de cada classe.

        Parameters
        ----------
        dataset : UserDataset
            Conjunto a classificar.

        Returns
        -------
        np.ndarray
            Matriz ``(n_usuarios, n_classes)``.

        Examples
        --------
        >>> modelo.predict_proba(teste).shape  # doctest: +SKIP
        (120, 3)
        """
        self.check_fitted()
        assert self.pipeline_ is not None

        local_probabilities = np.asarray(
            self.pipeline_.predict_proba(dataset.features), dtype=np.float64
        )
        probabilities = np.zeros((local_probabilities.shape[0], len(self.classes)))
        probabilities[:, self.present_classes_] = local_probabilities
        return probabilities

    def transformed_feature_names(self) -> list[str]:
        """Nomes das colunas após a transformação por blocos.

        Necessário para interpretar o SHAP: as dimensões do PCA não têm
        correspondência direta com as colunas originais, e apresentá-las com
        os nomes originais induziria a leitura errada.

        Returns
        -------
        list of str
            Nomes das componentes do PCA seguidos dos atributos estruturados.

        Examples
        --------
        >>> modelo.transformed_feature_names()[:2]  # doctest: +SKIP
        ['sem_pca_000', 'sem_pca_001']
        """
        self.check_fitted()
        semantic, structured = split_feature_blocks(self.feature_names_)

        names = [f"sem_pca_{index:03d}" for index in range(min(self.n_components, len(semantic)))]
        names.extend(self.feature_names_[index] for index in structured)
        return names
