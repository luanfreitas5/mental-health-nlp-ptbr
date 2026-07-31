"""Modelos tradicionais de aprendizado de máquina sobre atributos tabulares.

Cobre o baseline (``dummy``, ``tfidf_logistic``) e a família tradicional da
comparação principal (XGBoost) e da extensão exploratória (regressão
logística, random forest, LightGBM).

O baseline ``dummy`` é obrigatório e roda sempre: sem uma referência trivial,
não há como afirmar que a complexidade adicional dos Transformers e do modelo
híbrido trouxe ganho real — um F1 de 0,72 pode ser excelente ou pode ser o
que a distribuição das classes já entrega sozinha.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from config.logging import get_logger
from exceptions.model import MissingDependencyError, UnknownModelError
from models.base import BaseUserClassifier, UserDataset

logger = get_logger(__name__)


def build_estimator(estimator: str, params: dict[str, Any], n_classes: int) -> BaseEstimator:
    """Instancia um estimador do scikit-learn a partir do nome configurado.

    Parameters
    ----------
    estimator : str
        Nome do estimador (``dummy``, ``logistic_regression``, ``xgboost``, ...).
    params : dict
        Hiperparâmetros do modelo.
    n_classes : int
        Número de classes (necessário ao XGBoost multiclasse).

    Returns
    -------
    BaseEstimator
        Estimador não treinado.

    Raises
    ------
    UnknownModelError
        Se o nome não estiver registrado.
    MissingDependencyError
        Se a biblioteca do modelo não estiver instalada.

    Examples
    --------
    >>> type(build_estimator("dummy", {"strategy": "stratified"}, 3)).__name__
    'DummyClassifier'
    """
    if estimator == "dummy":
        return DummyClassifier(**params)

    if estimator == "logistic_regression":
        return LogisticRegression(**params)

    if estimator == "random_forest":
        return RandomForestClassifier(**params)

    if estimator == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise MissingDependencyError(
                "xgboost não está instalado. Rode 'uv sync --dev'."
            ) from error
        return XGBClassifier(num_class=n_classes, **params)

    if estimator == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise MissingDependencyError(
                "lightgbm não está instalado. Rode 'uv sync --dev'."
            ) from error
        return LGBMClassifier(**params)

    raise UnknownModelError(
        f"Estimador tabular desconhecido: '{estimator}'. "
        "Registrados: dummy, logistic_regression, random_forest, xgboost, lightgbm."
    )


@dataclass
class TabularClassifier(BaseUserClassifier):
    """Classificador sobre a matriz de atributos por usuário.

    Parameters
    ----------
    name : str
        Nome do modelo.
    params : dict
        Hiperparâmetros; a chave ``estimator`` seleciona o algoritmo.
    scaling : {'standard', 'robust', 'none'}, optional
        Escalonamento aplicado dentro do ``Pipeline``, by default ``'standard'``.

    Notes
    -----
    O escalonamento vive **dentro** do ``Pipeline``: ajustado no ``fit``, ele
    vê apenas o treino em cada fold da validação cruzada. Escalonar antes de
    particionar vazaria média e desvio do teste para o treino.

    Examples
    --------
    >>> modelo = TabularClassifier(name="xgboost", params={"estimator": "xgboost"})
    >>> modelo.fit(treino).predict(teste)  # doctest: +SKIP
    """

    scaling: str = "standard"
    estimator_name: str = "xgboost"
    pipeline_: Pipeline | None = field(default=None, init=False, repr=False)
    feature_names_: list[str] = field(default_factory=list, init=False, repr=False)

    def _build_pipeline(self, n_classes: int) -> Pipeline:
        """Monta o ``Pipeline`` de escalonamento + estimador."""
        params = {key: value for key, value in self.params.items() if key != "estimator"}
        estimator = build_estimator(self.estimator_name, params, n_classes)

        steps: list[tuple[str, Any]] = []
        # Modelos baseados em árvore são invariantes a transformações
        # monotônicas: escaloná-los custa tempo e não muda nada.
        tree_based = self.estimator_name in {"xgboost", "lightgbm", "random_forest", "dummy"}
        if self.scaling != "none" and not tree_based:
            scaler = StandardScaler() if self.scaling == "standard" else RobustScaler()
            steps.append(("scaler", scaler))

        steps.append(("model", estimator))
        return Pipeline(steps)

    def fit(self, dataset: UserDataset) -> TabularClassifier:
        """Treina o modelo sobre a matriz de atributos.

        Parameters
        ----------
        dataset : UserDataset
            Conjunto de treino, com rótulos.

        Returns
        -------
        TabularClassifier
            O próprio modelo, treinado.

        Examples
        --------
        >>> modelo.fit(treino)  # doctest: +SKIP
        """
        self.validate_dataset(dataset)
        assert dataset.labels is not None

        self.feature_names_ = list(dataset.feature_names)
        self.pipeline_ = self._build_pipeline(n_classes=len(self.classes))
        self.pipeline_.fit(dataset.features, dataset.labels)
        self.is_fitted = True

        logger.info(
            "Modelo '%s' treinado: %d usuários × %d atributos.",
            self.name,
            len(dataset),
            dataset.features.shape[1],
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

        probabilities = self.pipeline_.predict_proba(dataset.features)
        return np.asarray(probabilities, dtype=np.float64)

    def feature_importances(self) -> dict[str, float] | None:
        """Retorna a importância dos atributos, quando o estimador a expõe.

        Returns
        -------
        dict of str to float or None
            Atributo -> importância, ou ``None`` para modelos sem esse
            atributo (como o ``dummy``).

        Examples
        --------
        >>> modelo.feature_importances()  # doctest: +SKIP
        {'psy_risco_suicida_mean': 0.18, ...}
        """
        self.check_fitted()
        assert self.pipeline_ is not None

        estimator = self.pipeline_.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            # Multiclasse: a magnitude média entre as classes resume a
            # contribuição do atributo para a decisão como um todo.
            values = np.abs(np.asarray(estimator.coef_, dtype=float)).mean(axis=0)
        else:
            return None

        return dict(zip(self.feature_names_, values.tolist(), strict=False))
