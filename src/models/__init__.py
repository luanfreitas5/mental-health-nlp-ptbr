"""Modelos de classificação no nível do usuário.

Todos implementam :class:`~models.base.BaseUserClassifier` e consomem o mesmo
:class:`~models.base.UserDataset`. A interface única é o que garante uma
comparação justa entre famílias muito diferentes — sem ela, cada modelo
receberia um tratamento próprio no avaliador e as métricas deixariam de ser
comparáveis.

Escopo (``configs/model_params.yaml``): ``baseline`` roda sempre,
``comparison`` é o escopo garantido da dissertação (H1–H5) e ``exploratory``
só roda com ``--include-exploratory``.

Modules
-------
base
    :class:`BaseUserClassifier` e :class:`UserDataset`.
traditional
    :class:`TabularClassifier` — dummy, regressão logística, RF, XGBoost, LightGBM.
deep
    :class:`SequenceClassifier` — BiLSTM sobre a sequência temporal de embeddings.
transformer
    :class:`TransformerClassifier` — BERTimbau fine-tuned, agregado por usuário.
llm
    :class:`LLMClassifier` — LLM local via Ollama, zero-shot ou few-shot.
hybrid
    :class:`HybridClassifier` — embeddings (PCA) + atributos estruturados -> XGBoost.
factory
    :func:`create_models` — instancia os modelos declarados no YAML.
persistence
    :func:`save_model` e :func:`load_model`, com metadados de rastreabilidade.
"""

from models.base import BaseUserClassifier, UserDataset
from models.deep import SequenceClassifier
from models.factory import create_model, create_models
from models.hybrid import HybridClassifier
from models.llm import LLMClassifier
from models.persistence import load_metadata, load_model, save_model
from models.traditional import TabularClassifier, build_estimator
from models.transformer import TransformerClassifier, aggregate_user_probabilities

__all__ = [
    "BaseUserClassifier",
    "HybridClassifier",
    "LLMClassifier",
    "SequenceClassifier",
    "TabularClassifier",
    "TransformerClassifier",
    "UserDataset",
    "aggregate_user_probabilities",
    "build_estimator",
    "create_model",
    "create_models",
    "load_metadata",
    "load_model",
    "save_model",
]
