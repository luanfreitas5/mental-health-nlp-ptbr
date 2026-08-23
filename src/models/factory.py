"""Fábrica de modelos a partir das especificações de ``model_params.yaml``.

Padrão *Factory*: adicionar um modelo novo à comparação passa a ser uma
entrada no YAML mais um registro aqui, sem tocar no pipeline de treinamento
nem no de avaliação. É o que mantém a extensão exploratória (LSTM, CNN,
RoBERTa, DeBERTa, Gemma, Mistral) a custo marginal quase nulo.
"""

from __future__ import annotations

from collections.abc import Callable

from config.logging import get_logger
from config.settings import Config, ModelSpec
from constants.labels import CLASS_ORDER
from exceptions.model import UnknownModelError
from models.base import BaseUserClassifier
from models.deep import SequenceClassifier
from models.hybrid import HybridClassifier
from models.llm import LLMClassifier
from models.traditional import TabularClassifier
from models.transformer import TransformerClassifier

logger = get_logger(__name__)

#: Estimadores tabulares atendidos por :class:`TabularClassifier`.
TABULAR_ESTIMATORS: frozenset[str] = frozenset(
    {"dummy", "logistic_regression", "random_forest", "xgboost", "lightgbm"}
)

#: Estimadores sequenciais (recorrentes/convolucionais) atendidos por :class:`SequenceClassifier`.
SEQUENCE_ESTIMATORS: frozenset[str] = frozenset({"bilstm", "lstm", "cnn_text"})

_ModelBuilder = Callable[[str, ModelSpec, Config, list[str]], BaseUserClassifier]


def _build_tabular_model(
    name: str, spec: ModelSpec, config: Config, classes: list[str]
) -> BaseUserClassifier:
    """Constrói um modelo tabular (dummy, regressão logística, árvores, boosting)."""
    return TabularClassifier(
        name=name,
        params=dict(spec.params),
        classes=classes,
        scaling=config.features.scaling.method,
        estimator_name=spec.estimator,
    )


def _build_sequence_model(
    name: str, spec: ModelSpec, config: Config, classes: list[str]
) -> BaseUserClassifier:
    """Constrói um modelo sequencial (BiLSTM, LSTM, CNN de texto)."""
    del config
    return SequenceClassifier(
        name=name, params=dict(spec.params), classes=classes, estimator_name=spec.estimator
    )


def _build_transformer_model(
    name: str, spec: ModelSpec, config: Config, classes: list[str]
) -> BaseUserClassifier:
    """Constrói um modelo transformer (ex.: BERTimbau)."""
    del config
    return TransformerClassifier(name=name, params=dict(spec.params), classes=classes)


def _build_llm_model(
    name: str, spec: ModelSpec, config: Config, classes: list[str]
) -> BaseUserClassifier:
    """Constrói um classificador baseado em LLM."""
    return LLMClassifier(
        name=name, params=dict(spec.params), classes=classes, llm_config=config.llm
    )


def _build_hybrid_model(
    name: str, spec: ModelSpec, config: Config, classes: list[str]
) -> BaseUserClassifier:
    """Constrói um modelo híbrido (embeddings + redução de dimensionalidade + estimador)."""
    return HybridClassifier(
        name=name,
        params=dict(spec.params),
        classes=classes,
        n_components=config.features.semantic.reduction.n_components,
        random_state=config.random_seed,
    )


#: Registro estimador -> função construtora, usado por :func:`create_model`.
_MODEL_BUILDERS: dict[str, _ModelBuilder] = {
    **dict.fromkeys(TABULAR_ESTIMATORS, _build_tabular_model),
    **dict.fromkeys(SEQUENCE_ESTIMATORS, _build_sequence_model),
    "transformer": _build_transformer_model,
    "llm": _build_llm_model,
    "hybrid": _build_hybrid_model,
}


def create_model(name: str, spec: ModelSpec, config: Config) -> BaseUserClassifier:
    """Instancia um modelo a partir da sua especificação.

    Parameters
    ----------
    name : str
        Nome do modelo (chave em ``configs/model_params.yaml``).
    spec : ModelSpec
        Especificação validada (escopo, estimador, hiperparâmetros).
    config : Config
        Configuração completa do projeto.

    Returns
    -------
    BaseUserClassifier
        Modelo pronto para treinar.

    Raises
    ------
    UnknownModelError
        Se o estimador não estiver registrado.

    Examples
    --------
    >>> modelo = create_model("xgboost", spec, config)  # doctest: +SKIP
    >>> modelo.name  # doctest: +SKIP
    'xgboost'
    """
    classes = list(CLASS_ORDER)

    builder = _MODEL_BUILDERS.get(spec.estimator)
    if builder is None:
        raise UnknownModelError(
            f"Estimador desconhecido: '{spec.estimator}' (modelo '{name}'). "
            "Registrados: tabulares, bilstm/lstm/cnn_text, transformer, llm, hybrid."
        )
    model = builder(name, spec, config, classes)

    logger.debug("Modelo '%s' criado (estimador=%s, escopo=%s).", name, spec.estimator, spec.scope)
    return model


def _select_only(available: dict[str, ModelSpec], only: list[str]) -> dict[str, ModelSpec]:
    """Restringe as especificações disponíveis aos nomes pedidos em ``only``.

    Parameters
    ----------
    available : dict of str to ModelSpec
        Todas as especificações declaradas em ``configs/model_params.yaml``.
    only : list of str
        Nomes pedidos explicitamente (``--models``).

    Returns
    -------
    dict of str to ModelSpec
        Subconjunto de ``available`` restrito a ``only``.

    Raises
    ------
    UnknownModelError
        Se algum nome pedido não existir na configuração.
    """
    unknown = [name for name in only if name not in available]
    if unknown:
        raise UnknownModelError(
            f"Modelos não declarados em model_params.yaml: {unknown}. "
            f"Disponíveis: {sorted(available)}"
        )
    return {name: available[name] for name in only}


def create_models(
    config: Config,
    *,
    include_exploratory: bool = False,
    only: list[str] | None = None,
) -> dict[str, BaseUserClassifier]:
    """Instancia todos os modelos selecionados para a execução.

    Parameters
    ----------
    config : Config
        Configuração completa do projeto.
    include_exploratory : bool, optional
        Inclui a extensão exploratória, by default False.
    only : list of str, optional
        Restringe a execução a modelos específicos (``--models``).

    Returns
    -------
    dict of str to BaseUserClassifier
        Nome -> modelo pronto para treinar.

    Raises
    ------
    UnknownModelError
        Se algum nome pedido em ``only`` não existir na configuração.

    Examples
    --------
    >>> modelos = create_models(config)  # doctest: +SKIP
    >>> sorted(modelos)  # doctest: +SKIP
    ['bertimbau', 'bilstm', 'dummy', 'hybrid_xgboost', ...]
    """
    selected = config.models.select(include_exploratory=include_exploratory)

    if only:
        selected = _select_only(config.models.all_models(), only)

    models = {name: create_model(name, spec, config) for name, spec in selected.items()}
    logger.info("Modelos selecionados (%d): %s.", len(models), ", ".join(sorted(models)))
    return models
