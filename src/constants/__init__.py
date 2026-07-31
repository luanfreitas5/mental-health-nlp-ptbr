"""Constantes, enums e valores estruturais do projeto.

Distinção deliberada em relação a ``configs/``: aqui ficam apenas constantes
**estruturais** (nomes de coluna, classes, pronomes da língua, padrões regex).
Hiperparâmetros e limiares de pesquisa ficam nos YAMLs, para poderem ser
variados sem alterar código.

Modules
-------
columns
    Nomes canônicos de coluna por estágio e prefixos dos grupos de atributos.
labels
    :class:`UserLabel`, :class:`Sentiment`, :class:`FeatureGroup` e mapeamentos.
metrics
    Nomes das métricas, orientação de otimização e nomes de exibição.
regex
    Padrões compilados de PII, normalização e tokenização.
defaults
    Etapas do pipeline, pronomes do pt-BR e limites operacionais.
"""

from constants.defaults import (
    NEGATION_TERMS,
    PIPELINE_STAGES,
    PRONOUN_GROUPS,
    RANDOM_SEED,
    STAGE_DEPENDENCIES,
)
from constants.labels import (
    CLASS_DISPLAY_NAMES,
    CLASS_ORDER,
    INDEX_TO_LABEL,
    LABEL_TO_INDEX,
    RISK_CLASSES,
    Emotion,
    FeatureGroup,
    PsychologicalDimension,
    Sentiment,
    Split,
    UserLabel,
)
from constants.metrics import DEFAULT_METRICS, METRIC_DISPLAY_NAMES, is_higher_better

__all__ = [
    "CLASS_DISPLAY_NAMES",
    "CLASS_ORDER",
    "DEFAULT_METRICS",
    "INDEX_TO_LABEL",
    "LABEL_TO_INDEX",
    "METRIC_DISPLAY_NAMES",
    "NEGATION_TERMS",
    "PIPELINE_STAGES",
    "PRONOUN_GROUPS",
    "RANDOM_SEED",
    "RISK_CLASSES",
    "STAGE_DEPENDENCIES",
    "Emotion",
    "FeatureGroup",
    "PsychologicalDimension",
    "Sentiment",
    "Split",
    "UserLabel",
    "is_higher_better",
]
