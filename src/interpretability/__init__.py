"""Interpretabilidade do modelo final.

Num sistema de triagem de saúde mental, um modelo que não pode ser explicado
não é utilizável: a decisão precisa ser auditável por quem vai agir sobre ela.
A análise dos fatores associados aos sinais de risco também é uma das
contribuições científicas declaradas na proposta.

Modules
-------
importance
    Importância por permutação (sem o viés de cardinalidade das árvores),
    importância nativa e agregação por grupo de atributos.
shap_values
    Valores SHAP do modelo final e resumo por atributo.
"""

from interpretability.importance import (
    aggregate_importance_by_group,
    compute_permutation_importance,
    extract_model_importance,
    resolve_feature_group,
    top_features,
)
from interpretability.shap_values import compute_shap_values, save_shap_summary, summarize_shap

__all__ = [
    "aggregate_importance_by_group",
    "compute_permutation_importance",
    "compute_shap_values",
    "extract_model_importance",
    "resolve_feature_group",
    "save_shap_summary",
    "summarize_shap",
    "top_features",
]
