"""Visualizações do projeto, com tema e paleta compartilhados.

Todas as figuras usam a mesma paleta e são gravadas em ``.png`` (300 dpi) e
``.svg``. A consistência não é estética: cores diferentes para a mesma classe
entre figuras obrigam o leitor a reconsultar a legenda a cada página da
dissertação.

Modules
-------
theme
    Paleta, ``rcParams`` e :func:`save_figure`.
distributions
    Distribuição das classes, frequência de palavras, n-grams e nuvens.
evaluation_plots
    Matriz de confusão, curvas ROC e PR, calibração e comparação entre modelos.
temporal_plots
    Evolução do sentimento, ritmo circadiano e trajetória individual.
embeddings
    Projeções UMAP/t-SNE e rede de similaridade lexical.
interpretability_plots
    SHAP, importância por grupo, Ablation Study e diferença crítica.
"""

from visualization.distributions import (
    plot_class_distribution,
    plot_feature_distribution,
    plot_ngrams,
    plot_user_activity,
    plot_word_frequency,
    plot_wordcloud,
)
from visualization.embeddings import plot_embedding_projection, plot_interaction_network
from visualization.evaluation_plots import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_precision_recall_curves,
    plot_reliability_curve,
    plot_roc_curves,
    plot_slice_performance,
)
from visualization.interpretability_plots import (
    plot_ablation,
    plot_critical_difference,
    plot_feature_importance,
    plot_group_importance,
    plot_shap_summary,
)
from visualization.temporal_plots import (
    plot_activity_heatmap,
    plot_circadian_activity,
    plot_sentiment_evolution,
    plot_user_timeline,
)
from visualization.theme import apply_theme, get_class_palette, save_figure

__all__ = [
    "apply_theme",
    "get_class_palette",
    "plot_ablation",
    "plot_activity_heatmap",
    "plot_circadian_activity",
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_critical_difference",
    "plot_embedding_projection",
    "plot_feature_distribution",
    "plot_feature_importance",
    "plot_group_importance",
    "plot_interaction_network",
    "plot_model_comparison",
    "plot_ngrams",
    "plot_precision_recall_curves",
    "plot_reliability_curve",
    "plot_roc_curves",
    "plot_sentiment_evolution",
    "plot_shap_summary",
    "plot_slice_performance",
    "plot_user_activity",
    "plot_user_timeline",
    "plot_word_frequency",
    "plot_wordcloud",
    "save_figure",
]
