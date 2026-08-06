"""Etapa 10 — figuras, model card e datasheet.

Fecha o ciclo de IA responsável: um modelo sem documentação de uso pretendido,
limitações e desempenho por subgrupo não deveria ser publicado — muito menos
num domínio em que um erro tem consequência clínica.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.logging import get_logger
from data.reader import read_parquet, read_partitioned
from pipelines.base import PipelineStage, StageContext
from reports_templates.datasheet import build_datasheet
from reports_templates.model_card import build_model_card
from utils.files import list_files, read_json, write_text
from visualization.distributions import (
    plot_class_distribution,
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
    plot_feature_importance,
    plot_group_importance,
    plot_shap_summary,
)
from visualization.temporal_plots import (
    plot_activity_heatmap,
    plot_circadian_activity,
    plot_sentiment_evolution,
)
from visualization.theme import apply_theme, save_figure

logger = get_logger(__name__)


class ReportingStage(PipelineStage):
    """Gera todas as figuras e os documentos de IA responsável."""

    name = "report"
    description = "Gera figuras, model card e datasheet"

    def required_inputs(self, context: StageContext) -> list[Path]:
        """Exige a matriz de atributos (as figuras de avaliação são opcionais)."""
        return [context.paths.data.user_features]

    def _save(self, figure: Any, name: str, context: StageContext) -> str | None:
        """Grava uma figura, tolerando as que não puderam ser geradas."""
        if figure is None:
            logger.debug(
                "Figura '%s' não gerada (dependência ausente ou dados insuficientes).", name
            )
            return None

        paths = save_figure(
            figure,
            context.paths.reports.figures,
            name,
            formats=tuple(context.config.evaluation.reporting.figure_formats),
            dpi=context.config.evaluation.reporting.figure_dpi,
        )
        return str(paths[0]) if paths else None

    def _exploratory_figures(self, context: StageContext) -> dict[str, str | None]:
        """Gera as figuras exploratórias (distribuições, palavras, temporais)."""
        paths = context.paths
        figures: dict[str, str | None] = {}

        features = read_parquet(paths.data.user_features)
        figures["distribuicao_classes"] = self._save(
            plot_class_distribution(features), "distribuicao_classes", context
        )
        figures["perfil_atividade"] = self._save(
            plot_user_activity(features), "perfil_atividade", context
        )
        figures["projecao_embeddings"] = self._save(
            plot_embedding_projection(features), "projecao_embeddings", context
        )

        if not list_files(paths.data.tweets_labeled, "*.parquet"):
            return figures

        tweets = read_partitioned(paths.data.tweets_labeled, stage="label")
        labels = read_parquet(paths.data.user_labels)

        figures["frequencia_palavras"] = self._save(
            plot_word_frequency(tweets, labels), "frequencia_palavras", context
        )
        figures["bigramas"] = self._save(plot_ngrams(tweets, labels, n=2), "bigramas", context)
        figures["nuvem_palavras"] = self._save(
            plot_wordcloud(tweets, labels), "nuvem_palavras", context
        )
        figures["evolucao_sentimento"] = self._save(
            plot_sentiment_evolution(tweets, labels), "evolucao_sentimento", context
        )
        figures["atividade_circadiana"] = self._save(
            plot_circadian_activity(
                tweets, labels, context.config.features.temporal.insomnia_window
            ),
            "atividade_circadiana",
            context,
        )
        figures["mapa_atividade"] = self._save(
            plot_activity_heatmap(tweets), "mapa_atividade", context
        )
        figures["rede_similaridade"] = self._save(
            plot_interaction_network(tweets, labels), "rede_similaridade", context
        )
        return figures

    def _evaluation_figures(self, context: StageContext) -> dict[str, str | None]:
        """Gera as figuras de avaliação a partir do JSON de métricas."""
        paths = context.paths
        figures: dict[str, str | None] = {}

        metrics_path = paths.reports.metrics / "evaluation.json"
        if not metrics_path.is_file():
            logger.warning(
                "Métricas não encontradas em %s: figuras de avaliação puladas. "
                "Execute a etapa 'evaluate'.",
                metrics_path,
            )
            return figures

        payload = read_json(metrics_path)
        models = payload.get("models", {})
        primary = payload.get("primary_metric", "f1_macro")

        comparison_path = paths.reports.tables / "model_comparison.csv"
        if comparison_path.is_file():
            comparison = pl.read_csv(comparison_path)
            figures["comparacao_modelos"] = self._save(
                plot_model_comparison(comparison, primary), "comparacao_modelos", context
            )

        for name, result in models.items():
            if result.get("confusion_matrix"):
                figures[f"matriz_confusao_{name}"] = self._save(
                    plot_confusion_matrix(
                        result["confusion_matrix"], title=f"Matriz de Confusão — {name}"
                    ),
                    f"matriz_confusao_{name}",
                    context,
                )
            if result.get("calibration"):
                figures[f"calibracao_{name}"] = self._save(
                    plot_reliability_curve(result["calibration"], name),
                    f"calibracao_{name}",
                    context,
                )
            if result.get("slices"):
                figures[f"fatias_{name}"] = self._save(
                    plot_slice_performance(result["slices"], primary), f"fatias_{name}", context
                )

        # As curvas ROC/PR exigem as probabilidades, que ficam no arquivo de
        # predições e não no JSON resumido de métricas.
        predictions_path = paths.reports.metrics / "predictions.csv"
        if predictions_path.is_file() and models:
            best = max(models, key=lambda name: models[name]["metrics"].get(primary, 0.0))
            result = models[best]
            proba = np.array(result.get("predictions", {}).get("y_proba", []))
            y_true = np.array(result.get("predictions", {}).get("y_true", []))
            if proba.size and y_true.size:
                figures["curvas_roc"] = self._save(
                    plot_roc_curves(y_true, proba, best), "curvas_roc", context
                )
                figures["curvas_precisao_revocacao"] = self._save(
                    plot_precision_recall_curves(y_true, proba, best),
                    "curvas_precisao_revocacao",
                    context,
                )

        return figures

    def _interpretability_figures(self, context: StageContext) -> dict[str, str | None]:
        """Gera as figuras de importância, SHAP e ablação."""
        paths = context.paths
        figures: dict[str, str | None] = {}

        for path in sorted(paths.reports.interpretability.glob("permutation_importance_*.csv")):
            name = path.stem.replace("permutation_importance_", "")
            figures[f"importancia_{name}"] = self._save(
                plot_feature_importance(pl.read_csv(path)), f"importancia_{name}", context
            )

        for path in sorted(paths.reports.interpretability.glob("group_importance_*.csv")):
            name = path.stem.replace("group_importance_", "")
            figures[f"importancia_grupo_{name}"] = self._save(
                plot_group_importance(pl.read_csv(path)), f"importancia_grupo_{name}", context
            )

        for path in sorted(paths.reports.interpretability.glob("shap_summary_*.csv")):
            name = path.stem.replace("shap_summary_", "")
            figures[f"shap_{name}"] = self._save(
                plot_shap_summary(pl.read_csv(path)), f"shap_{name}", context
            )

        ablation_path = paths.reports.ablation / "ablation_summary.csv"
        if ablation_path.is_file():
            figures["ablacao"] = self._save(
                plot_ablation(
                    pl.read_csv(ablation_path), context.config.evaluation.metrics.primary
                ),
                "ablacao",
                context,
            )

        return figures

    def run(self, context: StageContext) -> dict[str, Any]:
        """Gera todas as figuras e documentos.

        Parameters
        ----------
        context : StageContext
            Contexto compartilhado.

        Returns
        -------
        dict
            Figuras geradas e caminhos do model card e do datasheet.

        Examples
        --------
        >>> ReportingStage().run(contexto)  # doctest: +SKIP
        """
        apply_theme(context.config.evaluation.reporting.figure_dpi)
        paths = context.paths

        figures: dict[str, str | None] = {}
        figures.update(self._exploratory_figures(context))
        figures.update(self._evaluation_figures(context))
        figures.update(self._interpretability_figures(context))

        generated = {name: path for name, path in figures.items() if path}
        logger.info("Figuras geradas: %d de %d tentativas.", len(generated), len(figures))

        documents: dict[str, str] = {}
        if context.config.evaluation.reporting.generate_model_card:
            metrics_path = paths.reports.metrics / "evaluation.json"
            payload = read_json(metrics_path) if metrics_path.is_file() else {}

            card = build_model_card(payload, context.config, paths)
            documents["model_card"] = str(
                write_text(paths.reports.model_cards / "model_card.md", card)
            )

            datasheet = build_datasheet(context.config, paths)
            documents["datasheet"] = str(
                write_text(paths.reports.datasheets / "datasheet.md", datasheet)
            )

        if context.tracker is not None:
            for path in documents.values():
                context.tracker.log_artifact(Path(path))

        return {
            "n_figuras": len(generated),
            "figuras": sorted(generated),
            "documentos": documents,
        }
