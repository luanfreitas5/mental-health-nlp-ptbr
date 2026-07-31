"""Etapa 9 — avaliação no conjunto de teste, testes estatísticos e ablação.

O conjunto de teste é tocado **uma única vez**, aqui. Ajustar qualquer coisa
depois de olhar o resultado do teste — hiperparâmetro, limiar, escolha de
modelo — transformaria a métrica reportada em métrica de validação disfarçada,
e a estimativa de generalização deixaria de valer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.logging import get_logger
from constants.labels import LABEL_TO_INDEX
from data.reader import read_parquet
from evaluation.ablation import run_ablation, summarize_ablation
from evaluation.evaluator import Evaluator
from evaluation.reports import save_reports
from evaluation.statistics import compare_all_models, mcnemar_test
from interpretability.importance import (
    aggregate_importance_by_group,
    compute_permutation_importance,
)
from interpretability.shap_values import compute_shap_values, save_shap_summary, summarize_shap
from models.persistence import load_model
from pipelines.base import PipelineStage, StageContext
from training.trainer import build_dataset, load_user_sequences, load_user_texts, split_features
from utils.files import read_json, write_json

logger = get_logger(__name__)


class EvaluationStage(PipelineStage):
    """Avalia os modelos treinados e produz os relatórios comparativos."""

    name = "evaluate"
    description = "Avalia no teste, compara modelos estatisticamente e roda a ablação"

    def required_inputs(self, context: StageContext) -> list[Path]:
        """Exige a matriz, as partições e o diretório de modelos treinados."""
        return [
            context.paths.data.user_features,
            context.paths.data.splits,
            context.paths.models.artifacts,
        ]

    def run(self, context: StageContext) -> dict[str, Any]:
        """Executa a avaliação completa.

        Parameters
        ----------
        context : StageContext
            Contexto compartilhado.

        Returns
        -------
        dict
            Métricas por modelo, resultado dos testes e caminhos dos relatórios.

        Examples
        --------
        >>> EvaluationStage().run(contexto)  # doctest: +SKIP
        """
        config = context.config
        paths = context.paths

        features = read_parquet(paths.data.user_features)
        splits = read_parquet(paths.data.splits)

        train = split_features(features, splits, "train")
        test = split_features(features, splits, "test")
        if test.is_empty():
            logger.error("Conjunto de teste vazio: nada a avaliar.")
            return {"skipped": True, "reason": "conjunto de teste vazio"}

        texts = (
            load_user_texts(read_parquet(paths.data.tweets_labeled))
            if paths.data.tweets_labeled.is_file()
            else None
        )
        sequences = load_user_sequences(
            paths.data.embeddings, config.features.semantic.primary_model.split("/")[-1]
        )

        evaluator = Evaluator(config)

        model_files = sorted(paths.models.artifacts.glob("*.joblib"))
        if not model_files:
            logger.error("Nenhum modelo treinado em %s.", paths.models.artifacts)
            return {"skipped": True, "reason": "nenhum modelo treinado"}

        results = self._evaluate_models(model_files, test, config, evaluator, texts, sequences)
        if not results:
            return {"skipped": True, "reason": "nenhum modelo pôde ser avaliado"}

        comparison = evaluator.compare(results)
        statistics = self._run_statistics(results, config, paths)
        ablation_summary = self._run_ablation_study(train, test, config, context, paths)

        # --- Interpretabilidade --------------------------------------------
        interpretability = self._run_interpretability(results, test, texts, sequences, context)

        written = save_reports(results, comparison, config, paths, statistics, ablation_summary)

        if context.tracker is not None:
            for path in written.values():
                context.tracker.log_artifact(Path(path))

        primary = config.evaluation.metrics.primary
        best = max(results, key=lambda name: results[name].metrics.get(primary, float("-inf")))
        logger.info("Melhor modelo: %s | %s", best, results[best].headline(primary))

        return {
            "modelos_avaliados": sorted(results),
            "melhor_modelo": best,
            "metrica_principal": primary,
            "resultados": {name: result.metrics.get(primary) for name, result in results.items()},
            "interpretabilidade": interpretability,
            "written": {key: str(value) for key, value in written.items()},
        }

    def _evaluate_models(
        self,
        model_files: list[Path],
        test: pl.DataFrame,
        config: Any,
        evaluator: Evaluator,
        texts: dict[str, list[str]] | None,
        sequences: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Avalia cada modelo treinado no conjunto de teste, isolando falhas por modelo."""
        results: dict[str, Any] = {}
        for model_file in model_files:
            try:
                model = load_model(model_file)
                spec = config.models.all_models().get(model.name)
                dataset = build_dataset(
                    test,
                    feature_groups=spec.feature_groups if spec else None,
                    texts=texts,
                    sequences=sequences,
                )
                results[model.name] = evaluator.evaluate(model, dataset, profile=test)
            except (ValueError, RuntimeError, OSError, KeyError):
                logger.exception("Avaliação de '%s' falhou.", model_file.stem)

        return results

    def _run_statistics(self, results: dict[str, Any], config: Any, paths: Any) -> dict[str, Any]:
        """Roda Wilcoxon/Friedman (se houver scores por fold) e o McNemar par a par."""
        statistics: dict[str, Any] = {}
        fold_scores_path = paths.reports.metrics / "cv_fold_scores.json"
        if fold_scores_path.is_file():
            fold_scores = {
                name: np.array(scores)
                for name, scores in read_json(fold_scores_path).items()
                if name in results
            }
            if len(fold_scores) >= 2:
                statistics = compare_all_models(fold_scores, config.evaluation.statistics)
        else:
            logger.info("Scores por fold ausentes: Wilcoxon e Friedman não executados.")

        if config.evaluation.statistics.mcnemar.enabled and len(results) >= 2:
            statistics["mcnemar"] = self._run_mcnemar(results, config)

        return statistics

    def _run_ablation_study(
        self,
        train: pl.DataFrame,
        test: pl.DataFrame,
        config: Any,
        context: StageContext,
        paths: Any,
    ) -> pl.DataFrame:
        """Roda o Ablation Study, se habilitado, sem interromper a avaliação em caso de falha."""
        ablation_summary = pl.DataFrame()
        if config.evaluation.ablation.enabled and not context.option("skip_ablation", False):
            try:
                ablation = run_ablation(
                    train, test, config, config.evaluation.ablation, LABEL_TO_INDEX
                )
                if ablation:
                    ablation_summary = summarize_ablation(ablation)
                    write_json(paths.reports.ablation / "ablation.json", ablation)
            except (KeyError, ValueError, RuntimeError):
                logger.exception("Ablation Study falhou.")

        return ablation_summary

    def _run_mcnemar(self, results: dict[str, Any], config: Any) -> dict[str, Any]:
        """Compara cada modelo com o melhor, no mesmo conjunto de teste."""
        primary = config.evaluation.metrics.primary
        best = max(results, key=lambda name: results[name].metrics.get(primary, float("-inf")))
        reference = results[best]

        comparisons: dict[str, Any] = {}
        for name, result in results.items():
            if name == best:
                continue
            test = mcnemar_test(
                np.array(reference.predictions["y_true"]),
                np.array(reference.predictions["y_pred"]),
                np.array(result.predictions["y_pred"]),
                correction=bool(config.evaluation.statistics.mcnemar.correction),
                alpha=config.evaluation.statistics.alpha,
            )
            comparisons[f"{best}_vs_{name}"] = test.__dict__

        return comparisons

    def _run_interpretability(
        self,
        results: dict[str, Any],
        test: pl.DataFrame,
        texts: dict[str, list[str]] | None,
        sequences: dict[str, Any] | None,
        context: StageContext,
    ) -> dict[str, Any]:
        """Calcula importância e SHAP para o melhor modelo tabular."""
        config = context.config
        paths = context.paths
        primary = config.evaluation.metrics.primary

        # A interpretabilidade roda sobre o melhor modelo que expõe um
        # pipeline scikit-learn: SHAP e permutação não se aplicam ao LLM nem
        # às redes recorrentes com a mesma semântica.
        candidates = {
            name: result
            for name, result in results.items()
            if (paths.models.artifacts / f"{name}.joblib").is_file()
        }
        if not candidates:
            return {}

        best = max(
            candidates, key=lambda name: candidates[name].metrics.get(primary, float("-inf"))
        )
        try:
            model = load_model(paths.models.artifacts / f"{best}.joblib")
            spec = config.models.all_models().get(best)
            dataset = build_dataset(
                test,
                feature_groups=spec.feature_groups if spec else None,
                texts=texts,
                sequences=sequences,
            )
        except (OSError, ValueError, KeyError) as error:
            logger.warning("Interpretabilidade pulada: %s", error)
            return {}

        summary: dict[str, Any] = {"modelo": best}

        settings = config.evaluation.interpretability
        if settings.permutation_importance.enabled:
            importance = compute_permutation_importance(
                model, dataset, settings.permutation_importance
            )
            if not importance.is_empty():
                path = paths.reports.interpretability / f"permutation_importance_{best}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                importance.write_csv(path)

                grouped = aggregate_importance_by_group(importance)
                if not grouped.is_empty():
                    grouped.write_csv(
                        paths.reports.interpretability / f"group_importance_{best}.csv"
                    )
                    summary["importancia_por_grupo"] = grouped.to_dicts()

        if settings.shap.enabled:
            shap_result = compute_shap_values(model, dataset, settings.shap)
            shap_summary = summarize_shap(shap_result, settings.shap.max_display)
            saved = save_shap_summary(shap_summary, paths.reports.interpretability, best)
            if saved is not None:
                summary["shap"] = str(saved)

        return summary
