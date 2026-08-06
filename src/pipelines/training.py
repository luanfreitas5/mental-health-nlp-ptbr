"""Etapa 8 — validação cruzada e treinamento final dos modelos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.logging import get_logger
from data.reader import read_parquet, read_partitioned
from models.factory import create_models
from pipelines.base import PipelineStage, StageContext
from training.cross_validation import cross_validate_all, extract_fold_scores
from training.trainer import (
    load_user_sequences,
    load_user_texts,
    split_features,
    train_all,
)
from utils.files import list_files, write_json

logger = get_logger(__name__)


class TrainingStage(PipelineStage):
    """Executa a validação cruzada e treina os modelos no conjunto completo."""

    name = "train"
    description = "Treina os modelos com validação cruzada e persiste os artefatos"

    def required_inputs(self, context: StageContext) -> list[Path]:
        """Exige a matriz de atributos e a tabela de partições."""
        return [context.paths.data.user_features, context.paths.data.splits]

    def run(self, context: StageContext) -> dict[str, Any]:
        """Treina todos os modelos selecionados.

        Parameters
        ----------
        context : StageContext
            Contexto compartilhado.

        Returns
        -------
        dict
            Modelos treinados e resumo da validação cruzada.

        Examples
        --------
        >>> TrainingStage().run(contexto)  # doctest: +SKIP
        """
        config = context.config
        paths = context.paths

        features = read_parquet(paths.data.user_features)
        splits = read_parquet(paths.data.splits)

        # O conjunto de desenvolvimento (treino + validação) alimenta a
        # validação cruzada; o teste permanece intocado até a etapa 'evaluate'.
        development = features.filter(
            features["user_id"].is_in(splits.filter(splits["split"] != "test")["user_id"])
        ).sort("user_id")
        train = split_features(features, splits, "train")

        texts = None
        if list_files(paths.data.tweets_labeled, "*.parquet"):
            texts = load_user_texts(read_partitioned(paths.data.tweets_labeled, stage="label"))

        sequences = load_user_sequences(
            paths.data.embeddings, config.features.semantic.primary_model.split("/")[-1]
        )

        models = create_models(
            config,
            include_exploratory=context.option("include_exploratory", False),
            only=context.option("models"),
        )

        # --- Validação cruzada -------------------------------------------
        cv_results: dict[str, Any] = {}
        if not context.option("skip_cv", False):
            from training.trainer import build_dataset

            specs = {
                name: config.models.all_models()[name]
                for name in models
                if name in config.models.all_models()
            }
            development_dataset = build_dataset(development, texts=texts, sequences=sequences)
            cv_results = cross_validate_all(specs, development_dataset, splits, config)

            write_json(
                paths.reports.metrics / "cross_validation.json",
                {
                    name: {key: value for key, value in result.items() if key != "per_fold_metrics"}
                    for name, result in cv_results.items()
                },
            )
        else:
            logger.info("Validação cruzada pulada (--skip-cv).")

        # --- Treinamento final -------------------------------------------
        trained = train_all(
            models,
            train,
            config,
            paths.models.artifacts,
            texts=texts,
            sequences=sequences,
            tracker=context.tracker,
        )

        fold_scores = extract_fold_scores(cv_results)
        if fold_scores:
            import numpy as np

            write_json(
                paths.reports.metrics / "cv_fold_scores.json",
                {name: np.asarray(scores).tolist() for name, scores in fold_scores.items()},
            )

        return {
            "modelos_treinados": sorted(trained),
            "n_usuarios_treino": train.height,
            "validacao_cruzada": {
                name: {"mean": result["mean"], "std": result["std"]}
                for name, result in cv_results.items()
            },
        }
