"""Etapa 6 — construção da matriz de atributos por usuário."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.logging import get_logger
from data.catalog import write_dataset_manifest
from data.reader import read_parquet, read_partitioned
from data.writer import write_parquet
from features.builder import build_user_features
from features.semantic import aggregate_embeddings
from pipelines.base import PipelineStage, StageContext
from schemas.features import list_feature_columns, validate_feature_matrix
from utils.files import write_json
from utils.hashing import hash_dataframe
from utils.validation import summarize_missing

logger = get_logger(__name__)


class FeaturesStage(PipelineStage):
    """Agrega tweets em uma linha por usuário, com todos os grupos de atributos."""

    name = "features"
    description = "Constrói a matriz de atributos por usuário (6 grupos)"

    def required_inputs(self, context: StageContext) -> list[Path]:
        """Exige os tweets rotulados e os rótulos por usuário."""
        return [context.paths.data.tweets_labeled, context.paths.data.user_labels]

    def _load_semantic(self, context: StageContext) -> pl.DataFrame | None:
        """Carrega e agrega os embeddings gravados pela etapa ``embed``."""
        config = context.config.features.semantic
        name = config.primary_model.split("/")[-1]

        array_path = context.paths.data.embeddings / f"{name}.npy"
        index_path = context.paths.data.embeddings / f"{name}_index.parquet"

        if not array_path.is_file() or not index_path.is_file():
            logger.warning(
                "Embeddings não encontrados para '%s': o grupo 'semantic' será omitido. "
                "Execute a etapa 'embed'.",
                name,
            )
            return None

        embeddings = np.load(array_path)
        index = read_parquet(index_path)
        return aggregate_embeddings(
            embeddings, index["user_id"].to_list(), config.user_aggregations
        )

    def run(self, context: StageContext) -> dict[str, Any]:
        """Monta e persiste a matriz de atributos.

        Parameters
        ----------
        context : StageContext
            Contexto compartilhado.

        Returns
        -------
        dict
            Dimensões da matriz, atributos por grupo e caminho gravado.

        Examples
        --------
        >>> FeaturesStage().run(contexto)  # doctest: +SKIP
        """
        config = context.config
        paths = context.paths

        tweets = read_partitioned(paths.data.tweets_labeled, stage="label")
        labels = read_parquet(paths.data.user_labels)

        metadata = (
            read_parquet(paths.data.user_metadata) if paths.data.user_metadata.is_file() else None
        )
        scores = (
            read_parquet(paths.data.psychological_scores)
            if paths.data.psychological_scores.is_file()
            else None
        )
        if scores is None:
            logger.warning(
                "Vetores psicológicos ausentes: o grupo 'psychological' será omitido. "
                "Execute a etapa 'psych' para incluí-lo."
            )

        features = build_user_features(
            tweets,
            config.features,
            metadata=metadata,
            psychological_scores=scores,
            labels=labels,
        )

        # Os embeddings são unidos aqui (e não dentro do builder) porque vêm de
        # um artefato separado, produzido pela etapa `embed`.
        semantic = self._load_semantic(context)
        if semantic is not None:
            features = features.join(semantic, on="user_id", how="left").fill_null(0.0)

        validate_feature_matrix(features, allow_nan=False)
        target = write_parquet(features, paths.data.user_features)

        by_group = {
            group: len(list_feature_columns(features, [group])) for group in config.features.groups
        }
        missing_summary = summarize_missing(features)

        write_json(
            paths.reports.metrics / "features_summary.json",
            {
                "n_usuarios": features.height,
                "n_atributos": len(list_feature_columns(features)),
                "atributos_por_grupo": by_group,
                "hash_matriz": hash_dataframe(features),
                "colunas_com_ausentes": missing_summary.filter(pl.col("n_missing") > 0).to_dicts()[
                    :20
                ],
            },
        )
        write_dataset_manifest(paths, {"n_usuarios": features.height})

        logger.info("Atributos por grupo: %s", by_group)
        return {
            "n_usuarios": features.height,
            "n_atributos": len(list_feature_columns(features)),
            "atributos_por_grupo": by_group,
            "written": str(target),
        }
