"""Etapa 6 — construção da matriz de atributos por usuário."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.logging import get_logger
from config.settings import FeaturesConfig
from constants.columns import USER_ID
from data.catalog import write_dataset_manifest
from data.reader import (
    list_collected_users,
    read_parquet,
    read_partitioned,
    read_user_partition,
    select_pending_users,
)
from data.writer import write_parquet, write_user_partition
from exceptions.data import InsufficientDataError
from features.builder import build_user_features_raw, finalize_user_features
from features.semantic import aggregate_embeddings
from pipelines.base import PipelineStage, StageContext
from schemas.features import list_feature_columns, validate_feature_matrix
from utils.files import list_files, write_json
from utils.hashing import hash_dataframe
from utils.parallel import resolve_worker_count, run_user_pool
from utils.validation import summarize_missing

logger = get_logger(__name__)


def _build_and_write_user_row(
    user_id: str,
    tweets_dir: Path,
    metadata: pl.DataFrame | None,
    scores_dir: Path | None,
    config: FeaturesConfig,
    features_raw_dir: Path,
) -> str | None:
    """Monta e grava a linha de atributos brutos de um único usuário — roda em processo worker.

    Função de nível de módulo (não um método), condição necessária para ser
    "picklable" e distribuível via ``ProcessPoolExecutor`` (ver
    :func:`utils.parallel.run_user_pool`).

    Parameters
    ----------
    user_id : str
        Identificador pseudonimizado do usuário a processar.
    tweets_dir : Path
        Diretório dos tweets rotulados, particionado por usuário.
    metadata : pl.DataFrame, optional
        Metadados públicos já restritos a este usuário (uma linha, ou
        ``None``), extraídos no processo principal antes da distribuição.
    scores_dir : Path, optional
        Diretório dos vetores psicológicos, particionado por usuário.
    config : FeaturesConfig
        Seção ``features`` de ``configs/features.yaml``.
    features_raw_dir : Path
        Diretório de destino de ``user_features_raw/``.

    Returns
    -------
    str or None
        O próprio ``user_id`` se algo foi gravado, ``None`` se o usuário não
        tinha tweets (nada a fazer).

    Examples
    --------
    >>> _build_and_write_user_row(
    ...     "u_a", tweets_dir, None, None, config.features, features_raw_dir
    ... )  # doctest: +SKIP
    """
    user_tweets = read_user_partition(tweets_dir, user_id)
    if user_tweets.is_empty():
        return None

    user_scores = None
    if scores_dir is not None:
        candidate = read_user_partition(scores_dir, user_id)
        user_scores = candidate if not candidate.is_empty() else None

    raw_row = build_user_features_raw(
        user_tweets,
        config,
        metadata=metadata,
        psychological_scores=user_scores,
    )
    write_user_partition(raw_row, features_raw_dir, user_id)
    return user_id


class FeaturesStage(PipelineStage):
    """Agrega tweets em uma linha por usuário, com todos os grupos de atributos.

    A construção dos seis grupos processa um usuário por vez e grava
    ``user_features_raw/`` imediatamente após cada um (retomável, limitável
    por ``--limit-users``). A imputação de valores ausentes por mediana e a
    junção com os rótulos rodam uma única vez, no final, sobre o acumulado —
    dependem da população inteira, não de um usuário isolado (ver
    :func:`features.builder.finalize_user_features`).
    """

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

        labels = read_parquet(paths.data.user_labels)
        metadata, scores_dir = self._load_optional_inputs(paths)

        available = list_collected_users(paths.data.tweets_labeled)
        already_processed = list_collected_users(paths.data.user_features_raw)
        pending = select_pending_users(available, already_processed, context.option("limit_users"))
        logger.info(
            "Matriz de atributos: %d usuários já processados, %d pendentes nesta execução.",
            len(already_processed),
            len(pending),
        )

        if pending:
            self._build_pending_user_features(
                paths.data.tweets_labeled,
                metadata,
                scores_dir,
                pending,
                config,
                paths,
                resolve_worker_count(context.option("workers")),
            )

        processed_users = list_collected_users(paths.data.user_features_raw)
        if not processed_users:
            raise InsufficientDataError(
                "Nenhum usuário foi processado ainda em "
                f"'{paths.data.user_features_raw}'. Execute a etapa 'features' novamente "
                "(sem --limit-users, ou com um limite maior)."
            )

        raw = read_partitioned(paths.data.user_features_raw)
        features = self._finalize_features(raw, config, labels, context)

        validate_feature_matrix(features, allow_nan=False)
        target = write_parquet(features, paths.data.user_features)

        by_group = self._write_feature_reports(features, config, paths)

        return {
            "n_usuarios": features.height,
            "n_atributos": len(list_feature_columns(features)),
            "atributos_por_grupo": by_group,
            "written": str(target),
        }

    def _load_optional_inputs(self, paths: Any) -> tuple[pl.DataFrame | None, Path | None]:
        """Carrega a metadata e localiza os vetores psicológicos opcionais, avisando se faltarem.

        A metadata é um artefato único e pequeno (uma linha por usuário) e por
        isso é lida por inteiro; os vetores psicológicos, por serem
        particionados por usuário, ficam como um diretório — cada usuário lê
        só o próprio arquivo dentro de :func:`_build_and_write_user_row`.
        """
        metadata = (
            read_parquet(paths.data.user_metadata) if paths.data.user_metadata.is_file() else None
        )
        scores_dir = (
            paths.data.psychological_scores
            if list_files(paths.data.psychological_scores, "*.parquet")
            else None
        )
        if scores_dir is None:
            logger.warning(
                "Vetores psicológicos ausentes: o grupo 'psychological' será omitido. "
                "Execute a etapa 'psych' para incluí-lo."
            )
        return metadata, scores_dir

    def _build_pending_user_features(
        self,
        tweets_dir: Path,
        metadata: pl.DataFrame | None,
        scores_dir: Path | None,
        pending: list[str],
        config: Any,
        paths: Any,
        max_workers: int,
    ) -> None:
        """Constrói e grava, em processos paralelos, a linha de atributos brutos dos pendentes.

        Cada usuário é independente dos demais (mesmos seis grupos de
        :func:`features.builder.build_user_features_raw`, sem estado
        compartilhado), então o laço é distribuído entre processos via
        :func:`utils.parallel.run_user_pool`. A metadata é particionada uma
        única vez aqui — cada worker recebe só a fatia (uma linha, ou
        ``None``) do usuário que vai processar, não o DataFrame inteiro.
        """
        metadata_groups = (
            metadata.partition_by(USER_ID, as_dict=True, maintain_order=True)
            if metadata is not None
            else {}
        )

        jobs = {
            user_id: partial(
                _build_and_write_user_row,
                user_id,
                tweets_dir,
                metadata_groups.get((user_id,)),
                scores_dir,
                config.features,
                paths.data.user_features_raw,
            )
            for user_id in pending
        }
        run_user_pool(
            jobs,
            description="Construindo atributos por usuário",
            max_workers=max_workers,
        )

    def _finalize_features(
        self, raw: pl.DataFrame, config: Any, labels: pl.DataFrame, context: StageContext
    ) -> pl.DataFrame:
        """Finaliza a matriz (imputação, rótulos) e junta os embeddings semânticos, se existirem.

        Os embeddings são unidos aqui (e não dentro do builder) porque vêm de
        um artefato separado, produzido pela etapa `embed`.
        """
        features = finalize_user_features(raw, config.features, labels=labels)
        semantic = self._load_semantic(context)
        if semantic is not None:
            features = features.join(semantic, on="user_id", how="left").fill_null(0.0)
        return features

    def _write_feature_reports(
        self, features: pl.DataFrame, config: Any, paths: Any
    ) -> dict[str, int]:
        """Grava o resumo de atributos e o manifesto do dataset; loga a contagem por grupo."""
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
        return by_group
