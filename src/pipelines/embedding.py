"""Etapa 5 — geração dos embeddings semânticos por tweet.

Separada da construção de features de propósito: gerar embeddings é a etapa
mais cara em GPU do pipeline, e persistir os vetores por tweet permite
reaproveitá-los em todas as agregações e em todos os modelos seguintes sem
recodificar milhões de textos a cada experimento.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.logging import get_logger
from constants.columns import TEXT_NORMALIZED, USER_ID
from data.reader import read_partitioned, select_pending_users
from exceptions.model import MissingDependencyError, ModelError
from features.semantic import EmbeddingEncoder, save_embeddings
from pipelines.base import PipelineStage, StageContext
from utils.files import list_files
from utils.progress import track

logger = get_logger(__name__)


class EmbeddingStage(PipelineStage):
    """Gera e persiste os embeddings de cada tweet.

    Cada usuário é codificado e gravado imediatamente num cache
    ``embeddings/_cache/<modelo>/<user_id>.npy`` (retomável, limitável por
    ``--limit-users``); ao final, o cache acumulado é reunido no par
    ``<modelo>.npy`` + ``<modelo>_index.parquet`` que as demais etapas
    (``features``, ``train``) já esperam — o formato final não muda, só como
    ele é construído.
    """

    name = "embed"
    description = "Gera embeddings semânticos dos tweets com encoder Transformer"

    def required_inputs(self, context: StageContext) -> list[Path]:
        """Exige os tweets limpos da etapa de pré-processamento."""
        return [context.paths.data.tweets_clean]

    def run(self, context: StageContext) -> dict[str, Any]:
        """Codifica os tweets e grava os vetores em disco.

        Parameters
        ----------
        context : StageContext
            Contexto compartilhado.

        Returns
        -------
        dict
            Modelos codificados, dimensões e caminhos gravados.

        Examples
        --------
        >>> EmbeddingStage().run(contexto)  # doctest: +SKIP
        """
        config = context.config.features.semantic
        paths = context.paths

        if not config.enabled:
            logger.info("Geração de embeddings desativada em configs/features.yaml.")
            return {"skipped": True, "reason": "desativada na configuração"}

        labeled_available = bool(list_files(paths.data.tweets_labeled, "*.parquet"))
        source = paths.data.tweets_labeled if labeled_available else paths.data.tweets_clean
        tweets = read_partitioned(source, stage="label" if labeled_available else "preprocess")
        all_users = set(tweets[USER_ID].unique().to_list())

        # Por padrão só o encoder principal roda; os demais entram na extensão
        # exploratória e custam uma passada completa cada.
        requested: dict[str, str] = {config.primary_model.split("/")[-1]: config.primary_model}
        if context.option("all_encoders", False):
            requested.update(config.models)

        limit = context.option("limit_users")
        written: dict[str, str] = {}
        dimensions: dict[str, int] = {}
        n_tweets_codificados = 0

        for name, model_name in requested.items():
            cache_dir = paths.data.embeddings / "_cache" / name
            cache_dir.mkdir(parents=True, exist_ok=True)
            already_processed = {file.stem for file in cache_dir.glob("*.npy")}
            pending = select_pending_users(all_users, already_processed, limit)
            logger.info(
                "Embeddings '%s': %d usuários já codificados, %d pendentes nesta execução.",
                name,
                len(already_processed),
                len(pending),
            )

            if pending:
                try:
                    encoder = EmbeddingEncoder(model_name, config)
                except (MissingDependencyError, ModelError) as error:
                    logger.warning("Encoder '%s' pulado: %s", name, error)
                    continue

                groups = tweets.filter(pl.col(USER_ID).is_in(pending)).partition_by(
                    USER_ID, as_dict=True, maintain_order=True
                )
                for user_id in track(pending, f"Codificando usuários ({name})"):
                    user_frame = groups.get((user_id,))
                    if user_frame is None or user_frame.is_empty():
                        continue
                    try:
                        user_embeddings = encoder.encode(user_frame[TEXT_NORMALIZED].to_list())
                    except (MissingDependencyError, ModelError) as error:
                        logger.warning("Encoder '%s' pulado: %s", name, error)
                        break
                    np.save(cache_dir / f"{user_id}.npy", user_embeddings)

            cached_users = sorted(file.stem for file in cache_dir.glob("*.npy"))
            if not cached_users:
                continue

            arrays = [np.load(cache_dir / f"{user_id}.npy") for user_id in cached_users]
            embeddings = np.vstack(arrays)
            owners = [
                user_id
                for user_id, array in zip(cached_users, arrays, strict=True)
                for _ in range(array.shape[0])
            ]

            path = save_embeddings(embeddings, owners, paths.data.embeddings, name)
            written[name] = str(path)
            dimensions[name] = int(embeddings.shape[1])
            n_tweets_codificados = max(n_tweets_codificados, embeddings.shape[0])

        if not written:
            logger.warning(
                "Nenhum embedding gerado: o grupo 'semantic' e os modelos sequenciais "
                "ficarão indisponíveis."
            )
            return {"skipped": True, "reason": "nenhum encoder pôde ser carregado"}

        return {
            "n_tweets_codificados": n_tweets_codificados,
            "modelos": sorted(written),
            "dimensoes": dimensions,
            "written": written,
        }
