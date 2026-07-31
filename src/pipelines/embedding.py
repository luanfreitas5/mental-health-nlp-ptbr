"""Etapa 5 — geração dos embeddings semânticos por tweet.

Separada da construção de features de propósito: gerar embeddings é a etapa
mais cara em GPU do pipeline, e persistir os vetores por tweet permite
reaproveitá-los em todas as agregações e em todos os modelos seguintes sem
recodificar milhões de textos a cada experimento.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.logging import get_logger
from data.reader import read_parquet
from exceptions.model import MissingDependencyError, ModelError
from features.semantic import EmbeddingEncoder, save_embeddings
from pipelines.base import PipelineStage, StageContext

logger = get_logger(__name__)


class EmbeddingStage(PipelineStage):
    """Gera e persiste os embeddings de cada tweet."""

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

        source = (
            paths.data.tweets_labeled
            if paths.data.tweets_labeled.is_file()
            else paths.data.tweets_clean
        )
        tweets = read_parquet(source).sort(["user_id", "created_at"])

        # Por padrão só o encoder principal roda; os demais entram na extensão
        # exploratória e custam uma passada completa cada.
        requested: dict[str, str] = {config.primary_model.split("/")[-1]: config.primary_model}
        if context.option("all_encoders", False):
            requested.update(config.models)

        written: dict[str, str] = {}
        dimensions: dict[str, int] = {}

        for name, model_name in requested.items():
            try:
                encoder = EmbeddingEncoder(model_name, config)
                embeddings = encoder.encode(tweets["text_normalized"].to_list())
            except (MissingDependencyError, ModelError) as error:
                logger.warning("Encoder '%s' pulado: %s", name, error)
                continue

            path = save_embeddings(
                embeddings, tweets["user_id"].to_list(), paths.data.embeddings, name
            )
            written[name] = str(path)
            dimensions[name] = int(embeddings.shape[1])

        if not written:
            logger.warning(
                "Nenhum embedding gerado: o grupo 'semantic' e os modelos sequenciais "
                "ficarão indisponíveis."
            )
            return {"skipped": True, "reason": "nenhum encoder pôde ser carregado"}

        return {
            "n_tweets_codificados": tweets.height,
            "modelos": sorted(written),
            "dimensoes": dimensions,
            "written": written,
        }
