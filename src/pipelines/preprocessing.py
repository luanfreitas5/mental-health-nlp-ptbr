"""Etapa 2 — limpeza, normalização e filtragem dos tweets coletados."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.logging import get_logger
from data.reader import read_user_histories
from data.writer import write_parquet
from pipelines.base import PipelineStage, StageContext
from preprocessing.pipeline import run_preprocessing

logger = get_logger(__name__)


class PreprocessingStage(PipelineStage):
    """Consolida os históricos por usuário e produz os tweets limpos."""

    name = "preprocess"
    description = "Limpa, normaliza e filtra os tweets coletados"

    def required_inputs(self, context: StageContext) -> list[Path]:
        """Exige o diretório de históricos produzido pela coleta."""
        return [context.paths.data.user_histories]

    def run(self, context: StageContext) -> dict[str, Any]:
        """Executa o pré-processamento e grava ``tweets_clean.parquet``.

        Parameters
        ----------
        context : StageContext
            Contexto compartilhado.

        Returns
        -------
        dict
            Contagens antes e depois, e caminho gravado.

        Examples
        --------
        >>> PreprocessingStage().run(contexto)  # doctest: +SKIP
        """
        paths = context.paths

        raw = read_user_histories(paths.data.user_histories)
        n_raw_tweets, n_raw_users = raw.height, raw["user_id"].n_unique()

        clean = run_preprocessing(raw, context.config)
        target = write_parquet(clean, paths.data.tweets_clean)

        return {
            "tweets_entrada": n_raw_tweets,
            "tweets_saida": clean.height,
            "usuarios_entrada": n_raw_users,
            "usuarios_saida": clean["user_id"].n_unique(),
            "taxa_retencao": round(clean.height / max(n_raw_tweets, 1), 4),
            "written": str(target),
        }
