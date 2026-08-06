"""Etapa 2 — limpeza, normalização e filtragem dos tweets coletados."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from config.logging import get_logger
from constants.columns import USER_ID
from data.reader import (
    list_collected_users,
    read_partitioned,
    read_user_histories,
    select_pending_users,
)
from data.writer import write_user_partition
from pipelines.base import PipelineStage, StageContext
from preprocessing.pipeline import run_preprocessing
from utils.progress import track

logger = get_logger(__name__)


class PreprocessingStage(PipelineStage):
    """Consolida os históricos por usuário e produz os tweets limpos.

    Processa um usuário por vez e grava o resultado em ``tweets_clean/``
    imediatamente após cada um: se a execução for interrompida, os usuários
    já processados não são refeitos na próxima chamada (ver
    :func:`data.reader.select_pending_users`). ``--limit-users`` limita
    quantos usuários pendentes esta execução processa.
    """

    name = "preprocess"
    description = "Limpa, normaliza e filtra os tweets coletados"

    def required_inputs(self, context: StageContext) -> list[Path]:
        """Exige o diretório de históricos produzido pela coleta."""
        return [context.paths.data.user_histories]

    def run(self, context: StageContext) -> dict[str, Any]:
        """Executa o pré-processamento e grava ``tweets_clean/`` particionado por usuário.

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

        already_processed = list_collected_users(paths.data.tweets_clean)
        pending = select_pending_users(
            set(raw[USER_ID].unique().to_list()),
            already_processed,
            context.option("limit_users"),
        )
        logger.info(
            "Pré-processamento: %d usuários já processados, %d pendentes nesta execução.",
            len(already_processed),
            len(pending),
        )

        groups = raw.partition_by(USER_ID, as_dict=True, maintain_order=True)
        for user_id in track(pending, "Pré-processando usuários"):
            user_raw = groups[(user_id,)]
            user_clean = run_preprocessing(user_raw, context.config, allow_empty=True)
            write_user_partition(user_clean, paths.data.tweets_clean, user_id)

        processed_users = list_collected_users(paths.data.tweets_clean)
        clean = (
            read_partitioned(paths.data.tweets_clean, stage="preprocess")
            if processed_users
            else pl.DataFrame()
        )

        return {
            "tweets_entrada": n_raw_tweets,
            "tweets_saida": clean.height,
            "usuarios_entrada": n_raw_users,
            "usuarios_saida": clean[USER_ID].n_unique() if not clean.is_empty() else 0,
            "usuarios_processados_nesta_execucao": len(pending),
            "taxa_retencao": round(clean.height / max(n_raw_tweets, 1), 4),
            "n_arquivos": len(processed_users),
            "written": str(paths.data.tweets_clean),
        }
