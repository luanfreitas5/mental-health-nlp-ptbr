"""Etapa 2 — limpeza, normalização e filtragem dos tweets coletados."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.logging import get_logger
from data.reader import (
    count_partitioned_rows,
    list_collected_users,
    read_user_history,
    select_pending_users,
)
from data.writer import write_user_partition
from pipelines.base import PipelineStage, StageContext
from preprocessing.pipeline import run_preprocessing
from utils.progress import track

logger = get_logger(__name__)


class PreprocessingStage(PipelineStage):
    """Consolida os históricos por usuário e produz os tweets limpos.

    Processa um usuário por vez, lendo e gravando um arquivo por vez: nunca
    materializa o histórico de mais de um usuário simultaneamente. O
    resultado vai para ``tweets_clean/`` imediatamente após cada usuário —
    se a execução for interrompida, os já processados não são refeitos na
    próxima chamada (ver :func:`data.reader.select_pending_users`).
    ``--limit-users`` limita quantos usuários pendentes esta execução
    processa.
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

        available = list_collected_users(paths.data.user_histories)
        n_raw_tweets = count_partitioned_rows(paths.data.user_histories)

        already_processed = list_collected_users(paths.data.tweets_clean)
        pending = select_pending_users(available, already_processed, context.option("limit_users"))
        logger.info(
            "Pré-processamento: %d usuários já processados, %d pendentes nesta execução.",
            len(already_processed),
            len(pending),
        )

        for user_id in track(pending, "Pré-processando usuários"):
            user_raw = read_user_history(paths.data.user_histories, user_id)
            if user_raw.is_empty():
                continue
            user_clean = run_preprocessing(user_raw, context.config, allow_empty=True)
            write_user_partition(user_clean, paths.data.tweets_clean, user_id)

        processed_users = list_collected_users(paths.data.tweets_clean)
        n_clean_tweets = count_partitioned_rows(paths.data.tweets_clean) if processed_users else 0

        return {
            "tweets_entrada": n_raw_tweets,
            "tweets_saida": n_clean_tweets,
            "usuarios_entrada": len(available),
            "usuarios_saida": len(processed_users),
            "usuarios_processados_nesta_execucao": len(pending),
            "taxa_retencao": round(n_clean_tweets / max(n_raw_tweets, 1), 4),
            "n_arquivos": len(processed_users),
            "written": str(paths.data.tweets_clean),
        }
