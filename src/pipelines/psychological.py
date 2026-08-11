"""Etapa 4 — extração do vetor psicológico por LLM local (Ollama).

É a etapa mais cara do pipeline em tempo de inferência e a única que depende
de um serviço externo ao processo. Por isso é opcional: se o Ollama não
estiver disponível, o pipeline segue sem o grupo ``psychological``, com aviso
explícito — e o Ablation Study passa a medir, na prática, o custo dessa
ausência.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.logging import get_logger
from config.settings import Config
from data.reader import (
    count_partitioned_rows,
    list_collected_users,
    read_user_partition,
    select_pending_users,
)
from data.writer import write_user_partition
from exceptions.model import LLMUnavailableError, MissingDependencyError
from labeling.llm import PsychologicalExtractor
from pipelines.base import PipelineStage, StageContext
from schemas.tweets import PsychologicalScoreSchema
from schemas.validation import validate_frame
from utils.files import list_files
from utils.progress import track

logger = get_logger(__name__)


class PsychologicalStage(PipelineStage):
    """Extrai o vetor psicológico de cada usuário via LLM.

    É a etapa mais lenta do pipeline (uma chamada ao LLM por lote de tweets),
    então cada usuário é gravado em ``psychological_scores/`` imediatamente
    após sua extração: uma interrupção não custa horas de inferência já
    feita, e ``--limit-users`` permite processar em lotes controlados.
    """

    name = "psych"
    description = "Extrai atributos psicológicos com LLM local (Ollama)"

    def required_inputs(self, context: StageContext) -> list[Path]:
        """Exige os tweets limpos da etapa de pré-processamento."""
        return [context.paths.data.tweets_clean]

    def _resolve_source_dir(self, context: StageContext) -> Path:
        """Aponta para os tweets rotulados, se existirem, ou os tweets limpos."""
        paths = context.paths
        labeled_available = bool(list_files(paths.data.tweets_labeled, "*.parquet"))
        return paths.data.tweets_labeled if labeled_available else paths.data.tweets_clean

    def _create_extractor(self, config: Config) -> PsychologicalExtractor:
        """Instancia o extrator e garante que o modelo Ollama está disponível."""
        extractor = PsychologicalExtractor(config.llm)
        extractor.client.ensure_model(config.llm.psychological_features.model)
        return extractor

    def _extract_pending_users(
        self,
        context: StageContext,
        extractor: PsychologicalExtractor,
        source_dir: Path,
        pending: list[str],
    ) -> None:
        """Extrai e grava o vetor psicológico de cada usuário pendente."""
        paths = context.paths
        for user_id in track(pending, "Extraindo vetor psicológico"):
            user_frame = read_user_partition(source_dir, user_id)
            if user_frame.is_empty():
                continue

            scores = extractor.extract_frame(user_frame)
            if scores.is_empty():
                logger.warning("Nenhum vetor psicológico válido para o usuário.")
                continue

            validate_frame(
                scores, PsychologicalScoreSchema, context="saída da extração psicológica"
            )
            write_user_partition(scores, paths.data.psychological_scores, user_id)

    def run(self, context: StageContext) -> dict[str, Any]:
        """Executa a extração psicológica.

        Parameters
        ----------
        context : StageContext
            Contexto compartilhado.

        Returns
        -------
        dict
            Número de lotes processados e caminho gravado, ou o motivo da
            etapa ter sido pulada.

        Examples
        --------
        >>> PsychologicalStage().run(contexto)  # doctest: +SKIP
        """
        config = context.config
        paths = context.paths

        if not config.llm.psychological_features.enabled:
            logger.info("Extração psicológica desativada em configs/llm.yaml.")
            return {"skipped": True, "reason": "desativada na configuração"}

        # Prefere os tweets já rotulados (têm sentimento), mas funciona sem eles.
        source_dir = self._resolve_source_dir(context)

        available = list_collected_users(source_dir)
        already_processed = list_collected_users(paths.data.psychological_scores)
        pending = select_pending_users(available, already_processed, context.option("limit_users"))
        logger.info(
            "Extração psicológica: %d usuários já processados, %d pendentes nesta execução.",
            len(already_processed),
            len(pending),
        )

        if pending:
            try:
                extractor = self._create_extractor(config)
            except (LLMUnavailableError, MissingDependencyError) as error:
                logger.warning(
                    "Extração psicológica pulada: %s. O grupo 'psychological' ficará ausente "
                    "da matriz de atributos.",
                    error,
                )
                return {"skipped": True, "reason": str(error)}

            self._extract_pending_users(context, extractor, source_dir, pending)

        processed_users = list_collected_users(paths.data.psychological_scores)
        if not processed_users:
            logger.warning("Nenhum vetor psicológico válido foi produzido.")
            return {"skipped": True, "reason": "nenhuma resposta válida do LLM"}

        return {
            "n_lotes": count_partitioned_rows(paths.data.psychological_scores),
            "n_usuarios": len(processed_users),
            "usuarios_processados_nesta_execucao": len(pending),
            "modelo": config.llm.psychological_features.model,
            "versao_prompt": config.llm.prompts.version,
            "written": str(paths.data.psychological_scores),
        }
