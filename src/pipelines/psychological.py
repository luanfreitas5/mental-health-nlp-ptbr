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
from data.reader import read_parquet
from data.writer import write_parquet
from exceptions.model import LLMUnavailableError, MissingDependencyError
from labeling.llm import PsychologicalExtractor
from pipelines.base import PipelineStage, StageContext
from schemas.tweets import PsychologicalScoreSchema
from schemas.validation import validate_frame

logger = get_logger(__name__)


class PsychologicalStage(PipelineStage):
    """Extrai o vetor psicológico de cada usuário via LLM."""

    name = "psych"
    description = "Extrai atributos psicológicos com LLM local (Ollama)"

    def required_inputs(self, context: StageContext) -> list[Path]:
        """Exige os tweets limpos da etapa de pré-processamento."""
        return [context.paths.data.tweets_clean]

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
        source = (
            paths.data.tweets_labeled
            if paths.data.tweets_labeled.is_file()
            else paths.data.tweets_clean
        )
        tweets = read_parquet(source)

        limit = context.option("limit_users")
        if limit:
            selected = tweets["user_id"].unique().sort().head(int(limit))
            tweets = tweets.filter(tweets["user_id"].is_in(selected))
            logger.info("Extração limitada a %d usuários (--limit-users).", int(limit))

        try:
            extractor = PsychologicalExtractor(config.llm)
            scores = extractor.extract_frame(tweets)
        except (LLMUnavailableError, MissingDependencyError) as error:
            logger.warning(
                "Extração psicológica pulada: %s. O grupo 'psychological' ficará ausente "
                "da matriz de atributos.",
                error,
            )
            return {"skipped": True, "reason": str(error)}

        if scores.is_empty():
            logger.warning("Nenhum vetor psicológico válido foi produzido.")
            return {"skipped": True, "reason": "nenhuma resposta válida do LLM"}

        validate_frame(scores, PsychologicalScoreSchema, context="saída da extração psicológica")
        target = write_parquet(scores, paths.data.psychological_scores)

        return {
            "n_lotes": scores.height,
            "n_usuarios": scores["user_id"].n_unique(),
            "modelo": config.llm.psychological_features.model,
            "versao_prompt": config.llm.prompts.version,
            "written": str(target),
        }
