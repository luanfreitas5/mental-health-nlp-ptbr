"""Etapa 1 — coleta longitudinal de tweets.

Protegida por uma **barreira ética**: a coleta só executa quando o ``.env``
declara o número CAAE da aprovação CEP/CONEP. Não é burocracia defensiva —
coletar publicações de pessoas em sofrimento psíquico sem aprovação prévia é
uma violação de protocolo de pesquisa, e o código não deve tornar isso fácil.

A barreira pode ser contornada com ``--allow-collection-without-ethics``
apenas para testes técnicos com contas próprias, e o uso fica registrado no
log com aviso explícito.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from config.logging import get_logger
from data.collector import TweetCollector
from data.queries import build_queries, summarize_queries
from data.writer import write_parquet
from exceptions.pipeline import EthicalGateError
from pipelines.base import PipelineStage, StageContext
from schemas.users import UserMetadataSchema
from schemas.validation import validate_frame

logger = get_logger(__name__)


class CollectionStage(PipelineStage):
    """Executa a busca semente e a coleta retrospectiva dos históricos."""

    name = "collect"
    description = "Coleta tweets do X/Twitter por palavras-chave e histórico de usuários"

    def check_ethical_gate(self, context: StageContext) -> None:
        """Verifica a aprovação ética antes de qualquer requisição.

        Parameters
        ----------
        context : StageContext
            Contexto compartilhado.

        Raises
        ------
        EthicalGateError
            Se não houver aprovação registrada e a barreira não tiver sido
            explicitamente dispensada.

        Examples
        --------
        >>> etapa.check_ethical_gate(contexto)  # doctest: +SKIP
        """
        approval = context.config.secrets.ethics_approval_id.strip()

        if approval:
            logger.info("Coleta autorizada sob o protocolo CEP/CONEP: %s.", approval)
            return

        if context.option("allow_collection_without_ethics", False):
            logger.warning(
                "BARREIRA ÉTICA DISPENSADA. Esta execução é válida apenas para teste "
                "técnico com contas próprias. Dados coletados assim NÃO podem compor a "
                "base da dissertação."
            )
            return

        raise EthicalGateError(
            "Coleta bloqueada: nenhuma aprovação CEP/CONEP registrada. Defina "
            "ETHICS_APPROVAL_ID no .env com o número do CAAE. Para um teste técnico "
            "sem dados de pesquisa, use --allow-collection-without-ethics. "
            "Ver docs/guides/ethics.md."
        )

    def run(self, context: StageContext) -> dict[str, Any]:
        """Executa a coleta completa.

        Parameters
        ----------
        context : StageContext
            Contexto compartilhado.

        Returns
        -------
        dict
            Número de consultas, usuários coletados e caminhos gravados.

        Examples
        --------
        >>> CollectionStage().run(contexto)  # doctest: +SKIP
        """
        self.check_ethical_gate(context)

        config = context.config
        paths = context.paths

        queries = build_queries(config.collection.seed_search)
        summary = summarize_queries(queries)
        for group, counts in summary.items():
            logger.info(
                "Grupo '%s': %d consultas (%d palavras-chave, %d hashtags).",
                group,
                counts["total"],
                counts.get("keyword", 0),
                counts.get("hashtag", 0),
            )

        if context.option("dry_run", False):
            logger.info("Modo dry-run: as consultas foram construídas, mas nada foi coletado.")
            return {"n_queries": len(queries), "groups": summary, "dry_run": True}

        metadata = TweetCollector(
            config=config.collection,
            salt=config.secrets.pseudonymization_salt,
            output_dir=paths.data.user_histories,
            seed_dir=paths.data.seed_tweets,
        ).run()

        written: dict[str, str] = {}
        if isinstance(metadata, pl.DataFrame) and not metadata.is_empty():
            validated = validate_frame(
                metadata.unique(subset=["user_id"]),
                UserMetadataSchema,
                context="saída da coleta (metadados)",
            )
            written["user_metadata"] = str(write_parquet(validated, paths.data.user_metadata))

        from data.reader import count_users

        n_users = count_users(paths.data.user_histories)
        logger.info("Coleta concluída: %d históricos de usuário em disco.", n_users)

        return {
            "n_queries": len(queries),
            "groups": summary,
            "n_users_collected": n_users,
            "written": written,
        }
