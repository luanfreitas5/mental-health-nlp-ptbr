"""Etapa 7 — particionamento treino/validação/teste e folds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from config.logging import get_logger
from data.reader import read_parquet
from data.splitter import build_split_table
from data.writer import write_parquet
from pipelines.base import PipelineStage, StageContext
from schemas.users import SplitSchema
from schemas.validation import validate_frame
from utils.validation import check_no_group_leakage

logger = get_logger(__name__)


class SplittingStage(PipelineStage):
    """Atribui cada usuário a uma partição e a um fold."""

    name = "split"
    description = "Particiona os usuários em treino/validação/teste e gera os folds"

    def required_inputs(self, context: StageContext) -> list[Path]:
        """Exige a matriz de atributos."""
        return [context.paths.data.user_features]

    def run(self, context: StageContext) -> dict[str, Any]:
        """Gera e persiste a tabela de partições.

        Parameters
        ----------
        context : StageContext
            Contexto compartilhado.

        Returns
        -------
        dict
            Tamanho de cada partição, distribuição por classe e caminho gravado.

        Examples
        --------
        >>> SplittingStage().run(contexto)  # doctest: +SKIP
        """
        config = context.config
        paths = context.paths

        features = read_parquet(paths.data.user_features, columns=["user_id", "user_label"])

        splits = build_split_table(
            features,
            config.general.split,
            config.general.cross_validation,
            config.random_seed,
        )
        validate_frame(splits, SplitSchema, context="saída do particionamento")

        # Verificação explícita de vazamento: a garantia mais importante do
        # projeto merece ser reafirmada aqui, e não só dentro do splitter.
        train_users = splits.filter(pl.col("split") == "train")["user_id"].to_list()
        test_users = splits.filter(pl.col("split") == "test")["user_id"].to_list()
        check_no_group_leakage(train_users, test_users)

        target = write_parquet(splits, paths.data.splits)

        distribution = {
            row["split"]: row["len"]
            for row in splits.group_by("split").len().sort("split").iter_rows(named=True)
        }
        by_class = (
            splits.group_by(["split", "user_label"]).len().sort(["split", "user_label"]).to_dicts()
        )

        logger.info("Partições: %s", distribution)
        return {
            "distribuicao": distribution,
            "distribuicao_por_classe": by_class,
            "n_folds": config.general.cross_validation.n_splits,
            "written": str(target),
        }
