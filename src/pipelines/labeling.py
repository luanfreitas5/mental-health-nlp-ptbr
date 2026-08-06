"""Etapa 3 — rotulação de sentimento por tweet e de classe por usuário."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import polars as pl

from config.logging import get_logger
from data.reader import read_partitioned
from data.writer import write_parquet, write_partitioned
from labeling.emotion import EmotionLabeler
from labeling.sentiment import SentimentLabeler
from labeling.validation import (
    apply_manual_labels,
    compute_agreement,
    drop_undecided,
    load_manual_labels,
    sample_for_manual_review,
)
from labeling.weak_supervision import assign_user_labels
from pipelines.base import PipelineStage, StageContext
from schemas.tweets import LabeledTweetSchema
from schemas.users import UserLabelSchema
from schemas.validation import validate_frame
from utils.files import write_json
from utils.validation import check_class_balance

logger = get_logger(__name__)


class LabelingStage(PipelineStage):
    """Rotula os tweets por sentimento e os usuários por classe."""

    name = "label"
    description = "Rotula sentimento por tweet e classe por usuário (supervisão fraca)"

    def required_inputs(self, context: StageContext) -> list[Path]:
        """Exige os tweets limpos da etapa anterior."""
        return [context.paths.data.tweets_clean]

    def run(self, context: StageContext) -> dict[str, Any]:
        """Executa a rotulação em dois níveis.

        Parameters
        ----------
        context : StageContext
            Contexto compartilhado.

        Returns
        -------
        dict
            Distribuições de sentimento e de classe, concordância com a
            revisão manual e caminhos gravados.

        Examples
        --------
        >>> LabelingStage().run(contexto)  # doctest: +SKIP
        """
        config = context.config
        paths = context.paths

        tweets = read_partitioned(paths.data.tweets_clean, stage="preprocess")

        # --- Nível tweet: sentimento (e emoções finas, se ativas) ----------
        if config.labeling.sentiment.enabled:
            tweets = SentimentLabeler(config.labeling.sentiment).label_frame(tweets)
        else:
            logger.warning("Rotulação de sentimento desativada: colunas neutras serão criadas.")
            tweets = tweets.with_columns(
                pl.lit("neutro").alias("sentiment"),
                pl.lit(0.0).alias("sentiment_score"),
                pl.lit(0.0).alias("sentiment_polarity"),
            )

        validate_frame(tweets, LabeledTweetSchema, context="saída da rotulação de sentimento")

        if config.labeling.emotion.enabled:
            try:
                tweets = EmotionLabeler(config.labeling.emotion).label_frame(tweets)
            except (OSError, ValueError) as error:
                # As emoções finas são complementares: sua ausência degrada as
                # features emocionais, mas não invalida a etapa.
                logger.warning("Rotulação de emoções falhou e será omitida: %s", error)

        write_partitioned(tweets, paths.data.tweets_labeled, "user_id", clear=True)
        labeled_path = paths.data.tweets_labeled

        # --- Nível usuário: supervisão fraca -------------------------------
        labels = assign_user_labels(tweets, config.labeling.user_labeling)

        manual = load_manual_labels(
            Path(config.labeling.user_labeling.consensus.manual_labels_file)
        )
        labels = apply_manual_labels(labels, manual)
        agreement = compute_agreement(labels)

        review_sample = sample_for_manual_review(
            labels,
            config.labeling.user_labeling.consensus.manual_review_sample_size,
            config.random_seed,
        )
        if not review_sample.is_empty():
            write_parquet(
                review_sample,
                Path(config.labeling.user_labeling.consensus.manual_review_file),
                log_hash=False,
            )

        labels = drop_undecided(labels, config.labeling.user_labeling)
        validate_frame(labels, UserLabelSchema, context="saída da rotulação de usuários")

        distribution = check_class_balance(
            labels["user_label"].to_list(),
            max_ratio=config.collection.sampling.max_class_imbalance_ratio,
        )
        labels_path = write_parquet(labels, paths.data.user_labels)

        # `Series.mean()` devolve `None` se `labels` ficar vazio (todos
        # "indefinido" descartados); `or 0.0` esconderia esse caso do
        # refurb sem deixar de ser necessário, por isso o `if` explícito.
        raw_agreement_mean = labels["label_agreement"].mean()
        agreement_mean = cast(float, raw_agreement_mean) if raw_agreement_mean is not None else 0.0

        write_json(
            paths.reports.metrics / "labeling_quality.json",
            {
                "distribuicao_classes": distribution,
                "concordancia_revisao_manual": agreement,
                "concordancia_media_fontes": agreement_mean,
                "n_usuarios_rotulados": labels.height,
            },
        )

        return {
            "n_tweets_rotulados": tweets.height,
            "n_usuarios_rotulados": labels.height,
            "distribuicao_classes": distribution,
            "concordancia": agreement,
            "written": {"tweets": str(labeled_path), "labels": str(labels_path)},
        }
