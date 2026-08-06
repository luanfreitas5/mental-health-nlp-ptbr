"""Etapa 3 — rotulação de sentimento por tweet e de classe por usuário."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import polars as pl

from config.logging import get_logger
from constants.columns import USER_ID
from data.reader import list_collected_users, read_partitioned, select_pending_users
from data.writer import write_parquet, write_user_partition
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
from utils.progress import track
from utils.validation import check_class_balance

logger = get_logger(__name__)


class LabelingStage(PipelineStage):
    """Rotula os tweets por sentimento e os usuários por classe.

    A rotulação por tweet (sentimento e emoções) processa um usuário por vez
    e grava ``tweets_labeled/`` imediatamente após cada um, retomável entre
    execuções (ver :func:`data.reader.select_pending_users`) e limitável por
    ``--limit-users``. A rotulação por usuário (supervisão fraca) roda por
    último, sobre todo o acumulado disponível em ``tweets_labeled/`` — ela
    depende de estatísticas da população inteira (equilíbrio de classes,
    amostragem para revisão manual) e por isso não é decomponível por usuário.
    """

    name = "label"
    description = "Rotula sentimento por tweet e classe por usuário (supervisão fraca)"

    def required_inputs(self, context: StageContext) -> list[Path]:
        """Exige os tweets limpos da etapa anterior."""
        return [context.paths.data.tweets_clean]

    def _label_tweets(self, context: StageContext) -> None:
        """Rotula, usuário a usuário, o sentimento (e emoções) pendentes."""
        config = context.config
        paths = context.paths

        clean = read_partitioned(paths.data.tweets_clean, stage="preprocess")
        already_processed = list_collected_users(paths.data.tweets_labeled)
        pending = select_pending_users(
            set(clean[USER_ID].unique().to_list()),
            already_processed,
            context.option("limit_users"),
        )
        logger.info(
            "Rotulação por tweet: %d usuários já processados, %d pendentes nesta execução.",
            len(already_processed),
            len(pending),
        )
        if not pending:
            return

        sentiment_config = config.labeling.sentiment
        sentiment_labeler = SentimentLabeler(sentiment_config) if sentiment_config.enabled else None
        if sentiment_labeler is None:
            logger.warning("Rotulação de sentimento desativada: colunas neutras serão criadas.")
        emotion_labeler = (
            EmotionLabeler(config.labeling.emotion) if config.labeling.emotion.enabled else None
        )

        groups = clean.filter(pl.col(USER_ID).is_in(pending)).partition_by(
            USER_ID, as_dict=True, maintain_order=True
        )
        for user_id in track(pending, "Rotulando usuários (sentimento/emoção)"):
            user_frame = groups.get((user_id,))
            if user_frame is None or user_frame.is_empty():
                continue

            if sentiment_labeler is not None:
                user_frame = sentiment_labeler.label_frame(user_frame)
            else:
                user_frame = user_frame.with_columns(
                    pl.lit("neutro").alias("sentiment"),
                    pl.lit(0.0).alias("sentiment_score"),
                    pl.lit(0.0).alias("sentiment_polarity"),
                )

            validate_frame(
                user_frame, LabeledTweetSchema, context="saída da rotulação de sentimento"
            )

            if emotion_labeler is not None:
                try:
                    user_frame = emotion_labeler.label_frame(user_frame)
                except (OSError, ValueError) as error:
                    # As emoções finas são complementares: sua ausência degrada as
                    # features emocionais, mas não invalida a etapa.
                    logger.warning("Rotulação de emoções falhou e será omitida: %s", error)

            write_user_partition(user_frame, paths.data.tweets_labeled, user_id)

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

        self._label_tweets(context)

        tweets = read_partitioned(paths.data.tweets_labeled, stage="label")
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
