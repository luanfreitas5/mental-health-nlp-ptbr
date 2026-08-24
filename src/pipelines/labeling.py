"""Etapa 3 — rotulação de sentimento por tweet e de classe por usuário."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import polars as pl

from config.logging import get_logger
from data.reader import (
    list_collected_users,
    read_partitioned,
    read_user_partition,
    select_pending_users,
)
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
from utils.progress import build_progress
from utils.validation import check_class_balance

logger = get_logger(__name__)


class LabelingStage(PipelineStage):
    """Rotula os tweets por sentimento e os usuários por classe.

    A rotulação por tweet (sentimento e emoções) acumula os tweets de vários
    usuários até atingir o ``batch_size`` dos encoders Transformer antes de
    cada chamada de inferência — rotular usuário a usuário sub-utilizaria a
    GPU sempre que um usuário tivesse menos tweets que o lote configurado
    (ver :func:`_iter_user_batches`). A escrita em ``tweets_labeled/``
    continua por usuário, imediatamente após cada lote ser rotulado,
    retomável entre execuções (ver :func:`data.reader.select_pending_users`)
    e limitável por ``--limit-users``. A rotulação por usuário (supervisão
    fraca) roda por último, sobre todo o acumulado disponível em
    ``tweets_labeled/`` — ela depende de estatísticas da população inteira
    (equilíbrio de classes, amostragem para revisão manual) e por isso não é
    decomponível por usuário ou por lote.
    """

    name = "label"
    description = "Rotula sentimento por tweet e classe por usuário (supervisão fraca)"

    def required_inputs(self, context: StageContext) -> list[Path]:
        """Exige os tweets limpos da etapa anterior."""
        return [context.paths.data.tweets_clean]

    def _label_tweets(self, context: StageContext) -> None:
        """Rotula, em lotes que cobrem vários usuários, o sentimento (e emoções) pendentes."""
        config = context.config
        paths = context.paths

        available = list_collected_users(paths.data.tweets_clean)
        already_processed = list_collected_users(paths.data.tweets_labeled)
        pending = select_pending_users(available, already_processed, context.option("limit_users"))
        logger.info(
            "Rotulação por tweet: %d usuários já processados, %d pendentes nesta execução.",
            len(already_processed),
            len(pending),
        )
        if not pending:
            return

        sentiment_labeler, emotion_labeler = self._build_labelers(config)
        batch_size = self._resolve_inference_batch_size(config)

        with build_progress() as progress:
            task = progress.add_task("Rotulando usuários (sentimento/emoção)", total=len(pending))
            for batch in self._iter_user_batches(pending, paths.data.tweets_clean, batch_size):
                self._label_and_write_batch(
                    batch,
                    sentiment_labeler,
                    emotion_labeler,
                    paths.data.tweets_labeled,
                )
                progress.advance(task, advance=len(batch))

    def _resolve_inference_batch_size(self, config: Any) -> int:
        """Tamanho mínimo de acúmulo de tweets entre usuários antes de cada inferência.

        Usa o maior ``batch_size`` entre sentimento e emoção habilitados, para
        que acumular até esse limite garanta lotes cheios para os dois
        encoders. Sem nenhum dos dois habilitado, o agrupamento não afeta
        nenhuma chamada de inferência, então ``1`` preserva o particionamento
        usuário a usuário.

        Parameters
        ----------
        config : Any
            Configuração completa do projeto.

        Returns
        -------
        int
            Quantidade mínima de tweets a acumular por lote de inferência.
        """
        sizes = [
            section.batch_size
            for section in (config.labeling.sentiment, config.labeling.emotion)
            if section.enabled
        ]
        return max(sizes) if sizes else 1

    def _iter_user_batches(
        self,
        pending: list[str],
        clean_dir: Path,
        batch_size: int,
    ) -> Iterator[list[tuple[str, pl.DataFrame]]]:
        """Agrupa usuários pendentes até acumular ao menos ``batch_size`` tweets.

        Cada usuário ainda é lido individualmente (preserva a leitura
        proporcional a um usuário usada pela retomada), mas o resultado é
        acumulado em memória até o lote atingir o tamanho de inferência —
        é esse acúmulo entre usuários que permite ao ``pipeline()`` de
        sentimento/emoção receber lotes do tamanho configurado em vez de, no
        limite, um usuário pequeno por chamada.

        Parameters
        ----------
        pending : list of str
            Usuários pendentes, na ordem de processamento.
        clean_dir : Path
            Diretório particionado por usuário dos tweets limpos.
        batch_size : int
            Quantidade mínima de tweets para fechar um lote.

        Yields
        ------
        list of tuple
            Pares ``(user_id, DataFrame)`` do lote, na ordem de leitura.

        Examples
        --------
        >>> list(stage._iter_user_batches(["u_a"], clean_dir, 32))  # doctest: +SKIP
        """
        batch: list[tuple[str, pl.DataFrame]] = []
        accumulated = 0
        for user_id in pending:
            user_frame = read_user_partition(clean_dir, user_id)
            batch.append((user_id, user_frame))
            accumulated += user_frame.height
            if accumulated >= batch_size:
                yield batch
                batch = []
                accumulated = 0
        if batch:
            yield batch

    def _build_labelers(self, config: Any) -> tuple[SentimentLabeler | None, EmotionLabeler | None]:
        """Instancia os rotuladores de sentimento e emoção conforme habilitados na configuração."""
        sentiment_config = config.labeling.sentiment
        sentiment_labeler = SentimentLabeler(sentiment_config) if sentiment_config.enabled else None
        if sentiment_labeler is None:
            logger.warning("Rotulação de sentimento desativada: colunas neutras serão criadas.")
        emotion_labeler = (
            EmotionLabeler(config.labeling.emotion) if config.labeling.emotion.enabled else None
        )
        return sentiment_labeler, emotion_labeler

    def _label_and_write_batch(
        self,
        batch: list[tuple[str, pl.DataFrame]],
        sentiment_labeler: SentimentLabeler | None,
        emotion_labeler: EmotionLabeler | None,
        labeled_dir: Path,
    ) -> None:
        """Rotula um lote de usuários numa única passada pelos encoders e grava cada partição.

        Os tweets do lote são concatenados antes da inferência — o que
        permite ao ``pipeline()`` de sentimento/emoção processar o lote
        inteiro de uma vez, em vez de um usuário por chamada — e depois
        divididos de volta por usuário para a escrita, preservando a
        retomada fina existente. Usuários sem tweets pendentes ainda são
        gravados (vazios): é o que os marca como "já processados" (ver
        :func:`data.writer.write_user_partition`).

        Parameters
        ----------
        batch : list of tuple
            Pares ``(user_id, DataFrame)`` do lote, produzidos por
            :func:`_iter_user_batches`.
        sentiment_labeler : SentimentLabeler, optional
            Rotulador de sentimento, ou ``None`` se desativado.
        emotion_labeler : EmotionLabeler, optional
            Rotulador de emoções, ou ``None`` se desativado.
        labeled_dir : Path
            Diretório particionado por usuário onde gravar o resultado.
        """
        non_empty = [(user_id, frame) for user_id, frame in batch if not frame.is_empty()]
        if not non_empty:
            for user_id, frame in batch:
                write_user_partition(frame, labeled_dir, user_id)
            return

        combined = pl.concat([frame for _, frame in non_empty], how="vertical_relaxed")
        labeled = self._label_user_frame(combined, sentiment_labeler, emotion_labeler)

        labeled_by_user: dict[str, pl.DataFrame] = {}
        offset = 0
        for user_id, frame in non_empty:
            labeled_by_user[user_id] = labeled.slice(offset, frame.height)
            offset += frame.height

        for user_id, frame in batch:
            write_user_partition(labeled_by_user.get(user_id, frame), labeled_dir, user_id)

    def _label_user_frame(
        self,
        user_frame: pl.DataFrame,
        sentiment_labeler: SentimentLabeler | None,
        emotion_labeler: EmotionLabeler | None,
    ) -> pl.DataFrame:
        """Aplica sentimento (ou neutro padrão) e emoções a um usuário, validando o esquema."""
        if sentiment_labeler is not None:
            user_frame = sentiment_labeler.label_frame(user_frame)
        else:
            user_frame = user_frame.with_columns(
                pl.lit("neutro").alias("sentiment"),
                pl.lit(0.0).alias("sentiment_score"),
                pl.lit(0.0).alias("sentiment_polarity"),
            )

        validate_frame(user_frame, LabeledTweetSchema, context="saída da rotulação de sentimento")

        if emotion_labeler is not None:
            try:
                user_frame = emotion_labeler.label_frame(user_frame)
            except (OSError, ValueError) as error:
                # As emoções finas são complementares: sua ausência degrada as
                # features emocionais, mas não invalida a etapa.
                logger.warning("Rotulação de emoções falhou e será omitida: %s", error)

        return user_frame

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
