"""Rotulação automática de sentimento por encoder Transformer.

O modelo usado aqui é um **encoder** fine-tuned para classificação de
sentimento em português, não um LLM generativo. A distinção é metodológica:
o encoder é determinístico, barato e avaliado em benchmark público, enquanto
o LLM (Seção 6 da proposta) fica reservado à extração de atributos
psicológicos, onde a geração livre é justamente o que se quer.

O sentimento é tratado como **constructo auxiliar**: entra como feature e
como triagem, e nunca como proxy do risco clínico — que é medido pelo rótulo
do usuário, um constructo distinto.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import polars as pl
from rich.progress import Progress

from config.environment import resolve_device
from config.logging import get_logger
from config.settings import SentimentSection
from constants.columns import SENTIMENT, SENTIMENT_POLARITY, SENTIMENT_SCORE, TEXT_NORMALIZED
from constants.labels import SENTIMENT_POLARITY as POLARITY_MAP
from constants.labels import Sentiment
from exceptions.model import MissingDependencyError, ModelError
from utils.progress import build_progress
from utils.validation import require_columns

logger = get_logger(__name__)


@dataclass(frozen=True)
class SentimentPrediction:
    """Predição de sentimento para um texto.

    Attributes
    ----------
    label : str
        Sentimento no vocabulário do projeto.
    score : float
        Confiança do modelo, em ``[0, 1]``.
    """

    label: str
    score: float


def _import_transformers() -> Any:
    """Importa ``transformers`` sob demanda, com erro explicativo se ausente."""
    try:
        import transformers  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise MissingDependencyError(
            "transformers não está instalado. Rode 'make install-llm' para instalar os "
            "extras de LLM (PyTorch + Transformers)."
        ) from error
    return transformers


def _import_torch() -> Any:
    """Importa ``torch`` sob demanda, com erro explicativo se ausente."""
    try:
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise MissingDependencyError(
            "PyTorch não está instalado. Rode 'make install-llm' para instalar os "
            "extras de LLM (PyTorch + Transformers)."
        ) from error
    return torch


def _resolve_fp16(requested: bool, device: str, model_name: str) -> bool:
    """Decide se a inferência deve usar precisão mista (fp16).

    Mesma lógica de :meth:`models.transformer.TransformerClassifier._resolve_fp16`:
    fp16 só acelera em Tensor Cores de GPU CUDA, então a flag é ignorada (com
    aviso) fora de ``cuda`` em vez de gerar um erro.
    """
    if requested and device != "cuda":
        logger.warning(
            "fp16 solicitado para '%s', mas o dispositivo é '%s': inferindo em precisão total.",
            model_name,
            device,
        )
        return False
    return requested


def _apply_dynamic_quantization(pipeline: Any, device: str, model_name: str) -> None:
    """Aplica quantização dinâmica int8 às camadas lineares do modelo.

    A quantização dinâmica do PyTorch (backend fbgemm/qnnpack) só é suportada
    em CPU; em GPU CUDA ela é ignorada (com aviso) porque fp16 já cobre o
    mesmo objetivo — reduzir o tempo de inferência — nesse dispositivo.
    """
    if device == "cuda":
        logger.warning(
            "Quantização dinâmica solicitada para '%s', mas o dispositivo é 'cuda': "
            "ignorada (use fp16 para acelerar em GPU).",
            model_name,
        )
        return

    torch = _import_torch()
    pipeline.model = torch.quantization.quantize_dynamic(
        pipeline.model, {torch.nn.Linear}, dtype=torch.qint8
    )
    logger.info("Quantização dinâmica int8 aplicada a '%s'.", model_name)


class SentimentLabeler:
    """Rotulador de sentimento baseado em encoder Transformer.

    Parameters
    ----------
    config : SentimentSection
        Seção ``sentiment`` de ``configs/labeling.yaml``.

    Examples
    --------
    >>> labeler = SentimentLabeler(config.labeling.sentiment)  # doctest: +SKIP
    >>> labeler.predict(["hoje foi um dia difícil"])  # doctest: +SKIP
    [SentimentPrediction(label='negativo', score=0.93)]
    """

    def __init__(self, config: SentimentSection) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self._pipeline: Any | None = None
        self._model_name: str = config.model_name

    def _load(self) -> Any:
        """Carrega o pipeline de classificação, com fallback para o modelo alternativo."""
        if self._pipeline is not None:
            return self._pipeline

        transformers = _import_transformers()
        candidates = [self.config.model_name]
        if self.config.fallback_model_name:
            candidates.append(self.config.fallback_model_name)

        use_fp16 = _resolve_fp16(self.config.fp16, self.device, self.config.model_name)
        pipeline_kwargs: dict[str, Any] = {}
        if use_fp16:
            torch = _import_torch()
            pipeline_kwargs["torch_dtype"] = torch.float16

        errors: list[str] = []
        for model_name in candidates:
            try:
                self._pipeline = transformers.pipeline(
                    task="text-classification",
                    model=model_name,
                    revision=self.config.revision if model_name == self.config.model_name else None,
                    device=0 if self.device == "cuda" else -1,
                    truncation=True,
                    max_length=self.config.max_length,
                    top_k=None,
                    **pipeline_kwargs,
                )
            except (OSError, ValueError) as error:
                errors.append(f"{model_name}: {error}")
                logger.warning("Falha ao carregar '%s'. Tentando alternativa.", model_name)
                continue

            self._model_name = model_name
            if self.config.quantize:
                _apply_dynamic_quantization(self._pipeline, self.device, model_name)
            logger.info(
                "Encoder de sentimento carregado: %s (device=%s, fp16=%s, quantize=%s).",
                model_name,
                self.device,
                use_fp16,
                self.config.quantize and self.device != "cuda",
            )
            return self._pipeline

        raise ModelError(
            "Nenhum modelo de sentimento pôde ser carregado. Tentativas:\n" + "\n".join(errors)
        )

    def _map_label(self, raw_label: str) -> str:
        """Traduz o rótulo do modelo para o vocabulário do projeto."""
        mapping = self.config.label_mapping
        if raw_label in mapping:
            return mapping[raw_label]
        lowered = raw_label.lower()
        if lowered in mapping:
            return mapping[lowered]

        logger.warning(
            "Rótulo '%s' não está em labeling.sentiment.label_mapping: tratado como indefinido.",
            raw_label,
        )
        return str(Sentiment.INDEFINIDO)

    def predict(
        self,
        texts: list[str],
        *,
        progress: Progress | None = None,
    ) -> list[SentimentPrediction]:
        """Classifica uma lista de textos.

        Predições com confiança abaixo de ``min_confidence`` viram
        ``indefinido``: um rótulo incerto propagado como certo contaminaria
        as features emocionais e, por consequência, a supervisão fraca.

        Parameters
        ----------
        texts : list of str
            Textos normalizados.
        progress : rich.progress.Progress, optional
            Barra de progresso já aberta onde registrar uma nova tarefa, by
            default ``None`` (cria e gerencia a própria barra). Usado para
            compartilhar uma única barra quando este rotulador roda em
            paralelo com o de emoções (ver
            :meth:`pipelines.labeling.LabelingStage._label_parallel`) — dois
            ``rich.progress.Progress`` independentes escrevendo no mesmo
            console ao mesmo tempo corrompem a renderização.

        Returns
        -------
        list of SentimentPrediction
            Predições na mesma ordem da entrada.

        Examples
        --------
        >>> labeler.predict(["estou bem"])  # doctest: +SKIP
        [SentimentPrediction(label='positivo', score=0.88)]
        """
        if not texts:
            return []

        classifier = self._load()
        predictions: list[SentimentPrediction] = []

        with nullcontext(progress) if progress is not None else build_progress() as active_progress:
            task = active_progress.add_task("Rotulando sentimento", total=len(texts))
            for start in range(0, len(texts), self.config.batch_size):
                batch = texts[start : start + self.config.batch_size]
                outputs = classifier(batch, batch_size=self.config.batch_size)

                for output in outputs:
                    scores = output if isinstance(output, list) else [output]
                    best = max(scores, key=lambda item: float(item["score"]))
                    label = self._map_label(str(best["label"]))
                    score = float(best["score"])

                    if score < self.config.min_confidence:
                        label = str(Sentiment.INDEFINIDO)

                    predictions.append(SentimentPrediction(label=label, score=score))

                active_progress.advance(task, advance=len(batch))

        return predictions

    def label_frame(
        self,
        frame: pl.DataFrame,
        text_column: str = TEXT_NORMALIZED,
        *,
        progress: Progress | None = None,
    ) -> pl.DataFrame:
        """Adiciona as colunas de sentimento a um DataFrame de tweets.

        Parameters
        ----------
        frame : pl.DataFrame
            Tweets limpos.
        text_column : str, optional
            Coluna de texto usada como entrada, by default ``text_normalized``.
        progress : rich.progress.Progress, optional
            Repassado a :meth:`predict`; ver o parâmetro homônimo lá.

        Returns
        -------
        pl.DataFrame
            Tweets com ``sentiment``, ``sentiment_score`` e ``sentiment_polarity``.

        Examples
        --------
        >>> labeler.label_frame(tweets)  # doctest: +SKIP
        """
        require_columns(frame, [text_column], context="rotulação de sentimento")

        predictions = self.predict(frame[text_column].to_list(), progress=progress)
        labels = [prediction.label for prediction in predictions]
        scores = [prediction.score for prediction in predictions]

        result = frame.with_columns(
            pl.Series(SENTIMENT, labels, dtype=pl.Utf8),
            pl.Series(SENTIMENT_SCORE, scores, dtype=pl.Float64),
        ).with_columns(
            pl.col(SENTIMENT)
            .replace_strict(POLARITY_MAP, default=0.0, return_dtype=pl.Float64)
            .alias(SENTIMENT_POLARITY)
        )

        distribution = result.group_by(SENTIMENT).len().sort("len", descending=True)
        logger.info("Distribuição de sentimento: %s", dict(distribution.iter_rows()))
        return result
