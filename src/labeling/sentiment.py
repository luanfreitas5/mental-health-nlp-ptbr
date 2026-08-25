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

from dataclasses import dataclass
from typing import Any

import polars as pl

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
                )
            except (OSError, ValueError) as error:
                errors.append(f"{model_name}: {error}")
                logger.warning("Falha ao carregar '%s'. Tentando alternativa.", model_name)
                continue

            self._model_name = model_name
            logger.info("Encoder de sentimento carregado: %s (device=%s).", model_name, self.device)
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

    def predict(self, texts: list[str]) -> list[SentimentPrediction]:
        """Classifica uma lista de textos.

        Predições com confiança abaixo de ``min_confidence`` viram
        ``indefinido``: um rótulo incerto propagado como certo contaminaria
        as features emocionais e, por consequência, a supervisão fraca.

        Parameters
        ----------
        texts : list of str
            Textos normalizados.

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

        with build_progress() as progress:
            task = progress.add_task("Rotulando sentimento", total=len(texts))
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

                progress.advance(task, advance=len(batch))

        return predictions

    def label_frame(
        self,
        frame: pl.DataFrame,
        text_column: str = TEXT_NORMALIZED,
    ) -> pl.DataFrame:
        """Adiciona as colunas de sentimento a um DataFrame de tweets.

        Parameters
        ----------
        frame : pl.DataFrame
            Tweets limpos.
        text_column : str, optional
            Coluna de texto usada como entrada, by default ``text_normalized``.

        Returns
        -------
        pl.DataFrame
            Tweets com ``sentiment``, ``sentiment_score`` e ``sentiment_polarity``.

        Examples
        --------
        >>> labeler.label_frame(tweets)  # doctest: +SKIP
        """
        require_columns(frame, [text_column], context="rotulação de sentimento")

        predictions = self.predict(frame[text_column].to_list())
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
