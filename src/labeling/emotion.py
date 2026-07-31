"""Classificação de emoções finas por encoder Transformer.

Complementa o sentimento (positivo/negativo/neutro) com as emoções que a
proposta lista como atributos: tristeza, raiva, medo, alegria, nojo e
surpresa. A distinção importa porque tristeza e raiva têm a mesma polaridade
negativa, mas significados clínicos bastante diferentes.

Emoções previstas pelo modelo que não estão em ``target_emotions`` são
descartadas — manter colunas que a análise não vai usar só aumenta a
dimensionalidade de um dataset já pequeno em número de usuários.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from config.environment import resolve_device
from config.logging import get_logger
from config.settings import EmotionSection
from constants.columns import EMOTION_PREFIX, TEXT_NORMALIZED
from exceptions.model import ModelError
from labeling.sentiment import _import_transformers
from utils.progress import build_progress
from utils.validation import require_columns

logger = get_logger(__name__)

#: Tradução dos rótulos em inglês dos modelos públicos para o vocabulário pt-BR.
EMOTION_TRANSLATIONS: dict[str, str] = {
    "sadness": "tristeza",
    "anger": "raiva",
    "fear": "medo",
    "joy": "alegria",
    "disgust": "nojo",
    "surprise": "surpresa",
    "others": "outros",
    "no emotion": "outros",
}


class EmotionLabeler:
    """Rotulador de emoções finas.

    Parameters
    ----------
    config : EmotionSection
        Seção ``emotion`` de ``configs/labeling.yaml``.

    Examples
    --------
    >>> labeler = EmotionLabeler(config.labeling.emotion)  # doctest: +SKIP
    >>> labeler.label_frame(tweets)  # doctest: +SKIP
    """

    def __init__(self, config: EmotionSection) -> None:
        self.config = config
        self.device = resolve_device("auto")
        self._pipeline: Any | None = None

    def _load(self) -> Any:
        """Carrega o pipeline multi-rótulo de emoções."""
        if self._pipeline is not None:
            return self._pipeline

        transformers = _import_transformers()
        try:
            self._pipeline = transformers.pipeline(
                task="text-classification",
                model=self.config.model_name,
                device=0 if self.device == "cuda" else -1,
                truncation=True,
                max_length=self.config.max_length,
                top_k=None,
            )
        except (OSError, ValueError) as error:
            raise ModelError(
                f"Não foi possível carregar o modelo de emoções '{self.config.model_name}': {error}"
            ) from error

        logger.info("Encoder de emoções carregado: %s.", self.config.model_name)
        return self._pipeline

    def _translate(self, label: str) -> str:
        """Traduz um rótulo do modelo para o vocabulário pt-BR do projeto."""
        return EMOTION_TRANSLATIONS.get(label.lower().strip(), label.lower().strip())

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        """Prevê a intensidade de cada emoção-alvo.

        Parameters
        ----------
        texts : list of str
            Textos normalizados.

        Returns
        -------
        list of dict
            Um dicionário ``emoção -> score`` por texto. Emoções que o modelo
            não previu recebem ``0.0``, para que todas as linhas tenham as
            mesmas colunas.

        Examples
        --------
        >>> labeler.predict(["que raiva disso"])  # doctest: +SKIP
        [{'tristeza': 0.05, 'raiva': 0.81, ...}]
        """
        if not texts:
            return []

        classifier = self._load()
        targets = list(self.config.target_emotions)
        results: list[dict[str, float]] = []

        with build_progress() as progress:
            task = progress.add_task("Classificando emoções", total=len(texts))
            for start in range(0, len(texts), self.config.batch_size):
                batch = texts[start : start + self.config.batch_size]
                outputs = classifier(batch, batch_size=self.config.batch_size)

                for output in outputs:
                    scores = output if isinstance(output, list) else [output]
                    predicted = {
                        self._translate(str(item["label"])): float(item["score"]) for item in scores
                    }
                    results.append({emotion: predicted.get(emotion, 0.0) for emotion in targets})

                progress.advance(task, advance=len(batch))

        return results

    def label_frame(
        self,
        frame: pl.DataFrame,
        text_column: str = TEXT_NORMALIZED,
    ) -> pl.DataFrame:
        """Adiciona uma coluna ``emotion_<nome>`` por emoção-alvo.

        Parameters
        ----------
        frame : pl.DataFrame
            Tweets limpos.
        text_column : str, optional
            Coluna de texto de entrada, by default ``text_normalized``.

        Returns
        -------
        pl.DataFrame
            Tweets com as colunas de emoção.

        Examples
        --------
        >>> labeler.label_frame(tweets).columns  # doctest: +SKIP
        [..., 'emotion_tristeza', 'emotion_raiva', ...]
        """
        require_columns(frame, [text_column], context="rotulação de emoções")

        predictions = self.predict(frame[text_column].to_list())
        columns = [
            pl.Series(
                f"{EMOTION_PREFIX}{emotion}",
                [prediction.get(emotion, 0.0) for prediction in predictions],
                dtype=pl.Float64,
            )
            for emotion in self.config.target_emotions
        ]

        logger.info("Adicionadas %d colunas de emoção.", len(columns))
        return frame.with_columns(columns)
