"""O LLM open-source como classificador da comparação principal.

Diferente dos demais modelos, este não é treinado: o ``fit`` apenas seleciona
os exemplos few-shot. Isso muda o significado da comparação, e a diferença
precisa estar explícita na dissertação — o LLM parte de conhecimento
pré-treinado, sem ver os dados de treino, enquanto XGBoost, BiLSTM e
BERTimbau os utilizam integralmente.

Os exemplos few-shot vêm **sempre** do split de treino. Usar exemplos do
teste seria vazamento direto, e um dos mais fáceis de cometer sem perceber ao
trabalhar com prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import ValidationError

from config.logging import get_logger
from config.settings import LLMConfig
from constants.labels import UserLabel
from exceptions.model import LLMResponseError, LLMUnavailableError
from labeling.llm import OllamaClient, UserClassification
from labeling.prompt import build_classifier_prompt
from models.base import BaseUserClassifier, UserDataset
from utils.progress import build_progress

logger = get_logger(__name__)


@dataclass
class LLMClassifier(BaseUserClassifier):
    """Classificador baseado em LLM local (Ollama), zero-shot ou few-shot.

    Parameters
    ----------
    name : str
        Nome do modelo.
    params : dict
        Hiperparâmetros (``model``, ``mode``, ``temperature``, ...).
    llm_config : LLMConfig
        Configuração de ``configs/llm.yaml``.

    Examples
    --------
    >>> modelo = LLMClassifier(name="ollama_primary", llm_config=config.llm)  # doctest: +SKIP
    >>> modelo.fit(treino).predict(teste)  # doctest: +SKIP
    """

    llm_config: LLMConfig | None = None
    client_: OllamaClient | None = field(default=None, init=False, repr=False)
    examples_: list[tuple[list[str], str]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Marca a dependência dos textos dos usuários."""
        self.requires_text = True

    def _require_config(self) -> LLMConfig:
        """Garante que a configuração de LLM foi injetada."""
        if self.llm_config is None:
            raise ValueError(
                f"O modelo '{self.name}' precisa de llm_config (configs/llm.yaml) para operar."
            )
        return self.llm_config

    def fit(self, dataset: UserDataset) -> LLMClassifier:
        """Seleciona os exemplos few-shot a partir do conjunto de treino.

        Parameters
        ----------
        dataset : UserDataset
            Conjunto de treino, com rótulos e textos.

        Returns
        -------
        LLMClassifier
            O próprio modelo, pronto para inferência.

        Examples
        --------
        >>> modelo.fit(treino)  # doctest: +SKIP
        """
        self.validate_dataset(dataset)
        config = self._require_config()

        self.client_ = OllamaClient(config)
        self.client_.ensure_model(str(self.params.get("model", config.classifier.model)))

        if config.classifier.mode == "few_shot":
            texts = dataset.require_texts()
            assert dataset.labels is not None

            rng = np.random.default_rng(config.classifier.few_shot.random_state)
            per_class = config.classifier.few_shot.n_examples_per_class

            for class_index, class_name in enumerate(self.classes):
                candidates = [
                    user_id
                    for position, user_id in enumerate(dataset.user_ids)
                    if int(dataset.labels[position]) == class_index and texts.get(user_id)
                ]
                if not candidates:
                    logger.warning(
                        "Sem exemplos de '%s' no treino: o few-shot ficará incompleto.",
                        class_name,
                    )
                    continue

                chosen = rng.choice(candidates, size=min(per_class, len(candidates)), replace=False)
                self.examples_.extend((texts[user_id][:5], class_name) for user_id in chosen)

            logger.info("Few-shot preparado com %d exemplos.", len(self.examples_))

        self.is_fitted = True
        return self

    def _classify_user(self, tweets: list[str]) -> np.ndarray:
        """Classifica um usuário e devolve a distribuição de probabilidade."""
        config = self._require_config()
        assert self.client_ is not None

        uniform = np.full(len(self.classes), 1.0 / len(self.classes))
        prompt = build_classifier_prompt(tweets, config, self.examples_ or None)

        try:
            payload = self.client_.generate(
                prompt,
                model=str(self.params.get("model", config.classifier.model)),
                temperature=float(self.params.get("temperature", config.classifier.temperature)),
                seed=config.classifier.seed,
                num_ctx=config.classifier.num_ctx,
            )
            classification = UserClassification(**payload)
        except (LLMResponseError, LLMUnavailableError, ValidationError) as error:
            logger.warning("Classificação por LLM falhou; usando distribuição uniforme: %s", error)
            return uniform

        label = classification.classe.strip().lower()
        if label not in self.classes:
            logger.warning("Classe '%s' fora do vocabulário: resposta descartada.", label)
            return uniform

        # A confiança declarada pelo modelo vira a massa da classe escolhida, e
        # o restante é distribuído entre as demais. É uma probabilidade
        # autorrelatada, não calibrada — a análise de calibração
        # (evaluation.calibration) mede exatamente o quanto se pode confiar nela.
        confidence = float(np.clip(classification.confianca, 1.0 / len(self.classes), 0.99))
        probabilities = np.full(
            len(self.classes), (1.0 - confidence) / max(len(self.classes) - 1, 1)
        )
        probabilities[self.classes.index(label)] = confidence
        return probabilities / probabilities.sum()

    def predict_proba(self, dataset: UserDataset) -> np.ndarray:
        """Classifica todos os usuários do conjunto.

        Parameters
        ----------
        dataset : UserDataset
            Conjunto a classificar.

        Returns
        -------
        np.ndarray
            Matriz ``(n_usuarios, n_classes)``.

        Examples
        --------
        >>> modelo.predict_proba(teste).shape  # doctest: +SKIP
        (120, 3)
        """
        self.check_fitted()
        texts = dataset.require_texts()

        results = np.full(
            (len(dataset.user_ids), len(self.classes)), 1.0 / len(self.classes), dtype=np.float64
        )

        with build_progress() as progress:
            task = progress.add_task(f"Classificando com {self.name}", total=len(dataset.user_ids))
            for index, user_id in enumerate(dataset.user_ids):
                user_texts = texts.get(user_id, [])
                if user_texts:
                    results[index] = self._classify_user(user_texts)
                progress.advance(task)

        predicted = results.argmax(axis=1)
        distribution = {
            self.classes[value]: int(count)
            for value, count in zip(*np.unique(predicted, return_counts=True), strict=True)
        }
        logger.info("Distribuição prevista pelo LLM: %s", distribution)
        return results

    def describe(self) -> dict[str, Any]:
        """Descreve o modelo, incluindo a versão do prompt.

        Returns
        -------
        dict
            Metadados do modelo para o MLflow e o model card.

        Examples
        --------
        >>> modelo.describe()["prompt_version"]  # doctest: +SKIP
        '1.0.0'
        """
        description = super().describe()
        if self.llm_config is not None:
            description["prompt_version"] = self.llm_config.prompts.version
            description["mode"] = self.llm_config.classifier.mode
            description["n_few_shot_examples"] = len(self.examples_)
        description["trained"] = False
        description["note"] = (
            "Modelo não treinado: parte de conhecimento pré-treinado e usa o split de "
            f"treino apenas para os exemplos few-shot. Classe padrão em falha: "
            f"{UserLabel.CONTROLE.value}."
        )
        return description
