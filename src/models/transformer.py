"""Fine-tuning de Transformers (BERTimbau) com agregação por usuário.

A estratégia é hierárquica em duas etapas: o modelo é ajustado no nível do
**tweet**, herdando o rótulo do usuário, e as probabilidades dos tweets são
agregadas para produzir a decisão no nível do **usuário**.

A alternativa — um Transformer hierárquico que consome o histórico inteiro de
uma vez — exigiria contexto muito maior que 512 tokens e um orçamento de GPU
incompatível com o prazo do mestrado. A agregação de probabilidades entrega a
maior parte do ganho a uma fração do custo, e a limitação está documentada no
model card.

O rótulo herdado introduz ruído conhecido: nem todo tweet de um usuário com
depressão expressa depressão. É por isso que a agregação usa a **média das
probabilidades**, e não o voto majoritário — a média é muito mais robusta a
rótulos individuais ruidosos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config.environment import resolve_device
from config.logging import get_logger
from exceptions.model import MissingDependencyError, ModelError
from models.base import BaseUserClassifier, UserDataset
from utils.progress import build_progress

logger = get_logger(__name__)


def _import_transformer_stack() -> tuple[Any, Any]:
    """Importa PyTorch e Transformers sob demanda."""
    try:
        import torch
        import transformers
    except ImportError as error:
        raise MissingDependencyError(
            "PyTorch/Transformers ausentes. Rode 'make install-llm' para instalar os extras de LLM."
        ) from error
    return torch, transformers


def flatten_user_texts(
    texts: dict[str, list[str]],
    user_ids: list[str],
    labels: np.ndarray | None = None,
) -> tuple[list[str], list[str], np.ndarray | None]:
    """Achata os tweets dos usuários numa lista única, propagando o rótulo.

    Parameters
    ----------
    texts : dict of str to list of str
        Tweets por usuário.
    user_ids : list of str
        Usuários a incluir, na ordem desejada.
    labels : np.ndarray, optional
        Rótulo de cada usuário, na mesma ordem.

    Returns
    -------
    tuple
        ``(textos, autor_de_cada_texto, rótulos_por_texto)``.

    Examples
    --------
    >>> flatten_user_texts({"u_a": ["x", "y"]}, ["u_a"], np.array([1]))[2].tolist()
    [1, 1]
    """
    flat_texts: list[str] = []
    owners: list[str] = []
    flat_labels: list[int] = []

    for position, user_id in enumerate(user_ids):
        user_texts = texts.get(user_id, [])
        flat_texts.extend(user_texts)
        owners.extend([user_id] * len(user_texts))
        if labels is not None:
            flat_labels.extend([int(labels[position])] * len(user_texts))

    return flat_texts, owners, (np.array(flat_labels) if labels is not None else None)


def aggregate_user_probabilities(
    probabilities: np.ndarray,
    owners: list[str],
    user_ids: list[str],
    n_classes: int,
    strategy: str = "mean",
) -> np.ndarray:
    """Agrega probabilidades de tweets em uma decisão por usuário.

    Parameters
    ----------
    probabilities : np.ndarray
        Matriz ``(n_tweets, n_classes)``.
    owners : list of str
        Autor de cada tweet, na mesma ordem.
    user_ids : list of str
        Ordem desejada dos usuários na saída.
    n_classes : int
        Número de classes.
    strategy : {'mean', 'max', 'majority'}, optional
        Forma de agregação, by default ``'mean'``.

    Returns
    -------
    np.ndarray
        Matriz ``(n_usuarios, n_classes)``. Usuários sem tweets recebem
        distribuição uniforme, e não zeros — zerar todas as classes tornaria
        o ``argmax`` arbitrário.

    Examples
    --------
    >>> agregado = aggregate_user_probabilities(
    ...     np.array([[0.2, 0.8], [0.6, 0.4]]), ["u_a", "u_a"], ["u_a"], 2
    ... )
    >>> agregado.round(2).tolist()
    [[0.4, 0.6]]
    """
    grouped: dict[str, list[np.ndarray]] = {}
    for row, owner in enumerate(owners):
        grouped.setdefault(owner, []).append(probabilities[row])

    result = np.full((len(user_ids), n_classes), 1.0 / n_classes, dtype=np.float64)
    for index, user_id in enumerate(user_ids):
        rows = grouped.get(user_id)
        if not rows:
            continue
        stacked = np.vstack(rows)

        if strategy == "max":
            aggregated = stacked.max(axis=0)
        elif strategy == "majority":
            votes = np.bincount(stacked.argmax(axis=1), minlength=n_classes)
            aggregated = votes / votes.sum()
        else:
            aggregated = stacked.mean(axis=0)

        total = aggregated.sum()
        result[index] = aggregated / total if total > 0 else result[index]

    return result


@dataclass
class TransformerClassifier(BaseUserClassifier):
    """Transformer fine-tuned no nível do tweet, agregado por usuário.

    Parameters
    ----------
    name : str
        Nome do modelo.
    params : dict
        Hiperparâmetros (``model_name``, ``max_length``, ``learning_rate``,
        ``batch_size``, ``epochs``, ``user_aggregation``, ``fp16``, ...).
        ``fp16`` ativa precisão mista (Tensor Cores) durante o fine-tuning e
        a inferência; é ignorado (com aviso) fora de GPU CUDA.

    Examples
    --------
    >>> modelo = TransformerClassifier(name="bertimbau", params={"epochs": 2})
    >>> modelo.fit(treino)  # doctest: +SKIP
    """

    model_: Any = field(default=None, init=False, repr=False)
    tokenizer_: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Marca a dependência dos textos dos usuários."""
        self.requires_text = True

    def _hyperparameter(self, key: str, default: Any) -> Any:
        """Lê um hiperparâmetro com valor padrão."""
        return self.params.get(key, default)

    def _resolve_fp16(self, device: str) -> bool:
        """Decide se a precisão mista (fp16) deve ser usada.

        Autocast fp16 só acelera em Tensor Cores de GPU CUDA; em CPU ele não
        traz ganho e pode até ser mais lento, então a flag é ignorada (com
        aviso) fora de ``cuda``.
        """
        requested = bool(self._hyperparameter("fp16", False))
        if requested and device != "cuda":
            logger.warning(
                "fp16 solicitado para '%s', mas o dispositivo é '%s': treinando em precisão total.",
                self.name,
                device,
            )
            return False
        return requested

    def _load(self) -> tuple[Any, Any]:
        """Carrega tokenizador e modelo de classificação."""
        if self.model_ is not None and self.tokenizer_ is not None:
            return self.tokenizer_, self.model_

        _, transformers = _import_transformer_stack()
        model_name = str(
            self._hyperparameter("model_name", "neuralmind/bert-base-portuguese-cased")
        )

        try:
            # `model_name` vem do hiperparâmetro configurado (definido pelo
            # operador do projeto, não por entrada externa/de usuário); fixar
            # uma revisão exigiria manter um mapa de hashes por modelo
            # configurável, o que não se encaixa no design orientado a YAML.
            self.tokenizer_ = transformers.AutoTokenizer.from_pretrained(model_name)  # nosec B615
            self.model_ = transformers.AutoModelForSequenceClassification.from_pretrained(  # nosec B615
                model_name, num_labels=len(self.classes)
            )
        except (OSError, ValueError) as error:
            raise ModelError(
                f"Não foi possível carregar o Transformer '{model_name}': {error}"
            ) from error

        logger.info("Transformer carregado: %s.", model_name)
        return self.tokenizer_, self.model_

    def fit(self, dataset: UserDataset) -> TransformerClassifier:
        """Ajusta o Transformer no nível do tweet.

        Parameters
        ----------
        dataset : UserDataset
            Conjunto de treino, com rótulos e textos.

        Returns
        -------
        TransformerClassifier
            O próprio modelo, treinado.

        Examples
        --------
        >>> modelo.fit(treino)  # doctest: +SKIP
        """
        self.validate_dataset(dataset)
        torch, _ = _import_transformer_stack()

        assert dataset.labels is not None
        texts, _, tweet_labels = flatten_user_texts(
            dataset.require_texts(), dataset.user_ids, dataset.labels
        )
        assert tweet_labels is not None

        tokenizer, model = self._load()
        device = resolve_device("auto")
        model.to(device)
        model.train()

        seed = int(self._hyperparameter("random_state", 42))
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        batch_size = int(self._hyperparameter("batch_size", 16))
        epochs = int(self._hyperparameter("epochs", 4))
        max_length = int(self._hyperparameter("max_length", 128))
        use_fp16 = self._resolve_fp16(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self._hyperparameter("learning_rate", 2e-5)),
            weight_decay=float(self._hyperparameter("weight_decay", 0.01)),
        )
        # `enabled=False` faz o GradScaler operar como no-op (scale=1.0):
        # mesmo laço de código serve para fp16 e para precisão total.
        scaler = torch.amp.GradScaler(device, enabled=use_fp16)

        counts = np.bincount(tweet_labels, minlength=len(self.classes))
        weights = torch.tensor(
            (counts.sum() / np.maximum(counts, 1)) / len(self.classes),
            dtype=torch.float32,
            device=device,
        )
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        n_samples = len(texts)
        torch.set_grad_enabled(True)

        with build_progress() as progress:
            task = progress.add_task(f"Fine-tuning {self.name}", total=epochs * n_samples)
            for _ in range(epochs):
                order = rng.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    indices = order[start : start + batch_size]
                    batch_texts = [texts[i] for i in indices]

                    encoded = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    ).to(device)
                    targets = torch.from_numpy(tweet_labels[indices]).long().to(device)

                    optimizer.zero_grad()
                    with torch.amp.autocast(
                        device_type=device, dtype=torch.float16, enabled=use_fp16
                    ):
                        logits = model(**encoded).logits
                        loss = criterion(logits, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    progress.advance(task, advance=len(indices))

        self.is_fitted = True
        logger.info("Transformer '%s' ajustado sobre %d tweets.", self.name, n_samples)
        return self

    def predict_proba(self, dataset: UserDataset) -> np.ndarray:
        """Prevê as probabilidades por usuário, agregando as dos tweets.

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
        torch, _ = _import_transformer_stack()

        texts, owners, _ = flatten_user_texts(dataset.require_texts(), dataset.user_ids)
        if not texts:
            return np.full((len(dataset.user_ids), len(self.classes)), 1.0 / len(self.classes))

        tokenizer, model = self._load()
        device = resolve_device("auto")
        model.to(device)
        model.eval()

        batch_size = int(self._hyperparameter("batch_size", 16))
        max_length = int(self._hyperparameter("max_length", 128))
        use_fp16 = self._resolve_fp16(device)
        chunks: list[np.ndarray] = []

        with torch.no_grad(), build_progress() as progress:
            task = progress.add_task(f"Inferência {self.name}", total=len(texts))
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=use_fp16):
                    logits = model(**encoded).logits
                # Softmax em fp32: converter antes evita perda de precisão da
                # normalização em probabilidades calculadas a partir de logits fp16.
                chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
                progress.advance(task, advance=len(batch_texts))

        return aggregate_user_probabilities(
            np.vstack(chunks),
            owners,
            dataset.user_ids,
            len(self.classes),
            strategy=str(self._hyperparameter("user_aggregation", "mean")),
        )
