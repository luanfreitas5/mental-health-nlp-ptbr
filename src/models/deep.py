"""Modelos de deep learning sobre a sequência de tweets do usuário.

A BiLSTM da comparação principal opera sobre a **sequência cronológica de
embeddings de tweets** de cada usuário, não sobre tokens de um texto único.
A escolha é o ponto central da abordagem centrada no usuário: a rede
recorrente modela como o conteúdo evolui ao longo do tempo, que é exatamente
a informação que a classificação por tweet isolado descarta.

Concatenar todos os tweets num único texto e passá-lo por uma LSTM de tokens
seria a alternativa óbvia, e é justamente o que se quer evitar — a fronteira
entre publicações, e a ordem entre elas, desapareceria.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np

from config.environment import resolve_device
from config.logging import get_logger
from exceptions.model import MissingDependencyError, TrainingError, UnknownModelError
from models.base import BaseUserClassifier, UserDataset
from utils.progress import build_progress

logger = get_logger(__name__)


def _import_torch() -> Any:
    """Importa o PyTorch sob demanda, com erro explicativo se ausente."""
    try:
        import torch  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise MissingDependencyError(
            "PyTorch não está instalado. Rode 'make install-llm' para instalar os extras "
            "de deep learning."
        ) from error
    return torch


def build_sequence_batch(
    sequences: list[np.ndarray],
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Empilha sequências de comprimentos distintos num tensor retangular.

    Sequências mais longas que ``max_length`` são truncadas mantendo os
    tweets **mais recentes**: o estado atual do usuário é mais informativo
    para uma decisão de triagem do que publicações de um ano atrás.

    Parameters
    ----------
    sequences : list of np.ndarray
        Uma matriz ``(n_tweets, dim)`` por usuário.
    max_length : int
        Comprimento máximo da sequência.

    Returns
    -------
    tuple
        ``(tensor, máscara)`` com formatos ``(n, max_length, dim)`` e
        ``(n, max_length)``.

    Raises
    ------
    ValueError
        Se a lista estiver vazia.

    Examples
    --------
    >>> tensor, mascara = build_sequence_batch([np.ones((3, 4))], max_length=5)
    >>> tensor.shape, mascara.sum()
    ((1, 5, 4), 3.0)
    """
    if not sequences:
        raise ValueError("Não é possível montar um lote a partir de zero sequências.")

    dimension = sequences[0].shape[1]
    batch = np.zeros((len(sequences), max_length, dimension), dtype=np.float32)
    mask = np.zeros((len(sequences), max_length), dtype=np.float32)

    for index, sequence in enumerate(sequences):
        window = sequence[-max_length:]
        length = window.shape[0]
        batch[index, :length] = window
        mask[index, :length] = 1.0

    return batch, mask


@cache
def _build_recurrent_classifier_class(torch: Any) -> type:
    """Cria a classe do módulo recorrente uma única vez, em escopo de módulo.

    O torch é opcional e importado sob demanda, então a classe não pode ser
    declarada no topo do módulo. Só que uma classe aninhada dentro de uma
    função ganha um ``__qualname__`` com ``<locals>``, que o ``pickle``
    (usado por ``joblib.dump`` para persistir o modelo) não consegue
    resolver de volta a um atributo do módulo. Por isso ela é construída uma
    única vez — memoizada por ``functools.cache``, já que ``_import_torch()``
    sempre devolve o mesmo objeto de módulo — e publicada em ``globals()``
    sob o nome simples ``RecurrentClassifier``, tornando-a "pickleável".
    """
    nn = torch.nn

    class RecurrentClassifier(nn.Module):
        """LSTM (opcionalmente bidirecional) sobre sequências de embeddings."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int,
            n_classes: int,
            dropout: float,
            *,
            bidirectional: bool,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_dim * (2 if bidirectional else 1), n_classes)

        def forward(self, inputs: Any, mask: Any) -> Any:
            """Propaga o lote e devolve os logits por classe."""
            output, _ = self.lstm(inputs)
            # Pooling que respeita a máscara: incluir as posições de
            # preenchimento faria a média depender do comprimento da
            # sequência, e não do conteúdo.
            expanded = mask.unsqueeze(-1)
            pooled = (output * expanded).sum(dim=1) / expanded.sum(dim=1).clamp(min=1e-9)
            return self.head(self.dropout(pooled))

    RecurrentClassifier.__module__ = __name__
    RecurrentClassifier.__qualname__ = RecurrentClassifier.__name__
    globals()[RecurrentClassifier.__name__] = RecurrentClassifier
    return RecurrentClassifier


@cache
def _build_convolutional_classifier_class(torch: Any) -> type:
    """Cria a classe do módulo convolucional uma única vez, em escopo de módulo.

    Mesmo motivo de :func:`_build_recurrent_classifier_class`: a classe
    precisa viver em ``globals()`` sob um nome simples para que o ``pickle``
    consiga resolvê-la de volta ao carregar um modelo salvo.
    """
    nn = torch.nn

    class ConvolutionalClassifier(nn.Module):
        """TextCNN (Kim, 2014) sobre sequências de embeddings de tweets do usuário.

        Um ramo ``Conv1d`` por tamanho de kernel varre a sequência cronológica
        de embeddings do usuário (o mesmo objeto de entrada da BiLSTM, não
        tokens de um texto único), seguido de *max-pooling* mascarado no
        tempo — as posições de preenchimento não podem vencer o máximo. Os
        mapas de cada kernel são concatenados antes da cabeça linear.
        """

        def __init__(
            self,
            input_dim: int,
            num_filters: int,
            kernel_sizes: list[int],
            n_classes: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.convolutions = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=input_dim,
                        out_channels=num_filters,
                        kernel_size=kernel_size,
                        padding="same",
                    )
                    for kernel_size in kernel_sizes
                ]
            )
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(num_filters * len(kernel_sizes), n_classes)

        def forward(self, inputs: Any, mask: Any) -> Any:
            """Propaga o lote e devolve os logits por classe."""
            # Conv1d espera (lote, canais, tempo); a entrada chega como
            # (lote, tempo, dim_embedding).
            transposed = inputs.transpose(1, 2)
            # Preenchimento vira um valor bem negativo (não -inf, para evitar
            # NaN se uma sequência inteira ficasse mascarada) antes do
            # max-pooling: do contrário, o viés de cada Conv1d ainda produz
            # ativação não-nula nas posições de preenchimento, e o máximo no
            # tempo poderia ser "vencido" por um tweet que nunca existiu.
            invalid = (mask == 0).unsqueeze(1)
            pooled_features = []
            for convolution in self.convolutions:
                activated = self.activation(convolution(transposed))
                masked = activated.masked_fill(invalid, -1e4)
                pooled_features.append(masked.max(dim=2).values)
            concatenated = torch.cat(pooled_features, dim=1)
            return self.head(self.dropout(concatenated))

    ConvolutionalClassifier.__module__ = __name__
    ConvolutionalClassifier.__qualname__ = ConvolutionalClassifier.__name__
    globals()[ConvolutionalClassifier.__name__] = ConvolutionalClassifier
    return ConvolutionalClassifier


# Publica as classes em `globals()` já na importação do módulo — não apenas
# no primeiro `fit`/`predict_proba`. Sem isso, um processo que só faz
# `load_model` (como a etapa `evaluate` em execução isolada) desserializa
# antes de qualquer chamada que construiria a classe, e o `pickle` falha
# com "Can't get attribute 'RecurrentClassifier'" (ou 'ConvolutionalClassifier').
with contextlib.suppress(MissingDependencyError):
    _torch_module = _import_torch()
    _build_recurrent_classifier_class(_torch_module)
    _build_convolutional_classifier_class(_torch_module)


class _RecurrentNetwork:
    """Fábrica da arquitetura recorrente (definida sob demanda, com o torch importado)."""

    @staticmethod
    def build(
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        n_classes: int,
        dropout: float,
        *,
        bidirectional: bool,
    ) -> Any:
        """Constrói o módulo BiLSTM com *pooling* mascarado e cabeça linear."""
        torch = _import_torch()
        classifier_class = _build_recurrent_classifier_class(torch)
        return classifier_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            n_classes=n_classes,
            dropout=dropout,
            bidirectional=bidirectional,
        )


class _ConvolutionalNetwork:
    """Fábrica da arquitetura convolucional (definida sob demanda, com o torch importado)."""

    @staticmethod
    def build(
        input_dim: int,
        num_filters: int,
        kernel_sizes: list[int],
        n_classes: int,
        dropout: float,
    ) -> Any:
        """Constrói o módulo TextCNN com *pooling* mascarado e cabeça linear."""
        torch = _import_torch()
        classifier_class = _build_convolutional_classifier_class(torch)
        return classifier_class(
            input_dim=input_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            n_classes=n_classes,
            dropout=dropout,
        )


def build_sequence_network(
    estimator: str,
    input_dim: int,
    n_classes: int,
    params: dict[str, Any],
) -> Any:
    """Instancia a arquitetura sequencial a partir do nome configurado.

    Parameters
    ----------
    estimator : str
        Nome do estimador (``bilstm``, ``lstm`` ou ``cnn_text``).
    input_dim : int
        Dimensão dos embeddings de entrada.
    n_classes : int
        Número de classes.
    params : dict
        Hiperparâmetros de ``configs/model_params.yaml``.

    Returns
    -------
    Any
        Módulo PyTorch não treinado.

    Raises
    ------
    UnknownModelError
        Se o nome não estiver registrado.

    Examples
    --------
    >>> build_sequence_network(  # doctest: +SKIP
    ...     "cnn_text", input_dim=300, n_classes=3, params={"num_filters": 64}
    ... )
    """
    # 'lstm' nunca chega aqui hoje — a entrada 'lstm' de model_params.yaml usa
    # `estimator: bilstm` com hiperparâmetros diferentes —, mas é aceita aqui
    # porque `factory.SEQUENCE_ESTIMATORS` já a registra como alias válido.
    if estimator in {"bilstm", "lstm"}:
        return _RecurrentNetwork.build(
            input_dim=input_dim,
            hidden_dim=int(params.get("hidden_dim", 128)),
            num_layers=int(params.get("num_layers", 2)),
            n_classes=n_classes,
            dropout=float(params.get("dropout", 0.3)),
            bidirectional=bool(params.get("bidirectional", True)),
        )

    if estimator == "cnn_text":
        return _ConvolutionalNetwork.build(
            input_dim=input_dim,
            num_filters=int(params.get("num_filters", 128)),
            kernel_sizes=list(params.get("kernel_sizes", [3, 4, 5])),
            n_classes=n_classes,
            dropout=float(params.get("dropout", 0.5)),
        )

    raise UnknownModelError(
        f"Estimador sequencial desconhecido: '{estimator}'. Registrados: bilstm, lstm, cnn_text."
    )


@dataclass
class SequenceClassifier(BaseUserClassifier):
    """Classificador recorrente ou convolucional sobre sequências de embeddings.

    Parameters
    ----------
    name : str
        Nome do modelo.
    params : dict
        Hiperparâmetros de ``configs/model_params.yaml``: ``hidden_dim``,
        ``num_layers``, ``bidirectional`` (BiLSTM/LSTM) ou ``num_filters``,
        ``kernel_sizes`` (CNN), além dos comuns ``dropout``,
        ``learning_rate``, ``batch_size``, ``epochs``, ``patience``,
        ``max_tweets_per_user``.
    estimator_name : {'bilstm', 'lstm', 'cnn_text'}, optional
        Arquitetura a construir, by default ``'bilstm'``. Ver
        :func:`build_sequence_network`.

    Examples
    --------
    >>> modelo = SequenceClassifier(name="bilstm", params={"epochs": 5})
    >>> modelo.fit(treino)  # doctest: +SKIP
    >>> cnn = SequenceClassifier(name="cnn_text", params={"epochs": 5}, estimator_name="cnn_text")
    >>> cnn.fit(treino)  # doctest: +SKIP
    """

    estimator_name: str = "bilstm"
    model_: Any = field(default=None, init=False, repr=False)
    input_dim_: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Marca a dependência de sequências de embeddings."""
        self.requires_sequences = True

    def _hyperparameter(self, key: str, default: Any) -> Any:
        """Lê um hiperparâmetro com valor padrão."""
        return self.params.get(key, default)

    @staticmethod
    def _train_epoch(
        torch: Any,
        model: Any,
        tensor_inputs: Any,
        tensor_mask: Any,
        tensor_labels: Any,
        criterion: Any,
        optimizer: Any,
        batch_size: int,
    ) -> float:
        """Executa uma época de treino e devolve a perda média ponderada."""
        n_samples = tensor_inputs.shape[0]
        permutation = torch.randperm(n_samples, device=tensor_inputs.device)
        epoch_loss = 0.0

        for start in range(0, n_samples, batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(tensor_inputs[indices], tensor_mask[indices])
            loss = criterion(logits, tensor_labels[indices])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(indices)

        return epoch_loss / max(n_samples, 1)

    def fit(self, dataset: UserDataset) -> SequenceClassifier:
        """Treina a rede com parada antecipada por perda de validação.

        Parameters
        ----------
        dataset : UserDataset
            Conjunto de treino, com rótulos e sequências.

        Returns
        -------
        SequenceClassifier
            O próprio modelo, treinado.

        Raises
        ------
        TrainingError
            Se as sequências estiverem vazias ou com dimensões inconsistentes.

        Examples
        --------
        >>> modelo.fit(treino)  # doctest: +SKIP
        """
        self.validate_dataset(dataset)
        torch = _import_torch()

        sequences = dataset.require_sequences()
        assert dataset.labels is not None

        ordered = [sequences[user_id] for user_id in dataset.user_ids if user_id in sequences]
        if len(ordered) != len(dataset.user_ids):
            raise TrainingError(
                f"Sequências ausentes para {len(dataset.user_ids) - len(ordered)} usuário(s). "
                "Reexecute a etapa 'embed'."
            )

        max_length = int(self._hyperparameter("max_tweets_per_user", 200))
        batch_size = int(self._hyperparameter("batch_size", 16))
        epochs = int(self._hyperparameter("epochs", 30))
        patience = int(self._hyperparameter("patience", 5))
        seed = int(self._hyperparameter("random_state", 42))

        torch.manual_seed(seed)
        device = resolve_device("auto")

        inputs, mask = build_sequence_batch(ordered, max_length)
        self.input_dim_ = inputs.shape[2]

        self.model_ = build_sequence_network(
            self.estimator_name,
            input_dim=self.input_dim_,
            n_classes=len(self.classes),
            params=self.params,
        ).to(device)

        tensor_inputs = torch.from_numpy(inputs).to(device)
        tensor_mask = torch.from_numpy(mask).to(device)
        tensor_labels = torch.from_numpy(np.asarray(dataset.labels, dtype=np.int64)).to(device)

        # Pesos por classe: sem eles a rede converge para a classe majoritária
        # e o recall de `ideacao_suicida` — a métrica de maior custo clínico —
        # despenca.
        counts = np.bincount(np.asarray(dataset.labels), minlength=len(self.classes))
        weights = torch.tensor(
            (counts.sum() / np.maximum(counts, 1)) / len(self.classes),
            dtype=torch.float32,
            device=device,
        )

        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=float(self._hyperparameter("learning_rate", 1e-3))
        )

        best_loss, epochs_without_improvement = float("inf"), 0

        torch.set_grad_enabled(True)
        with build_progress() as progress:
            task = progress.add_task(f"Treinando {self.name}", total=epochs)
            for epoch in range(epochs):
                self.model_.train()
                epoch_loss = self._train_epoch(
                    torch,
                    self.model_,
                    tensor_inputs,
                    tensor_mask,
                    tensor_labels,
                    criterion,
                    optimizer,
                    batch_size,
                )

                if epoch_loss < best_loss - 1e-4:
                    best_loss, epochs_without_improvement = epoch_loss, 0
                else:
                    epochs_without_improvement += 1

                progress.advance(task)
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Parada antecipada na época %d (perda=%.4f).", epoch + 1, epoch_loss
                    )
                    break

        self.is_fitted = True
        logger.info("Modelo '%s' treinado (perda final=%.4f).", self.name, best_loss)
        return self

    def predict_proba(self, dataset: UserDataset) -> np.ndarray:
        """Prevê as probabilidades de cada classe.

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
        torch = _import_torch()

        sequences = dataset.require_sequences()
        dimension = self.input_dim_
        ordered = [
            sequences.get(user_id, np.zeros((1, dimension), dtype=np.float32))
            for user_id in dataset.user_ids
        ]

        inputs, mask = build_sequence_batch(
            ordered, int(self._hyperparameter("max_tweets_per_user", 200))
        )
        device = resolve_device("auto")

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(
                torch.from_numpy(inputs).to(device), torch.from_numpy(mask).to(device)
            )
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        return np.asarray(probabilities, dtype=np.float64)
