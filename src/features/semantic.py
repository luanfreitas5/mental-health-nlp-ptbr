"""Embeddings semânticos e sua agregação por usuário (prefixo ``sem_``).

Cobre a Seção 3 da proposta. Cada tweet é codificado por um encoder
Transformer e os vetores são agregados no nível do usuário.

Decisões que merecem justificativa:

* **Média + desvio como agregação.** A média descreve o "centro semântico" do
  perfil; o desvio descreve a dispersão temática. Perfis com sofrimento
  psíquico persistente tendem a ser tematicamente mais concentrados, e essa
  informação se perde se apenas a média for mantida.
* **Redução por PCA ajustada só no treino.** 768 dimensões (× 2 agregações)
  contra algumas centenas de usuários é um convite ao sobreajuste. O PCA é
  parte do ``Pipeline``, ajustado exclusivamente na partição de treino — se
  fosse ajustado sobre tudo, a estrutura de covariância do teste vazaria para
  o modelo.
* **Normalização L2.** Torna os vetores comparáveis por cosseno e evita que a
  norma (correlacionada com o comprimento do texto) domine a distância.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.environment import resolve_device
from config.logging import get_logger
from config.settings import SemanticSection
from constants.columns import SEMANTIC_PREFIX, TEXT_NORMALIZED, USER_ID
from exceptions.model import MissingDependencyError, ModelError
from utils.progress import build_progress
from utils.validation import require_columns

logger = get_logger(__name__)


class EmbeddingEncoder:
    """Codificador de textos em vetores densos.

    Parameters
    ----------
    model_name : str
        Identificador do modelo no Hugging Face Hub.
    config : SemanticSection
        Seção ``semantic`` de ``configs/features.yaml``.

    Examples
    --------
    >>> encoder = EmbeddingEncoder("neuralmind/bert-base-portuguese-cased", cfg)  # doctest: +SKIP
    >>> encoder.encode(["hoje foi um dia difícil"]).shape  # doctest: +SKIP
    (1, 768)
    """

    def __init__(self, model_name: str, config: SemanticSection) -> None:
        self.model_name = model_name
        self.config = config
        self.device = resolve_device(config.device)
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        """Carrega tokenizador e modelo em modo de inferência."""
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            import torch  # pyright: ignore[reportMissingImports]
            from transformers import (  # pyright: ignore[reportMissingImports]
                AutoModel,
                AutoTokenizer,
            )
        except ImportError as error:
            raise MissingDependencyError(
                "PyTorch/Transformers ausentes. Rode 'make install-llm' para instalar os "
                "extras de LLM."
            ) from error

        try:
            # `model_name` vem de `configs/features.yaml` (definido pelo
            # operador do projeto, não por entrada externa/de usuário); fixar
            # uma revisão exigiria manter um mapa de hashes por modelo
            # configurável, o que não se encaixa no design orientado a YAML.
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # nosec B615
            model = AutoModel.from_pretrained(self.model_name)  # nosec B615
        except (OSError, ValueError) as error:
            raise ModelError(
                f"Não foi possível carregar o encoder '{self.model_name}': {error}"
            ) from error

        model.to(self.device)
        model.eval()
        torch.set_grad_enabled(False)

        self._tokenizer, self._model = tokenizer, model
        logger.info(
            "Encoder semântico carregado: %s (device=%s, precision=%s).",
            self.model_name,
            self.device,
            self.config.precision,
        )
        return self._tokenizer, self._model

    def _resolve_autocast_dtype(self) -> Any | None:
        """Decide o dtype de autocast do forward pass.

        Autocast (fp16/bf16) só acelera em Tensor Cores de GPU CUDA; fora
        disso é ignorado (com aviso) e o encoder roda em fp32.
        """
        if self.config.precision == "fp32":
            return None
        if self.device != "cuda":
            logger.warning(
                "Precisão '%s' solicitada para '%s', mas o dispositivo é '%s': "
                "codificando em precisão total.",
                self.config.precision,
                self.model_name,
                self.device,
            )
            return None

        import torch  # pyright: ignore[reportMissingImports]

        return torch.bfloat16 if self.config.precision == "bf16" else torch.float16

    def encode(self, texts: list[str]) -> np.ndarray:
        """Codifica uma lista de textos.

        O *mean pooling* respeita a máscara de atenção: incluir as posições de
        preenchimento na média deslocaria o vetor em função do comprimento do
        texto, e não do seu conteúdo.

        Parameters
        ----------
        texts : list of str
            Textos normalizados.

        Returns
        -------
        np.ndarray
            Matriz ``(n_textos, dim)`` de vetores.

        Examples
        --------
        >>> encoder.encode(["texto"]).ndim  # doctest: +SKIP
        2
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        import torch  # pyright: ignore[reportMissingImports]

        tokenizer, model = self._load()
        autocast_dtype = self._resolve_autocast_dtype()
        vectors: list[np.ndarray] = []

        with build_progress() as progress:
            task = progress.add_task(
                f"Embeddings ({self.model_name.split('/')[-1]})", total=len(texts)
            )
            for start in range(0, len(texts), self.config.batch_size):
                batch = texts[start : start + self.config.batch_size]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                ).to(self.device)

                # Autocast só envolve o forward pass: menos tempo por lote e
                # memória livre para lotes maiores. O restante (pooling,
                # normalização) roda em fp32 — `.float()` evita que a perda
                # de precisão do fp16/bf16 se propague para a similaridade
                # por cosseno entre embeddings.
                with torch.amp.autocast(
                    device_type=self.device,
                    dtype=autocast_dtype or torch.float16,
                    enabled=autocast_dtype is not None,
                ):
                    output = model(**encoded).last_hidden_state
                if autocast_dtype is not None:
                    output = output.float()

                if self.config.pooling == "cls":
                    pooled = output[:, 0, :]
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1).float()
                    pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

                if self.config.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

                vectors.append(pooled.cpu().numpy().astype(np.float32))
                progress.advance(task, advance=len(batch))

        return np.vstack(vectors)


def aggregate_embeddings(
    embeddings: np.ndarray,
    user_ids: list[str],
    aggregations: list[str],
) -> pl.DataFrame:
    """Agrega os embeddings dos tweets no nível do usuário.

    Parameters
    ----------
    embeddings : np.ndarray
        Matriz ``(n_tweets, dim)``.
    user_ids : list of str
        Identificador do autor de cada linha, na mesma ordem.
    aggregations : list of str
        Agregações a aplicar (``mean``, ``std``, ``max``).

    Returns
    -------
    pl.DataFrame
        Uma linha por usuário, com colunas ``sem_<agregacao>_<i>``.

    Raises
    ------
    ValueError
        Se o número de vetores não corresponder ao de identificadores.

    Examples
    --------
    >>> aggregate_embeddings(np.ones((2, 3)), ["u_a", "u_a"], ["mean"]).width
    4
    """
    if len(user_ids) != embeddings.shape[0]:
        raise ValueError(
            f"Incompatibilidade: {embeddings.shape[0]} vetores para {len(user_ids)} "
            "identificadores de usuário."
        )

    order: list[str] = []
    index_by_user: dict[str, list[int]] = {}
    for position, user_id in enumerate(user_ids):
        if user_id not in index_by_user:
            index_by_user[user_id] = []
            order.append(user_id)
        index_by_user[user_id].append(position)

    dimension = embeddings.shape[1]
    columns: dict[str, list[float]] = {USER_ID: order}  # type: ignore[dict-item]

    for aggregation in aggregations:
        matrix = np.zeros((len(order), dimension), dtype=np.float32)
        for row, user_id in enumerate(order):
            block = embeddings[index_by_user[user_id]]
            if aggregation == "mean":
                matrix[row] = block.mean(axis=0)
            elif aggregation == "std":
                # ddof=0: com um único tweet o desvio é 0, e não NaN.
                matrix[row] = block.std(axis=0)
            elif aggregation == "max":
                matrix[row] = block.max(axis=0)
            else:
                raise ValueError(
                    f"Agregação de embedding não suportada: '{aggregation}'. Use mean, std ou max."
                )

        for dim in range(dimension):
            columns[f"{SEMANTIC_PREFIX}{aggregation}_{dim:03d}"] = matrix[:, dim].tolist()

    return pl.DataFrame(columns).sort(USER_ID)


def build_semantic_features(
    tweets: pl.DataFrame,
    config: SemanticSection,
    model_name: str | None = None,
) -> pl.DataFrame:
    """Gera e agrega os embeddings semânticos por usuário.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos, com ``user_id`` e ``text_normalized``.
    config : SemanticSection
        Seção ``semantic`` de ``configs/features.yaml``.
    model_name : str, optional
        Sobrescreve ``semantic.primary_model`` (usado na comparação entre
        encoders da extensão exploratória).

    Returns
    -------
    pl.DataFrame
        Uma linha por usuário, com as colunas de prefixo ``sem_``.

    Examples
    --------
    >>> build_semantic_features(tweets, config.features.semantic)  # doctest: +SKIP
    """
    require_columns(tweets, [USER_ID, TEXT_NORMALIZED], context="features semânticas")

    ordered = tweets.sort(USER_ID)
    encoder = EmbeddingEncoder(model_name or config.primary_model, config)
    embeddings = encoder.encode(ordered[TEXT_NORMALIZED].to_list())

    result = aggregate_embeddings(embeddings, ordered[USER_ID].to_list(), config.user_aggregations)

    logger.info(
        "Features semânticas: %d colunas para %d usuários (modelo=%s).",
        result.width - 1,
        result.height,
        encoder.model_name,
    )
    return result


def save_embeddings(
    embeddings: np.ndarray, user_ids: list[str], directory: Path, name: str
) -> Path:
    """Persiste os embeddings brutos para reuso entre experimentos.

    Gerar embeddings é a etapa mais cara do pipeline; salvá-los permite
    comparar diferentes cabeças de classificação sem recodificar milhões de
    tweets a cada execução.

    Parameters
    ----------
    embeddings : np.ndarray
        Matriz de vetores.
    user_ids : list of str
        Autor de cada linha.
    directory : Path
        Diretório de destino.
    name : str
        Nome lógico do modelo (ex.: ``"bertimbau"``).

    Returns
    -------
    Path
        Caminho do arquivo ``.npy`` gravado.

    Examples
    --------
    >>> save_embeddings(matriz, ids, Path("data/interim/embeddings"), "bertimbau")  # doctest: +SKIP
    """
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    array_path = target_dir / f"{name}.npy"
    np.save(array_path, embeddings)

    index_path = target_dir / f"{name}_index.parquet"
    pl.DataFrame({USER_ID: user_ids}).write_parquet(index_path)

    logger.info("Embeddings salvos em %s (%s).", array_path, embeddings.shape)
    return array_path
