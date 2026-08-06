"""Treinamento final dos modelos e construção dos conjuntos de dados.

Aqui a matriz de atributos, os textos e as sequências de embeddings viram um
:class:`~models.base.UserDataset` por partição. A montagem acontece num único
lugar de propósito: todo modelo precisa ver exatamente os mesmos usuários em
cada partição, e replicar essa lógica por família de modelo é como
inconsistências de comparação aparecem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.logging import get_logger
from config.settings import Config
from constants.columns import SPLIT, TEXT_NORMALIZED, USER_ID, USER_LABEL
from constants.labels import LABEL_TO_INDEX
from data.reader import read_parquet, read_partitioned
from experiment.tracker import ExperimentTracker
from models.base import BaseUserClassifier, UserDataset
from models.persistence import save_model
from schemas.features import list_feature_columns
from utils.files import list_files
from utils.hashing import hash_dataframe
from utils.timing import log_duration

logger = get_logger(__name__)


def load_user_texts(
    tweets: pl.DataFrame, user_ids: list[str] | None = None
) -> dict[str, list[str]]:
    """Agrupa os tweets normalizados por usuário, em ordem cronológica.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos.
    user_ids : list of str, optional
        Restringe aos usuários informados.

    Returns
    -------
    dict of str to list of str
        Tweets por usuário.

    Examples
    --------
    >>> textos = load_user_texts(tweets)  # doctest: +SKIP
    >>> len(textos["u_ab12cd34"])  # doctest: +SKIP
    120
    """
    frame = tweets
    if user_ids is not None:
        frame = frame.filter(pl.col(USER_ID).is_in(user_ids))

    grouped = (
        frame.sort(["user_id", "created_at"])
        .group_by(USER_ID, maintain_order=True)
        .agg(pl.col(TEXT_NORMALIZED).alias("_textos"))
    )

    return {
        row[USER_ID]: [str(text) for text in row["_textos"]]
        for row in grouped.iter_rows(named=True)
    }


def load_user_sequences(
    embeddings_dir: Path,
    model_name: str,
    user_ids: list[str] | None = None,
) -> dict[str, np.ndarray] | None:
    """Carrega as sequências de embeddings por tweet, agrupadas por usuário.

    Parameters
    ----------
    embeddings_dir : Path
        Diretório com ``<modelo>.npy`` e ``<modelo>_index.parquet``.
    model_name : str
        Nome lógico do encoder (ex.: ``"bertimbau"``).
    user_ids : list of str, optional
        Restringe aos usuários informados.

    Returns
    -------
    dict of str to np.ndarray or None
        Sequência de vetores por usuário, ou ``None`` se os arquivos não
        existirem (a BiLSTM é então pulada, com aviso).

    Examples
    --------
    >>> load_user_sequences(Path("data/interim/embeddings"), "bertimbau")  # doctest: +SKIP
    """
    array_path = Path(embeddings_dir) / f"{model_name}.npy"
    index_path = Path(embeddings_dir) / f"{model_name}_index.parquet"

    if not array_path.is_file() or not index_path.is_file():
        logger.warning(
            "Embeddings por tweet não encontrados em %s: os modelos sequenciais serão "
            "pulados. Execute a etapa 'embed'.",
            embeddings_dir,
        )
        return None

    embeddings = np.load(array_path)
    index = pl.read_parquet(index_path)
    owners = index[USER_ID].to_list()

    wanted = set(user_ids) if user_ids is not None else None
    sequences: dict[str, list[np.ndarray]] = {}
    for position, owner in enumerate(owners):
        if wanted is not None and owner not in wanted:
            continue
        sequences.setdefault(owner, []).append(embeddings[position])

    return {owner: np.vstack(vectors) for owner, vectors in sequences.items()}


def build_dataset(
    features: pl.DataFrame,
    *,
    feature_groups: list[str] | None = None,
    texts: dict[str, list[str]] | None = None,
    sequences: dict[str, np.ndarray] | None = None,
) -> UserDataset:
    """Monta um :class:`UserDataset` a partir da matriz de atributos.

    Parameters
    ----------
    features : pl.DataFrame
        Matriz de atributos, com ``user_id`` e (opcionalmente) ``user_label``.
    feature_groups : list of str, optional
        Restringe a determinados grupos; ``None`` usa todos.
    texts : dict, optional
        Tweets por usuário.
    sequences : dict, optional
        Sequências de embeddings por usuário.

    Returns
    -------
    UserDataset
        Conjunto pronto para treino ou avaliação.

    Raises
    ------
    ValueError
        Se a matriz não contiver nenhuma coluna de atributos.

    Examples
    --------
    >>> build_dataset(matriz_treino, texts=textos)  # doctest: +SKIP
    """
    columns = list_feature_columns(features, feature_groups)
    if not columns:
        raise ValueError(
            "A matriz não possui colunas de atributos para os grupos solicitados: "
            f"{feature_groups or 'todos'}."
        )

    labels = None
    if USER_LABEL in features.columns:
        labels = np.array(
            [LABEL_TO_INDEX[value] for value in features[USER_LABEL].to_list()], dtype=np.int64
        )

    return UserDataset(
        user_ids=features[USER_ID].to_list(),
        features=features.select(columns).to_numpy().astype(np.float64),
        feature_names=columns,
        labels=labels,
        texts=texts,
        sequences=sequences,
    )


def split_features(features: pl.DataFrame, splits: pl.DataFrame, split: str) -> pl.DataFrame:
    """Recorta a matriz de atributos de uma partição.

    Parameters
    ----------
    features : pl.DataFrame
        Matriz completa.
    splits : pl.DataFrame
        Tabela de partições.
    split : str
        ``"train"``, ``"val"`` ou ``"test"``.

    Returns
    -------
    pl.DataFrame
        Subconjunto da matriz.

    Examples
    --------
    >>> split_features(matriz, particoes, "test")  # doctest: +SKIP
    """
    users = splits.filter(pl.col(SPLIT) == split)[USER_ID]
    return features.filter(pl.col(USER_ID).is_in(users)).sort(USER_ID)


def train_model(
    model: BaseUserClassifier,
    train_dataset: UserDataset,
    tracker: ExperimentTracker | None = None,
    dataset_hash: str = "",
) -> BaseUserClassifier:
    """Treina um modelo e registra os metadados da execução.

    Parameters
    ----------
    model : BaseUserClassifier
        Modelo instanciado pela fábrica.
    train_dataset : UserDataset
        Conjunto de treino.
    tracker : ExperimentTracker, optional
        Rastreador do MLflow.
    dataset_hash : str, optional
        Hash do dataset, registrado junto da execução.

    Returns
    -------
    BaseUserClassifier
        Modelo treinado.

    Examples
    --------
    >>> train_model(modelo, treino)  # doctest: +SKIP
    """
    with log_duration(f"Treinamento de '{model.name}'"):
        model.fit(train_dataset)

    if tracker is not None:
        tracker.log_params(model.describe())
        tracker.log_dataset(dataset_hash, len(train_dataset), train_dataset.features.shape[1])

    return model


def train_all(
    models: dict[str, BaseUserClassifier],
    train_features: pl.DataFrame,
    config: Config,
    paths_models: Path,
    *,
    texts: dict[str, list[str]] | None = None,
    sequences: dict[str, np.ndarray] | None = None,
    tracker: ExperimentTracker | None = None,
) -> dict[str, BaseUserClassifier]:
    """Treina todos os modelos selecionados e os persiste.

    Parameters
    ----------
    models : dict of str to BaseUserClassifier
        Modelos a treinar.
    train_features : pl.DataFrame
        Matriz de treino, com rótulo.
    config : Config
        Configuração completa do projeto.
    paths_models : Path
        Diretório onde persistir os modelos.
    texts : dict, optional
        Tweets por usuário (Transformers e LLM).
    sequences : dict, optional
        Sequências de embeddings (modelos recorrentes).
    tracker : ExperimentTracker, optional
        Rastreador do MLflow.

    Returns
    -------
    dict of str to BaseUserClassifier
        Modelos efetivamente treinados. Modelos que falharam são omitidos e
        registrados em log — uma falha isolada não pode custar a reexecução
        de toda a comparação.

    Examples
    --------
    >>> train_all(modelos, treino, config, Path("models/artifacts"))  # doctest: +SKIP
    """
    dataset_hash = hash_dataframe(train_features)[:16]
    trained: dict[str, BaseUserClassifier] = {}

    for name, model in models.items():
        spec = config.models.all_models().get(name)
        groups = spec.feature_groups if spec else None

        try:
            dataset = build_dataset(
                train_features,
                feature_groups=groups,
                texts=texts,
                sequences=sequences,
            )
        except ValueError:
            logger.exception("Conjunto de treino inválido para '%s'.", name)
            continue

        if model.requires_sequences and sequences is None:
            logger.warning("Modelo '%s' pulado: sequências de embeddings indisponíveis.", name)
            continue
        if model.requires_text and texts is None:
            logger.warning("Modelo '%s' pulado: textos dos usuários indisponíveis.", name)
            continue

        try:
            if tracker is not None:
                with tracker.run(name, tags={"stage": "train"}):
                    trained[name] = train_model(model, dataset, tracker, dataset_hash)
                    save_model(trained[name], paths_models, dataset_hash=dataset_hash)
            else:
                trained[name] = train_model(model, dataset, None, dataset_hash)
                save_model(trained[name], paths_models, dataset_hash=dataset_hash)
        except (ValueError, RuntimeError, MemoryError, OSError):
            logger.exception("Treinamento de '%s' falhou.", name)

    logger.info("Modelos treinados com sucesso: %s.", ", ".join(sorted(trained)) or "nenhum")
    return trained


def load_training_inputs(config: Config, paths: Any) -> dict[str, Any]:
    """Carrega os artefatos necessários à etapa de treinamento.

    Parameters
    ----------
    config : Config
        Configuração completa do projeto.
    paths : ProjectPaths
        Caminhos do projeto.

    Returns
    -------
    dict
        ``features``, ``splits``, ``texts`` e ``sequences``.

    Examples
    --------
    >>> load_training_inputs(config, paths)  # doctest: +SKIP
    """
    features = read_parquet(paths.data.user_features)
    splits = read_parquet(paths.data.splits)

    texts: dict[str, list[str]] | None = None
    tweets_path = paths.data.tweets_labeled
    if list_files(tweets_path, "*.parquet"):
        texts = load_user_texts(read_partitioned(tweets_path, stage="label"))
    else:
        logger.warning("Tweets rotulados ausentes: Transformers e LLM serão pulados.")

    sequences = load_user_sequences(
        paths.data.embeddings, config.features.semantic.primary_model.split("/")[-1]
    )

    return {"features": features, "splits": splits, "texts": texts, "sequences": sequences}
