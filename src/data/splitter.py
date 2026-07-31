"""Particionamento treino/validação/teste e folds de validação cruzada.

A unidade amostral é o **usuário**, nunca o tweet. Se tweets de uma mesma
pessoa caíssem em treino e em teste, o modelo aprenderia a reconhecer o autor
(estilo, vocabulário, temas recorrentes) em vez do sinal clínico — e a
métrica de teste ficaria inflada de um jeito que nenhuma inspeção de código
revelaria. Todo o módulo existe para tornar esse erro impossível.

A estratificação preserva a proporção das classes em cada partição, o que
importa especialmente para ``ideacao_suicida``, a classe minoritária e a de
maior custo clínico.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold, train_test_split

from config.logging import get_logger
from config.settings import CrossValidationSection, SplitSection
from constants.columns import FOLD, SPLIT, USER_ID, USER_LABEL
from constants.labels import Split
from exceptions.data import InsufficientDataError
from utils.validation import check_no_group_leakage

logger = get_logger(__name__)


def create_splits(
    users: pl.DataFrame,
    config: SplitSection,
    random_seed: int,
) -> pl.DataFrame:
    """Atribui cada usuário a uma partição (treino, validação ou teste).

    Parameters
    ----------
    users : pl.DataFrame
        Um registro por usuário, com ``user_id`` e ``user_label``.
    config : SplitSection
        Seção ``split`` de ``configs/config.yaml``.
    random_seed : int
        Semente da partição — fixa, para que a mesma divisão seja obtida em
        qualquer reexecução.

    Returns
    -------
    pl.DataFrame
        Colunas ``user_id``, ``user_label`` e ``split``.

    Raises
    ------
    InsufficientDataError
        Se alguma classe tiver menos usuários que o número de partições.

    Examples
    --------
    >>> usuarios = pl.DataFrame(
    ...     {"user_id": [f"u_{i}" for i in range(20)], "user_label": ["controle"] * 20}
    ... )
    >>> particoes = create_splits(usuarios, SplitSection(), random_seed=42)
    >>> sorted(particoes["split"].unique().to_list())
    ['test', 'train', 'val']
    """
    if users.is_empty():
        raise InsufficientDataError("Não há usuários para particionar.")

    counts = users.group_by(USER_LABEL).len().sort("len")
    smallest = int(counts["len"][0])
    if smallest < 3:
        raise InsufficientDataError(
            f"A classe menos frequente tem apenas {smallest} usuário(s): é impossível "
            "estratificar em treino, validação e teste. Colete mais dados."
        )

    user_ids: list[str] = users[USER_ID].to_list()
    labels = users[USER_LABEL].to_list()

    # 1) Separa o teste do restante.
    # `train_test_split` é genérico sobre o tipo do array de entrada; os
    # stubs não conseguem propagar `list[str]` através dele.
    train_val_ids, test_ids = cast(
        "tuple[list[str], list[str]]",
        train_test_split(
            user_ids,
            test_size=config.test_size,
            random_state=random_seed,
            shuffle=config.shuffle,
            stratify=labels,
        ),
    )

    # 2) Separa a validação de dentro do restante. A proporção é recalculada
    #    sobre o que sobrou, para que val_size continue sendo a fração do
    #    total (e não do subconjunto).
    label_by_id = dict(zip(user_ids, labels, strict=True))
    if config.val_size > 0:
        relative_val = config.val_size / (1.0 - config.test_size)
        train_ids, val_ids = cast(
            "tuple[list[str], list[str]]",
            train_test_split(
                train_val_ids,
                test_size=relative_val,
                random_state=random_seed,
                shuffle=config.shuffle,
                stratify=[label_by_id[user] for user in train_val_ids],
            ),
        )
    else:
        train_ids, val_ids = train_val_ids, []

    check_no_group_leakage(train_ids, test_ids)
    check_no_group_leakage(val_ids, test_ids)
    check_no_group_leakage(train_ids, val_ids)

    assignment = {
        **{user: str(Split.TRAIN) for user in train_ids},
        **{user: str(Split.VAL) for user in val_ids},
        **{user: str(Split.TEST) for user in test_ids},
    }

    result = users.select([USER_ID, USER_LABEL]).with_columns(
        pl.col(USER_ID).replace_strict(assignment, default=str(Split.TRAIN)).alias(SPLIT)
    )

    distribution = result.group_by(SPLIT).len().sort(SPLIT)
    logger.info("Partições criadas: %s", dict(distribution.iter_rows()))
    return result


def assign_folds(
    splits: pl.DataFrame,
    config: CrossValidationSection,
    random_seed: int,
) -> pl.DataFrame:
    """Atribui folds de validação cruzada aos usuários de treino e validação.

    O teste recebe ``fold = -1`` e nunca participa da validação cruzada: o
    conjunto de teste é tocado uma única vez, na avaliação final.

    Parameters
    ----------
    splits : pl.DataFrame
        Saída de :func:`create_splits`.
    config : CrossValidationSection
        Seção ``cross_validation`` de ``configs/config.yaml``.
    random_seed : int
        Semente dos folds.

    Returns
    -------
    pl.DataFrame
        ``splits`` acrescido da coluna ``fold``.

    Raises
    ------
    InsufficientDataError
        Se alguma classe tiver menos usuários que o número de folds.

    Examples
    --------
    >>> com_folds = assign_folds(particoes, CrossValidationSection(), 42)  # doctest: +SKIP
    >>> com_folds["fold"].min()  # doctest: +SKIP
    -1
    """
    development = splits.filter(pl.col(SPLIT) != str(Split.TEST))
    if development.is_empty():
        raise InsufficientDataError("Não há usuários de desenvolvimento para gerar folds.")

    class_counts = development.group_by(USER_LABEL).len()
    # `Series.min()` é tipado de forma genérica nos stubs do Polars (pode
    # devolver qualquer `PythonLiteral`); a coluna `len()` é sempre inteira,
    # e `development` não está vazio, então sempre há ao menos um grupo.
    smallest = cast(int, class_counts["len"].min())
    if smallest < config.n_splits:
        raise InsufficientDataError(
            f"A classe menos frequente tem {smallest} usuário(s), abaixo dos "
            f"{config.n_splits} folds configurados. Reduza cross_validation.n_splits "
            "ou colete mais dados."
        )

    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=random_seed,
    )

    dev_ids = development[USER_ID].to_list()
    dev_labels = development[USER_LABEL].to_list()

    fold_by_user: dict[str, int] = {}
    placeholder = np.zeros(len(dev_ids))
    for fold_index, (_, validation_index) in enumerate(splitter.split(placeholder, dev_labels)):
        for position in validation_index:
            fold_by_user[dev_ids[position]] = fold_index

    result = splits.with_columns(
        pl.col(USER_ID).replace_strict(fold_by_user, default=-1, return_dtype=pl.Int64).alias(FOLD)
    )

    logger.info(
        "Folds atribuídos: %d folds sobre %d usuários de desenvolvimento.",
        config.n_splits,
        development.height,
    )
    return result


def build_split_table(
    users: pl.DataFrame,
    split_config: SplitSection,
    cv_config: CrossValidationSection,
    random_seed: int,
) -> pl.DataFrame:
    """Constrói a tabela completa de partições e folds.

    Parameters
    ----------
    users : pl.DataFrame
        Um registro por usuário, com ``user_id`` e ``user_label``.
    split_config : SplitSection
        Configuração das partições.
    cv_config : CrossValidationSection
        Configuração da validação cruzada.
    random_seed : int
        Semente global.

    Returns
    -------
    pl.DataFrame
        Colunas ``user_id``, ``user_label``, ``split`` e ``fold``.

    Examples
    --------
    >>> build_split_table(usuarios, split_cfg, cv_cfg, 42)  # doctest: +SKIP
    """
    splits = create_splits(users, split_config, random_seed)
    return assign_folds(splits, cv_config, random_seed)


def filter_split(frame: pl.DataFrame, splits: pl.DataFrame, split: str) -> pl.DataFrame:
    """Seleciona as linhas pertencentes a uma partição.

    Parameters
    ----------
    frame : pl.DataFrame
        Matriz de atributos (uma linha por usuário).
    splits : pl.DataFrame
        Tabela de partições.
    split : str
        ``"train"``, ``"val"`` ou ``"test"``.

    Returns
    -------
    pl.DataFrame
        Subconjunto correspondente à partição.

    Examples
    --------
    >>> filter_split(features, particoes, "train")  # doctest: +SKIP
    """
    wanted = splits.filter(pl.col(SPLIT) == split)[USER_ID]
    return frame.filter(pl.col(USER_ID).is_in(wanted))
