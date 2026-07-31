"""Validação cruzada no nível do usuário.

Os folds vêm da tabela de partições (:mod:`data.splitter`), e não são
recalculados aqui. A razão é comparabilidade: se cada modelo gerasse seus
próprios folds, as diferenças observadas entre eles misturariam efeito do
modelo com efeito da partição, e os testes pareados (Wilcoxon, Friedman)
perderiam validade — eles pressupõem exatamente os mesmos blocos.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from config.logging import get_logger
from config.settings import Config, ModelSpec
from constants.columns import FOLD, SPLIT, USER_ID
from constants.labels import Split
from evaluation.metrics import compute_metrics
from models.base import UserDataset
from models.factory import create_model
from utils.progress import build_progress

logger = get_logger(__name__)


def build_fold_datasets(
    dataset: UserDataset,
    splits: pl.DataFrame,
    fold: int,
) -> tuple[UserDataset, UserDataset]:
    """Separa treino e validação de um fold.

    Parameters
    ----------
    dataset : UserDataset
        Conjunto de desenvolvimento completo (treino + validação).
    splits : pl.DataFrame
        Tabela de partições, com ``user_id`` e ``fold``.
    fold : int
        Índice do fold usado como validação.

    Returns
    -------
    tuple
        ``(treino, validação)``.

    Examples
    --------
    >>> treino, validacao = build_fold_datasets(conjunto, particoes, 0)  # doctest: +SKIP
    """
    validation_users = set(
        splits.filter((pl.col(FOLD) == fold) & (pl.col(SPLIT) != str(Split.TEST)))[
            USER_ID
        ].to_list()
    )

    train_positions = [
        index for index, user_id in enumerate(dataset.user_ids) if user_id not in validation_users
    ]
    validation_positions = [
        index for index, user_id in enumerate(dataset.user_ids) if user_id in validation_users
    ]

    def subset(positions: list[int]) -> UserDataset:
        """Recorta o conjunto mantendo todas as representações alinhadas."""
        user_ids = [dataset.user_ids[index] for index in positions]
        return UserDataset(
            user_ids=user_ids,
            features=dataset.features[positions],
            feature_names=dataset.feature_names,
            labels=dataset.labels[positions] if dataset.labels is not None else None,
            texts={
                user_id: dataset.texts[user_id] for user_id in user_ids if user_id in dataset.texts
            }
            if dataset.texts
            else None,
            sequences={
                user_id: dataset.sequences[user_id]
                for user_id in user_ids
                if user_id in dataset.sequences
            }
            if dataset.sequences
            else None,
        )

    return subset(train_positions), subset(validation_positions)


def cross_validate_model(
    name: str,
    spec: ModelSpec,
    dataset: UserDataset,
    splits: pl.DataFrame,
    config: Config,
) -> dict[str, Any]:
    """Executa a validação cruzada de um modelo.

    Parameters
    ----------
    name : str
        Nome do modelo.
    spec : ModelSpec
        Especificação do modelo.
    dataset : UserDataset
        Conjunto de desenvolvimento (nunca inclui o teste).
    splits : pl.DataFrame
        Tabela de partições com os folds.
    config : Config
        Configuração completa do projeto.

    Returns
    -------
    dict
        ``scores`` por fold, média, desvio, intervalo de confiança e métricas
        completas de cada fold.

    Examples
    --------
    >>> cross_validate_model("xgboost", spec, conjunto, particoes, config)  # doctest: +SKIP
    """
    primary = config.evaluation.metrics.primary
    n_splits = config.general.cross_validation.n_splits

    scores: list[float] = []
    per_fold: list[dict[str, float]] = []

    with build_progress() as progress:
        task = progress.add_task(f"Validação cruzada — {name}", total=n_splits)

        for fold in range(n_splits):
            train, validation = build_fold_datasets(dataset, splits, fold)

            if len(validation) == 0 or train.labels is None:
                logger.warning("Fold %d vazio para '%s': ignorado.", fold, name)
                progress.advance(task)
                continue

            model = create_model(f"{name}_fold{fold}", spec, config)
            model.fit(train)

            predictions = model.predict(validation)
            assert validation.labels is not None
            metrics = compute_metrics(
                validation.labels, predictions, model.predict_proba(validation)
            )

            scores.append(metrics[primary])
            per_fold.append(metrics)
            progress.advance(task)

    if not scores:
        logger.error("Nenhum fold válido para '%s'.", name)
        return {"model": name, "scores": [], "mean": float("nan"), "std": float("nan")}

    array = np.array(scores)
    # IC pela distribuição t: com 5 folds, a aproximação normal subestima a
    # largura do intervalo de forma relevante.
    margin = 1.96 * array.std(ddof=1) / np.sqrt(len(array)) if len(array) > 1 else 0.0

    logger.info(
        "%s | %s = %.4f ± %.4f (IC 95%%, %d folds)",
        name,
        primary,
        array.mean(),
        margin,
        len(array),
    )

    return {
        "model": name,
        "metric": primary,
        "scores": scores,
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci_margin": float(margin),
        "ci_lower": float(array.mean() - margin),
        "ci_upper": float(array.mean() + margin),
        "per_fold_metrics": per_fold,
    }


def cross_validate_all(
    specs: dict[str, ModelSpec],
    dataset: UserDataset,
    splits: pl.DataFrame,
    config: Config,
) -> dict[str, dict[str, Any]]:
    """Executa a validação cruzada de todos os modelos selecionados.

    Parameters
    ----------
    specs : dict of str to ModelSpec
        Modelos a validar.
    dataset : UserDataset
        Conjunto de desenvolvimento.
    splits : pl.DataFrame
        Tabela de partições.
    config : Config
        Configuração completa do projeto.

    Returns
    -------
    dict
        Resultado da validação cruzada por modelo.

    Examples
    --------
    >>> cross_validate_all(specs, conjunto, particoes, config)  # doctest: +SKIP
    """
    results: dict[str, dict[str, Any]] = {}

    for name, spec in specs.items():
        try:
            results[name] = cross_validate_model(name, spec, dataset, splits, config)
        except (ValueError, RuntimeError, MemoryError):
            # Um modelo que falha não pode interromper a comparação inteira:
            # o custo de reexecutar tudo é alto demais.
            logger.exception("Validação cruzada de '%s' falhou.", name)

    return results


def extract_fold_scores(results: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    """Extrai os scores por fold para os testes estatísticos pareados.

    Parameters
    ----------
    results : dict
        Saída de :func:`cross_validate_all`.

    Returns
    -------
    dict of str to np.ndarray
        Scores por fold de cada modelo, restrito aos modelos que completaram
        o mesmo número de folds — comparar séries de tamanhos diferentes
        invalidaria os testes pareados.

    Examples
    --------
    >>> extract_fold_scores(resultados)  # doctest: +SKIP
    """
    complete = {
        name: np.array(result["scores"]) for name, result in results.items() if result.get("scores")
    }
    if not complete:
        return {}

    expected = max(len(scores) for scores in complete.values())
    aligned = {name: scores for name, scores in complete.items() if len(scores) == expected}

    dropped = set(complete) - set(aligned)
    if dropped:
        logger.warning(
            "Modelos excluídos dos testes pareados por número diferente de folds: %s.",
            sorted(dropped),
        )

    return aligned
