"""Importância de atributos por permutação e pelos coeficientes do modelo.

A importância nativa das árvores (ganho de impureza) é enviesada a favor de
variáveis com muitos valores distintos — o que, nesta matriz, favorece
sistematicamente as dimensões contínuas de embedding sobre as razões lexicais.
A importância por permutação mede diretamente a queda de desempenho ao
embaralhar a coluna, é calculada no conjunto de teste e não sofre desse viés.

As duas são reportadas lado a lado: divergência entre elas é informação, não
inconsistência — costuma indicar exatamente onde o viés da métrica nativa está
atuando.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from sklearn.inspection import permutation_importance

from config.logging import get_logger
from config.settings import PermutationImportanceSection
from constants.columns import FEATURE_GROUP_PREFIXES
from models.base import BaseUserClassifier, UserDataset

logger = get_logger(__name__)


def compute_permutation_importance(
    model: BaseUserClassifier,
    dataset: UserDataset,
    config: PermutationImportanceSection,
) -> pl.DataFrame:
    """Calcula a importância dos atributos por permutação.

    Parameters
    ----------
    model : BaseUserClassifier
        Modelo treinado, com atributo ``pipeline_`` compatível com scikit-learn.
    dataset : UserDataset
        Conjunto de teste, com rótulos.
    config : PermutationImportanceSection
        Seção ``interpretability.permutation_importance``.

    Returns
    -------
    pl.DataFrame
        Colunas ``atributo``, ``importancia``, ``desvio`` e ``grupo``,
        ordenadas por importância decrescente. Vazio se o modelo não expuser
        um estimador scikit-learn.

    Examples
    --------
    >>> compute_permutation_importance(modelo, teste, config)  # doctest: +SKIP
    """
    estimator = getattr(model, "pipeline_", None)
    if estimator is None:
        logger.warning(
            "O modelo '%s' não expõe um pipeline scikit-learn: importância por permutação "
            "não calculada.",
            model.name,
        )
        return pl.DataFrame()

    if not dataset.has_labels:
        logger.warning("Importância por permutação exige rótulos no conjunto de teste.")
        return pl.DataFrame()

    result = permutation_importance(
        estimator,
        dataset.features,
        dataset.labels,
        n_repeats=config.n_repeats,
        random_state=config.random_state,
        scoring=config.scoring,
        n_jobs=1,
    )

    # Com `scoring` de string única (nosso caso), `permutation_importance`
    # devolve um único `Bunch`; os stubs generalizam para o overload que
    # aceita múltiplos scorers e infere `dict[str, Bunch]`.
    frame = pl.DataFrame(
        {
            "atributo": dataset.feature_names,
            "importancia": result.importances_mean.tolist(),  # pyright: ignore[reportAttributeAccessIssue]
            "desvio": result.importances_std.tolist(),  # pyright: ignore[reportAttributeAccessIssue]
        }
    ).with_columns(
        pl.col("atributo").map_elements(resolve_feature_group, return_dtype=pl.Utf8).alias("grupo")
    )

    logger.info(
        "Importância por permutação calculada para %d atributos (%s).",
        frame.height,
        config.scoring,
    )
    return frame.sort("importancia", descending=True)


def resolve_feature_group(feature_name: str) -> str:
    """Identifica a qual grupo um atributo pertence, pelo prefixo.

    Parameters
    ----------
    feature_name : str
        Nome da coluna.

    Returns
    -------
    str
        Nome do grupo, ou ``"outro"`` se o prefixo não for reconhecido.

    Examples
    --------
    >>> resolve_feature_group("psy_tristeza_mean")
    'psychological'
    >>> resolve_feature_group("n_tweets")
    'outro'
    """
    for group, prefix in FEATURE_GROUP_PREFIXES.items():
        if feature_name.startswith(prefix):
            return group
    return "outro"


def aggregate_importance_by_group(importance: pl.DataFrame) -> pl.DataFrame:
    """Soma a importância dos atributos dentro de cada grupo.

    Complementa o Ablation Study por um caminho independente: a ablação mede
    o impacto de *remover* o grupo, esta agregação mede o quanto o modelo já
    treinado *usa* o grupo. Convergência entre as duas reforça a conclusão.

    Parameters
    ----------
    importance : pl.DataFrame
        Saída de :func:`compute_permutation_importance`.

    Returns
    -------
    pl.DataFrame
        Importância total e média por grupo, com participação percentual.

    Examples
    --------
    >>> aggregate_importance_by_group(importancia)  # doctest: +SKIP
    """
    if importance.is_empty():
        return pl.DataFrame()

    grouped = (
        importance.group_by("grupo")
        .agg(
            pl.col("importancia").sum().alias("importancia_total"),
            pl.col("importancia").mean().alias("importancia_media"),
            pl.len().alias("n_atributos"),
        )
        .sort("importancia_total", descending=True)
    )

    total = float(grouped["importancia_total"].sum())
    if total <= 0:
        return grouped

    return grouped.with_columns(
        (100 * pl.col("importancia_total") / total).alias("participacao_pct")
    )


def extract_model_importance(model: BaseUserClassifier) -> pl.DataFrame:
    """Extrai a importância nativa do estimador, quando disponível.

    Parameters
    ----------
    model : BaseUserClassifier
        Modelo treinado.

    Returns
    -------
    pl.DataFrame
        Colunas ``atributo``, ``importancia_nativa`` e ``grupo``; vazio se o
        estimador não expuser importâncias.

    Examples
    --------
    >>> extract_model_importance(modelo)  # doctest: +SKIP
    """
    extractor = getattr(model, "feature_importances", None)
    if extractor is None:
        return pl.DataFrame()

    values: dict[str, float] | None = extractor()
    if not values:
        return pl.DataFrame()

    return (
        pl.DataFrame({"atributo": list(values), "importancia_nativa": list(values.values())})
        .with_columns(
            pl.col("atributo")
            .map_elements(resolve_feature_group, return_dtype=pl.Utf8)
            .alias("grupo")
        )
        .sort("importancia_nativa", descending=True)
    )


def top_features(importance: pl.DataFrame, n: int = 25) -> list[dict[str, Any]]:
    """Seleciona os atributos mais importantes.

    Parameters
    ----------
    importance : pl.DataFrame
        Tabela de importância.
    n : int, optional
        Quantidade de atributos, by default 25.

    Returns
    -------
    list of dict
        Registros dos ``n`` atributos mais importantes.

    Examples
    --------
    >>> top_features(importancia, n=5)  # doctest: +SKIP
    """
    if importance.is_empty():
        return []
    return importance.head(n).to_dicts()
