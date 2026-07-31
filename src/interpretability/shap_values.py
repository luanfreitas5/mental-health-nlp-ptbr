"""Explicações SHAP do modelo final.

Interpretabilidade não é opcional neste domínio: um sistema de triagem de
saúde mental cuja decisão não pode ser explicada não é utilizável na prática,
e a análise dos fatores associados aos sinais de risco é uma das contribuições
científicas declaradas na proposta.

Duas ressalvas registradas no relatório e no model card:

* **SHAP explica o modelo, não o fenômeno.** Um valor alto para
  ``psy_risco_suicida_mean`` significa que o modelo se apoia nessa feature —
  não que ela seja causa de risco.
* **Componentes de PCA não são interpretáveis diretamente.** No modelo
  híbrido, as dimensões semânticas passam por PCA; elas aparecem como
  ``sem_pca_*`` justamente para impedir a leitura equivocada de que
  correspondem a termos específicos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config.logging import get_logger
from config.settings import ShapSection
from interpretability.importance import resolve_feature_group
from models.base import BaseUserClassifier, UserDataset

logger = get_logger(__name__)


def _import_shap() -> Any:
    """Importa o ``shap`` sob demanda, devolvendo ``None`` se ausente."""
    try:
        import shap
    except ImportError:
        logger.warning(
            "O pacote 'shap' não está instalado: a análise de interpretabilidade será "
            "pulada. Rode 'uv sync --dev' para habilitá-la."
        )
        return None
    return shap


def compute_shap_values(
    model: BaseUserClassifier,
    dataset: UserDataset,
    config: ShapSection,
) -> dict[str, Any]:
    """Calcula os valores SHAP do modelo sobre o conjunto de teste.

    Parameters
    ----------
    model : BaseUserClassifier
        Modelo treinado.
    dataset : UserDataset
        Conjunto a explicar.
    config : ShapSection
        Seção ``interpretability.shap`` de ``configs/evaluation.yaml``.

    Returns
    -------
    dict
        ``values`` (array por classe), ``feature_names``, ``sample`` (matriz
        usada) e ``explainer``. Dicionário vazio se o SHAP não puder ser
        aplicado ao modelo.

    Examples
    --------
    >>> compute_shap_values(
    ...     modelo, teste, config.evaluation.interpretability.shap
    ... )  # doctest: +SKIP
    """
    if not config.enabled:
        return {}

    shap = _import_shap()
    if shap is None:
        return {}

    pipeline = getattr(model, "pipeline_", None)
    if pipeline is None:
        logger.warning(
            "O modelo '%s' não expõe um pipeline scikit-learn: SHAP não aplicável.", model.name
        )
        return {}

    rng = np.random.default_rng(config.random_state)
    n_samples = min(config.sample_size, dataset.features.shape[0])
    indices = rng.choice(dataset.features.shape[0], size=n_samples, replace=False)
    sample = dataset.features[indices]

    # O modelo híbrido transforma os blocos antes da cabeça; explicar a
    # entrada crua atribuiria valores a colunas que o estimador nunca vê.
    feature_names = dataset.feature_names
    transformer = pipeline[:-1]
    estimator = pipeline[-1]
    transformed = transformer.transform(sample) if len(pipeline) > 1 else sample

    renamer = getattr(model, "transformed_feature_names", None)
    if renamer is not None:
        feature_names = renamer()

    try:
        if config.explainer == "tree":
            explainer = shap.TreeExplainer(estimator)
        elif config.explainer == "linear":
            explainer = shap.LinearExplainer(estimator, transformed)
        else:
            explainer = shap.KernelExplainer(estimator.predict_proba, transformed[:100])

        values = explainer.shap_values(transformed)
    except (ValueError, TypeError, AttributeError) as error:
        logger.warning("Falha ao calcular valores SHAP para '%s': %s", model.name, error)
        return {}

    logger.info(
        "Valores SHAP calculados para '%s' sobre %d amostras (%s).",
        model.name,
        n_samples,
        config.explainer,
    )
    return {
        "values": values,
        "feature_names": list(feature_names),
        "sample": transformed,
        "explainer": config.explainer,
        "sample_indices": indices.tolist(),
    }


def summarize_shap(shap_result: dict[str, Any], max_display: int = 25) -> pl.DataFrame:
    """Resume os valores SHAP na importância média absoluta por atributo.

    Parameters
    ----------
    shap_result : dict
        Saída de :func:`compute_shap_values`.
    max_display : int, optional
        Número de atributos no resumo, by default 25.

    Returns
    -------
    pl.DataFrame
        Colunas ``atributo``, ``shap_medio_absoluto`` e ``grupo``, em ordem
        decrescente.

    Examples
    --------
    >>> summarize_shap(resultado_shap, max_display=10)  # doctest: +SKIP
    """
    if not shap_result or "values" not in shap_result:
        return pl.DataFrame()

    values = shap_result["values"]

    # A API do SHAP varia: multiclasse devolve lista de arrays (uma por
    # classe) ou um array 3D. Ambos são reduzidos à média absoluta por
    # atributo, agregando as classes.
    if isinstance(values, list):
        stacked = np.mean([np.abs(item).mean(axis=0) for item in values], axis=0)
    elif values.ndim == 3:
        stacked = np.abs(values).mean(axis=(0, 2))
    else:
        stacked = np.abs(values).mean(axis=0)

    names = shap_result["feature_names"][: len(stacked)]

    return (
        pl.DataFrame(
            {"atributo": names, "shap_medio_absoluto": np.asarray(stacked).ravel().tolist()}
        )
        .with_columns(
            pl.col("atributo")
            .map_elements(resolve_feature_group, return_dtype=pl.Utf8)
            .alias("grupo")
        )
        .sort("shap_medio_absoluto", descending=True)
        .head(max_display)
    )


def save_shap_summary(summary: pl.DataFrame, directory: Path, model_name: str) -> Path | None:
    """Grava o resumo SHAP em CSV.

    Parameters
    ----------
    summary : pl.DataFrame
        Saída de :func:`summarize_shap`.
    directory : Path
        Diretório de destino.
    model_name : str
        Nome do modelo, usado no nome do arquivo.

    Returns
    -------
    Path or None
        Caminho gravado, ou ``None`` se não havia o que gravar.

    Examples
    --------
    >>> save_shap_summary(resumo, Path("reports/interpretability"), "hybrid")  # doctest: +SKIP
    """
    if summary.is_empty():
        return None

    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"shap_summary_{model_name}.csv"
    summary.write_csv(target)

    logger.info("Resumo SHAP gravado em %s.", target)
    return target
