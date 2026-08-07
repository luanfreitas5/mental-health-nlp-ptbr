"""Ablation Study sobre os grupos de atributos.

É o experimento que testa H2, H3 e H4 diretamente: remover um grupo por vez
do modelo híbrido e medir o quanto a métrica principal cai.

Dois modos, propositalmente complementares:

* **leave-one-out** — mede a contribuição *marginal*: o quanto o grupo
  acrescenta além do que os outros já capturam.
* **only-one** — mede a contribuição *absoluta*: o quanto o grupo sozinho
  consegue.

Rodar apenas o primeiro leva a conclusões erradas com frequência. Grupos
correlacionados (por exemplo, emocional e psicológico) têm contribuição
marginal quase nula porque um substitui o outro, e um leave-one-out isolado
sugeriria que ambos são dispensáveis — quando, na verdade, remover os dois
juntos derrubaria o desempenho.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from rich.progress import Progress, TaskID

from config.logging import get_logger
from config.settings import AblationSection, Config
from constants.columns import USER_ID, USER_LABEL
from evaluation.metrics import compute_metrics
from models.base import UserDataset
from models.factory import create_model
from schemas.features import list_feature_columns
from utils.progress import build_progress

logger = get_logger(__name__)


@dataclass(frozen=True)
class AblationResult:
    """Resultado de uma configuração do Ablation Study.

    Attributes
    ----------
    configuration : str
        Descrição da configuração (ex.: ``"sem_temporal"``).
    groups : list of str
        Grupos de atributos utilizados.
    n_features : int
        Número de atributos na configuração.
    score_mean : float
        Média da métrica principal entre as repetições.
    score_std : float
        Desvio-padrão entre as repetições.
    delta : float
        Diferença em relação à configuração completa (negativo = piora).
    """

    configuration: str
    groups: list[str]
    n_features: int
    score_mean: float
    score_std: float
    delta: float


def build_dataset_for_groups(
    features: pl.DataFrame,
    groups: list[str],
    label_to_index: dict[str, int],
) -> UserDataset:
    """Monta um :class:`UserDataset` restrito a determinados grupos.

    Parameters
    ----------
    features : pl.DataFrame
        Matriz completa de atributos, com rótulo.
    groups : list of str
        Grupos a incluir.
    label_to_index : dict of str to int
        Mapeamento de rótulo para índice inteiro.

    Returns
    -------
    UserDataset
        Conjunto pronto para treino ou avaliação.

    Examples
    --------
    >>> build_dataset_for_groups(matriz, ["emotional"], mapa)  # doctest: +SKIP
    """
    columns = list_feature_columns(features, groups)
    labels = (
        np.array([label_to_index[value] for value in features[USER_LABEL].to_list()])
        if USER_LABEL in features.columns
        else None
    )

    return UserDataset(
        user_ids=features[USER_ID].to_list(),
        features=features.select(columns).to_numpy().astype(np.float64),
        feature_names=columns,
        labels=labels,
    )


def _resolve_available_groups(
    train_features: pl.DataFrame,
    ablation_config: AblationSection,
) -> list[str]:
    """Filtra os grupos configurados aos que possuem colunas na matriz."""
    available = [
        group for group in ablation_config.groups if list_feature_columns(train_features, [group])
    ]

    missing = set(ablation_config.groups) - set(available)
    if missing:
        logger.warning("Grupos sem colunas na matriz, excluídos da ablação: %s.", sorted(missing))

    return available


def _run_ablation_repeats(
    train: UserDataset,
    test: UserDataset,
    name: str,
    spec: Any,
    config: Config,
    ablation_config: AblationSection,
    metric: str,
) -> list[float]:
    """Repete o treino/avaliação do modelo base para uma configuração de grupos."""
    scores: list[float] = []
    for repeat in range(ablation_config.n_repeats):
        # A semente varia entre repetições para capturar a variância do
        # próprio treinamento, e não apenas a dos dados.
        params = {**spec.params, "random_state": config.random_seed + repeat}
        model = create_model(
            f"{ablation_config.base_model}_{name}",
            spec.model_copy(update={"params": params}),
            config,
        )
        model.fit(train)
        predictions = model.predict(test)
        assert test.labels is not None
        scores.append(compute_metrics(test.labels, predictions)[metric])
    return scores


def _evaluate_group_configuration(
    groups: list[str],
    name: str,
    train_features: pl.DataFrame,
    test_features: pl.DataFrame,
    label_to_index: dict[str, int],
    spec: Any,
    config: Config,
    ablation_config: AblationSection,
    metric: str,
) -> AblationResult:
    """Treina e avalia o modelo base restrito a um conjunto de grupos."""
    train = build_dataset_for_groups(train_features, groups, label_to_index)
    test = build_dataset_for_groups(test_features, groups, label_to_index)
    scores = _run_ablation_repeats(train, test, name, spec, config, ablation_config, metric)

    return AblationResult(
        configuration=name,
        groups=groups,
        n_features=len(train.feature_names),
        score_mean=float(np.mean(scores)),
        score_std=float(np.std(scores)),
        delta=0.0,
    )


def _run_leave_one_out(
    available: list[str],
    evaluate: Callable[[list[str], str], AblationResult],
    full: AblationResult,
    progress: Progress,
    task: TaskID,
) -> dict[str, Any]:
    """Executa a etapa leave-one-out: remove um grupo por vez e mede a queda."""
    leave_one_out: dict[str, Any] = {}
    for group in available:
        remaining = [name for name in available if name != group]
        result = evaluate(remaining, f"sem_{group}")
        leave_one_out[group] = result.__dict__ | {
            "delta": result.score_mean - full.score_mean,
        }
        progress.advance(task)
    return leave_one_out


def _run_only_one(
    available: list[str],
    evaluate: Callable[[list[str], str], AblationResult],
    full: AblationResult,
    progress: Progress,
    task: TaskID,
) -> dict[str, Any]:
    """Executa a etapa only-one: avalia cada grupo isoladamente."""
    only_one: dict[str, Any] = {}
    for group in available:
        result = evaluate([group], f"apenas_{group}")
        only_one[group] = result.__dict__ | {
            "delta": result.score_mean - full.score_mean,
        }
        progress.advance(task)
    return only_one


def _rank_group_contributions(
    full: AblationResult,
    leave_one_out: dict[str, Any],
) -> dict[str, float]:
    """Ordena os grupos pela contribuição marginal (queda ao remover), decrescente."""
    # Contribuição marginal = queda causada pela remoção do grupo. Quanto
    # maior, mais o grupo acrescenta além do que os demais já capturam.
    contributions = {
        group: full.score_mean - entry["score_mean"] for group, entry in leave_one_out.items()
    }
    return dict(sorted(contributions.items(), key=operator.itemgetter(1), reverse=True))


def run_ablation(
    train_features: pl.DataFrame,
    test_features: pl.DataFrame,
    config: Config,
    ablation_config: AblationSection,
    label_to_index: dict[str, int],
) -> dict[str, Any]:
    """Executa o Ablation Study completo.

    Parameters
    ----------
    train_features, test_features : pl.DataFrame
        Matrizes de treino e teste, com rótulo.
    config : Config
        Configuração completa do projeto.
    ablation_config : AblationSection
        Seção ``ablation`` de ``configs/evaluation.yaml``.
    label_to_index : dict of str to int
        Mapeamento de rótulo para índice inteiro.

    Returns
    -------
    dict
        ``baseline`` (todos os grupos), ``leave_one_out``, ``only_one`` e um
        ranking dos grupos por contribuição marginal.

    Raises
    ------
    UnknownModelError
        Se ``ablation.base_model`` não estiver declarado no YAML.

    Examples
    --------
    >>> run_ablation(treino, teste, config, ablacao, mapa)  # doctest: +SKIP
    """
    if not ablation_config.enabled:
        logger.info("Ablation Study desativado em configs/evaluation.yaml.")
        return {}

    spec = config.models.all_models()[ablation_config.base_model]
    metric = config.evaluation.metrics.primary
    available = _resolve_available_groups(train_features, ablation_config)

    def evaluate(groups: list[str], name: str) -> AblationResult:
        """Treina e avalia o modelo base restrito a um conjunto de grupos."""
        return _evaluate_group_configuration(
            groups,
            name,
            train_features,
            test_features,
            label_to_index,
            spec,
            config,
            ablation_config,
            metric,
        )

    total_steps = 1 + len(available) + (len(available) if ablation_config.include_only_one else 0)
    results: dict[str, Any] = {}

    with build_progress() as progress:
        task = progress.add_task("Ablation Study", total=total_steps)

        full = evaluate(available, "completo")
        results["baseline"] = full.__dict__
        progress.advance(task)

        results["leave_one_out"] = _run_leave_one_out(available, evaluate, full, progress, task)

        if ablation_config.include_only_one:
            results["only_one"] = _run_only_one(available, evaluate, full, progress, task)

    results["ranking"] = _rank_group_contributions(full, results["leave_one_out"])
    results["metric"] = metric

    logger.info(
        "Ablação concluída. Métrica completa (%s): %.4f. Contribuição marginal: %s",
        metric,
        full.score_mean,
        ", ".join(f"{group}={value:+.4f}" for group, value in results["ranking"].items()),
    )
    return results


def summarize_ablation(results: dict[str, Any]) -> pl.DataFrame:
    """Resume o Ablation Study numa tabela pronta para relatório.

    Parameters
    ----------
    results : dict
        Saída de :func:`run_ablation`.

    Returns
    -------
    pl.DataFrame
        Colunas ``grupo``, ``contribuicao_marginal``, ``score_sem_grupo``,
        ``score_apenas_grupo`` e ``n_atributos``.

    Examples
    --------
    >>> summarize_ablation(resultados)  # doctest: +SKIP
    """
    if not results or "leave_one_out" not in results:
        return pl.DataFrame()

    only_one = results.get("only_one", {})
    records = [
        {
            "grupo": group,
            "contribuicao_marginal": results["ranking"].get(group, 0.0),
            "score_sem_grupo": entry["score_mean"],
            "score_apenas_grupo": only_one.get(group, {}).get("score_mean"),
            "n_atributos": entry["n_features"],
        }
        for group, entry in results["leave_one_out"].items()
    ]

    return pl.DataFrame(records).sort("contribuicao_marginal", descending=True)
