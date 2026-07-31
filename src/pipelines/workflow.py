"""Registro e orquestração das etapas do pipeline.

O registro é a única fonte de verdade sobre quais etapas existem e em que
ordem: ``main.py``, a ajuda da linha de comando e a execução completa leem
todos daqui. Acrescentar uma etapa é registrá-la neste módulo — nenhum outro
arquivo precisa mudar.
"""

from __future__ import annotations

from typing import Any

from config.logging import get_logger
from constants.defaults import PIPELINE_STAGES
from exceptions.pipeline import StageExecutionError, UnknownStageError
from pipelines.base import PipelineStage, StageContext
from pipelines.collection import CollectionStage
from pipelines.embedding import EmbeddingStage
from pipelines.evaluation import EvaluationStage
from pipelines.features import FeaturesStage
from pipelines.labeling import LabelingStage
from pipelines.preprocessing import PreprocessingStage
from pipelines.psychological import PsychologicalStage
from pipelines.reporting import ReportingStage
from pipelines.splitting import SplittingStage
from pipelines.training import TrainingStage
from utils.timing import log_duration

logger = get_logger(__name__)

#: Etapas disponíveis, na ordem de dependência.
STAGES: dict[str, type[PipelineStage]] = {
    "collect": CollectionStage,
    "preprocess": PreprocessingStage,
    "label": LabelingStage,
    "psych": PsychologicalStage,
    "embed": EmbeddingStage,
    "features": FeaturesStage,
    "split": SplittingStage,
    "train": TrainingStage,
    "evaluate": EvaluationStage,
    "report": ReportingStage,
}

#: Etapas executadas por ``--stage all``. A coleta fica de fora de propósito:
#: leva dias, exige aprovação ética e não deve ser disparada por engano numa
#: execução completa.
DEFAULT_PIPELINE: tuple[str, ...] = (
    "preprocess",
    "label",
    "psych",
    "embed",
    "features",
    "split",
    "train",
    "evaluate",
    "report",
)


def get_stage(name: str) -> PipelineStage:
    """Instancia uma etapa pelo nome.

    Parameters
    ----------
    name : str
        Nome da etapa (ex.: ``"train"``).

    Returns
    -------
    PipelineStage
        Etapa pronta para executar.

    Raises
    ------
    UnknownStageError
        Se a etapa não estiver registrada.

    Examples
    --------
    >>> get_stage("split").name
    'split'
    """
    if name not in STAGES:
        raise UnknownStageError(
            f"Etapa desconhecida: '{name}'. Disponíveis: {', '.join(STAGES)}, all."
        )
    return STAGES[name]()


def describe_stages() -> str:
    """Descreve todas as etapas registradas, para a ajuda da linha de comando.

    Returns
    -------
    str
        Uma linha por etapa, na ordem de dependência.

    Examples
    --------
    >>> "preprocess" in describe_stages()
    True
    """
    lines = [f"  {'all':<12} Executa o pipeline completo (exceto a coleta)"]
    lines.extend(
        f"  {name:<12} {STAGES[name].description}" for name in PIPELINE_STAGES if name in STAGES
    )
    return "\n".join(lines)


def run_stage(name: str, context: StageContext) -> dict[str, Any]:
    """Executa uma única etapa, com verificação de dependências.

    Parameters
    ----------
    name : str
        Nome da etapa.
    context : StageContext
        Contexto compartilhado.

    Returns
    -------
    dict
        Resumo produzido pela etapa.

    Raises
    ------
    UnknownStageError
        Se a etapa não existir.
    StageDependencyError
        Se um artefato exigido estiver ausente.
    StageExecutionError
        Se a etapa falhar durante a execução.

    Examples
    --------
    >>> run_stage("split", contexto)  # doctest: +SKIP
    """
    stage = get_stage(name)
    stage.check_dependencies(context)

    logger.info("=" * 72)
    logger.info("Etapa '%s' — %s", stage.name, stage.description)
    logger.info("=" * 72)

    try:
        with log_duration(f"Etapa '{stage.name}'"):
            result = stage.run(context)
    except (
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
        MemoryError,
    ) as error:
        # Erros do domínio (MentalHealthNLPError) sobem intactos: já trazem
        # mensagem acionável. Os demais ganham o contexto da etapa.
        raise StageExecutionError(
            f"A etapa '{stage.name}' falhou: {error}",
            context={"stage": stage.name},
        ) from error

    if result.get("skipped"):
        logger.warning("Etapa '%s' pulada: %s", stage.name, result.get("reason"))
    else:
        logger.info("Etapa '%s' concluída.", stage.name)

    return result


def run_pipeline(
    stages: list[str] | None,
    context: StageContext,
    *,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    """Executa uma sequência de etapas.

    Parameters
    ----------
    stages : list of str or None
        Etapas a executar; ``None`` usa :data:`DEFAULT_PIPELINE`.
    context : StageContext
        Contexto compartilhado.
    continue_on_error : bool, optional
        Prossegue para a próxima etapa em caso de falha, by default False.
        Útil numa execução longa e não supervisionada, onde interromper tudo
        por causa de uma etapa opcional custa caro.

    Returns
    -------
    dict
        Resumo de cada etapa executada.

    Raises
    ------
    StageExecutionError
        Na primeira falha, quando ``continue_on_error`` é ``False``.

    Examples
    --------
    >>> run_pipeline(["split", "train"], contexto)  # doctest: +SKIP
    """
    selected = list(stages) if stages else list(DEFAULT_PIPELINE)
    logger.info("Pipeline: %s", " -> ".join(selected))

    results: dict[str, Any] = {}
    for name in selected:
        try:
            results[name] = run_stage(name, context)
        except Exception as error:
            if not continue_on_error:
                raise
            logger.exception("Etapa '%s' falhou e será ignorada.", name)
            results[name] = {"failed": True, "error": str(error)}

    logger.info("Pipeline concluído: %d etapa(s) executada(s).", len(results))
    return results
