"""Orquestrador principal do projeto.

Ponto de entrada único de todas as etapas, selecionadas por ``--stage``.

Examples
--------
Pipeline completo (exceto a coleta, que exige aprovação ética)::

    python src/main.py --stage all

Uma etapa isolada::

    python src/main.py --stage features

Treinar apenas alguns modelos, incluindo a extensão exploratória::

    python src/main.py --stage train --models xgboost hybrid_xgboost --include-exploratory

Ver as etapas disponíveis::

    python src/main.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Torna `src/` a raiz de importação quando o script é executado diretamente
# (`python src/main.py`), sem exigir instalação do pacote.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.environment import log_environment, seed_everything
from config.logging import configure_logging, get_logger
from config.paths import get_paths
from config.settings import load_config
from config.version import build_run_id, get_version
from data.catalog import log_catalog
from exceptions.base import MentalHealthNLPError
from experiment.tracker import ExperimentTracker
from pipelines.base import StageContext
from pipelines.workflow import STAGES, describe_stages, run_pipeline
from utils.files import write_json

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Monta o analisador de argumentos da linha de comando.

    Returns
    -------
    argparse.ArgumentParser
        Analisador configurado.

    Examples
    --------
    >>> build_parser().prog
    'main.py'
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Detecção longitudinal de sinais de depressão e ideação suicida em "
            "redes sociais com Transformers e Modelos de Linguagem."
        ),
        epilog=f"Etapas disponíveis:\n{describe_stages()}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--stage",
        nargs="+",
        default=["all"],
        choices=[*STAGES, "all"],
        metavar="ETAPA",
        help="Etapa(s) a executar. 'all' roda o pipeline completo (exceto a coleta).",
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Diretório alternativo de configurações (padrão: configs/).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Sobrescreve o nível de log definido em configs/logging.yaml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sobrescreve a semente global (padrão: configs/config.yaml).",
    )

    # --- Seleção de modelos ------------------------------------------------
    models = parser.add_argument_group("seleção de modelos")
    models.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="NOME",
        help="Restringe a execução a modelos específicos de model_params.yaml.",
    )
    models.add_argument(
        "--include-exploratory",
        action="store_true",
        help="Inclui a extensão exploratória além da comparação principal.",
    )
    models.add_argument(
        "--skip-cv",
        action="store_true",
        help="Pula a validação cruzada e treina direto no conjunto de treino.",
    )
    models.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Pula o Ablation Study na etapa de avaliação.",
    )

    # --- Coleta -------------------------------------------------------------
    collection = parser.add_argument_group("coleta")
    collection.add_argument(
        "--dry-run",
        action="store_true",
        help="Constrói as consultas de coleta sem executar nenhuma requisição.",
    )
    collection.add_argument(
        "--allow-collection-without-ethics",
        action="store_true",
        help=(
            "Dispensa a barreira ética. APENAS para teste técnico com contas próprias; "
            "dados assim coletados não podem compor a base da pesquisa."
        ),
    )

    # --- Execução -----------------------------------------------------------
    execution = parser.add_argument_group("execução")
    execution.add_argument(
        "--all-encoders",
        action="store_true",
        help="Gera embeddings com todos os encoders declarados, não só o principal.",
    )
    execution.add_argument(
        "--limit-users",
        type=int,
        default=None,
        help="Limita o número de usuários processados (útil na extração por LLM).",
    )
    execution.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Prossegue para a próxima etapa mesmo se uma falhar.",
    )
    execution.add_argument(
        "--no-tracking",
        action="store_true",
        help="Desativa o rastreamento de experimentos no MLflow.",
    )
    execution.add_argument(
        "--status",
        action="store_true",
        help="Exibe a situação dos artefatos de dados e encerra.",
    )
    execution.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {get_version()}",
    )

    return parser


def build_context(arguments: argparse.Namespace) -> StageContext:
    """Monta o contexto compartilhado a partir dos argumentos.

    Parameters
    ----------
    arguments : argparse.Namespace
        Argumentos da linha de comando.

    Returns
    -------
    StageContext
        Contexto pronto para as etapas.

    Examples
    --------
    >>> build_context(build_parser().parse_args([]))  # doctest: +SKIP
    """
    config = load_config(arguments.config_dir)
    paths = get_paths()
    paths.ensure_directories()

    seed = arguments.seed if arguments.seed is not None else config.random_seed
    seed_everything(seed)

    tracker = None
    if not arguments.no_tracking:
        tracker = ExperimentTracker(config.general.experiment, root=paths.root)

    options = {
        "models": arguments.models,
        "include_exploratory": arguments.include_exploratory,
        "skip_cv": arguments.skip_cv,
        "skip_ablation": arguments.skip_ablation,
        "dry_run": arguments.dry_run,
        "allow_collection_without_ethics": arguments.allow_collection_without_ethics,
        "all_encoders": arguments.all_encoders,
        "limit_users": arguments.limit_users,
        "seed": seed,
    }

    return StageContext(config=config, paths=paths, tracker=tracker, options=options)


def resolve_stages(requested: list[str]) -> list[str] | None:
    """Converte a seleção da linha de comando na lista de etapas.

    Parameters
    ----------
    requested : list of str
        Etapas pedidas; ``["all"]`` significa o pipeline completo.

    Returns
    -------
    list of str or None
        Lista de etapas, ou ``None`` para usar o pipeline padrão.

    Examples
    --------
    >>> resolve_stages(["all"]) is None
    True
    >>> resolve_stages(["train", "evaluate"])
    ['train', 'evaluate']
    """
    return None if "all" in requested else requested.copy()


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada da linha de comando.

    Parameters
    ----------
    argv : list of str, optional
        Argumentos; ``None`` usa ``sys.argv``.

    Returns
    -------
    int
        Código de saída: 0 em sucesso, 1 em erro previsto do domínio,
        2 em erro inesperado, 130 se interrompido pelo usuário.

    Examples
    --------
    >>> main(["--status"])  # doctest: +SKIP
    0
    """
    arguments = build_parser().parse_args(argv)
    configure_logging(level=arguments.log_level)

    run_id = build_run_id("pipeline")
    logger.info("=" * 72)
    logger.info("%s v%s | execução %s", "mental-health-nlp-ptbr", get_version(), run_id)
    logger.info("=" * 72)

    try:
        context = build_context(arguments)

        if arguments.status:
            log_catalog(context.paths)
            return 0

        log_environment()

        results = run_pipeline(
            resolve_stages(arguments.stage),
            context,
            continue_on_error=arguments.continue_on_error,
        )

        summary_path = context.paths.reports.metrics / f"{run_id}_summary.json"
        write_json(summary_path, {"run_id": run_id, "stages": _serialize(results)})
        logger.info("Resumo da execução gravado em %s.", summary_path)

    except KeyboardInterrupt:
        logger.warning("Execução interrompida pelo usuário.")
        return 130
    except MentalHealthNLPError as error:
        # Erros previstos do domínio já trazem mensagem acionável; o traceback
        # completo só polui a saída.
        logger.error("%s", error)  # noqa: TRY400
        return 1
    except Exception:
        logger.exception("Falha inesperada na execução do pipeline.")
        return 2

    logger.info("Execução %s concluída com sucesso.", run_id)
    return 0


def _serialize(results: dict[str, Any]) -> dict[str, Any]:
    """Converte o resumo das etapas em estrutura serializável em JSON."""
    serializable: dict[str, Any] = {}
    for stage, payload in results.items():
        serializable[stage] = {
            key: value
            for key, value in payload.items()
            if value is None or isinstance(value, (str | int | float | bool | list | tuple | dict))
        }
    return serializable


if __name__ == "__main__":
    raise SystemExit(main())
