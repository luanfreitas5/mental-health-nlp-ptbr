"""Contrato comum das etapas do pipeline.

Cada etapa é uma unidade independente que lê artefatos do disco e grava
outros. O acoplamento entre elas é o sistema de arquivos, não a memória — o
que permite executar qualquer etapa isoladamente, retomar uma execução
interrompida e inspecionar o resultado intermediário de qualquer estágio.

Num projeto em que a coleta leva dias e o fine-tuning leva horas, poder
reexecutar apenas a etapa que mudou é o que torna a iteração viável.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.logging import get_logger
from config.paths import ProjectPaths
from config.settings import Config
from constants.defaults import STAGE_DEPENDENCIES
from exceptions.pipeline import StageDependencyError
from experiment.tracker import ExperimentTracker

logger = get_logger(__name__)


@dataclass
class StageContext:
    """Contexto compartilhado por todas as etapas.

    Attributes
    ----------
    config : Config
        Configuração completa e validada do projeto.
    paths : ProjectPaths
        Caminhos do projeto.
    tracker : ExperimentTracker, optional
        Rastreador de experimentos.
    options : dict
        Opções da linha de comando (``--models``, ``--include-exploratory``, ...).
    """

    config: Config
    paths: ProjectPaths
    tracker: ExperimentTracker | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def option(self, key: str, default: Any = None) -> Any:
        """Lê uma opção da linha de comando.

        Parameters
        ----------
        key : str
            Nome da opção.
        default : Any, optional
            Valor padrão quando ausente.

        Returns
        -------
        Any
            Valor da opção.

        Examples
        --------
        >>> contexto.option("include_exploratory", False)  # doctest: +SKIP
        False
        """
        return self.options.get(key, default)


class PipelineStage(ABC):
    """Etapa executável do pipeline.

    Attributes
    ----------
    name : str
        Identificador da etapa (usado em ``--stage``).
    description : str
        Descrição em pt-BR, exibida na ajuda da linha de comando.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, context: StageContext) -> dict[str, Any]:
        """Executa a etapa.

        Parameters
        ----------
        context : StageContext
            Contexto compartilhado.

        Returns
        -------
        dict
            Resumo do que foi produzido (contagens, caminhos gravados).
        """

    def required_inputs(self, context: StageContext) -> list[Path]:
        """Lista os artefatos que a etapa exige em disco.

        Parameters
        ----------
        context : StageContext
            Contexto compartilhado.

        Returns
        -------
        list of Path
            Caminhos obrigatórios (lista vazia por padrão).
        """
        del context
        return []

    def check_dependencies(self, context: StageContext) -> None:
        """Valida se os artefatos exigidos existem.

        Falhar aqui, antes de qualquer processamento, evita descobrir a
        dependência faltante depois de vinte minutos de execução.

        Parameters
        ----------
        context : StageContext
            Contexto compartilhado.

        Raises
        ------
        StageDependencyError
            Se algum artefato obrigatório estiver ausente.

        Examples
        --------
        >>> etapa.check_dependencies(contexto)  # doctest: +SKIP
        """
        missing = [path for path in self.required_inputs(context) if not path.exists()]
        if not missing:
            return

        dependencies = STAGE_DEPENDENCIES.get(self.name, ())
        hint = f" Execute antes: {', '.join(dependencies)}." if dependencies else ""
        raise StageDependencyError(
            f"A etapa '{self.name}' exige artefatos que não existem: "
            f"{[str(path.name) for path in missing]}.{hint}",
            context={"stage": self.name},
        )
