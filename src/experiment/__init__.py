"""Rastreamento de experimentos e reprodutibilidade.

Modules
-------
tracker
    :class:`ExperimentTracker` — MLflow com registro de código, ambiente e
    hash dos dados; degrada silenciosamente quando o servidor está indisponível.
"""

from experiment.tracker import ExperimentTracker

__all__ = ["ExperimentTracker"]
