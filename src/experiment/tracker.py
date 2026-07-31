"""Rastreamento de experimentos com MLflow.

Cada execução registra a tríade que torna um resultado reproduzível:
**código** (SHA do git), **ambiente** (versões das bibliotecas) e **dados**
(hash do dataset). Sem os três, um número numa tabela não pode ser
reproduzido — nem por terceiros, nem pelo próprio autor seis meses depois.

O rastreamento é opcional e falha de forma silenciosa: um servidor MLflow
indisponível não deve derrubar um treinamento de horas. As falhas são
registradas em log como aviso, e o pipeline segue.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from config.environment import describe_environment
from config.logging import get_logger
from config.settings import ExperimentSection
from config.version import describe_version

logger = get_logger(__name__)


def _import_mlflow() -> Any | None:
    """Importa o MLflow sob demanda, devolvendo ``None`` se ausente."""
    try:
        import mlflow
    except ImportError:
        logger.warning(
            "MLflow não está instalado: o rastreamento de experimentos será pulado. "
            "Rode 'uv sync --dev' para habilitá-lo."
        )
        return None
    return mlflow


class ExperimentTracker:
    """Registra parâmetros, métricas e artefatos de cada execução.

    Parameters
    ----------
    config : ExperimentSection
        Seção ``experiment`` de ``configs/config.yaml``.
    root : Path, optional
        Raiz usada para resolver um ``tracking_uri`` relativo.

    Attributes
    ----------
    enabled : bool
        ``False`` quando desativado na configuração ou quando o MLflow não
        está disponível.

    Examples
    --------
    >>> tracker = ExperimentTracker(config.general.experiment)
    >>> with tracker.run("xgboost"):  # doctest: +SKIP
    ...     tracker.log_metrics({"f1_macro": 0.76})
    """

    def __init__(self, config: ExperimentSection, root: Path | None = None) -> None:
        self.config = config
        self._mlflow = _import_mlflow() if config.enabled else None
        self.enabled = self._mlflow is not None

        if not self.enabled:
            return

        uri = config.tracking_uri
        if root is not None and not uri.startswith(("http://", "https://", "file:")):
            uri = (Path(root) / uri).as_uri()

        assert self._mlflow is not None  # garantido por `self.enabled` acima
        try:
            self._mlflow.set_tracking_uri(uri)
            self._mlflow.set_experiment(config.experiment_name)
            logger.info("MLflow ativo | experimento='%s' | uri=%s", config.experiment_name, uri)
        except Exception as error:
            logger.warning("MLflow indisponível (%s): rastreamento desativado.", error)
            self.enabled = False

    @contextmanager
    def run(self, run_name: str, tags: dict[str, str] | None = None) -> Iterator[None]:
        """Abre uma execução do MLflow.

        Parameters
        ----------
        run_name : str
            Nome da execução (normalmente o nome do modelo).
        tags : dict, optional
            Tags adicionais.

        Yields
        ------
        None
            Contexto da execução.

        Examples
        --------
        >>> with tracker.run("hybrid_xgboost"):  # doctest: +SKIP
        ...     pass
        """
        if not self.enabled:
            yield
            return

        version = describe_version()
        all_tags = {
            "git_sha": version["git_sha"],
            "project_version": version["version"],
            **(tags or {}),
        }

        assert self._mlflow is not None  # garantido por `self.enabled` acima
        try:
            with self._mlflow.start_run(run_name=run_name, tags=all_tags):
                self.log_params({f"env_{k}": v for k, v in describe_environment().items()})
                yield
        except Exception as error:
            logger.warning("Falha ao registrar a execução '%s' no MLflow: %s", run_name, error)
            yield

    def log_params(self, params: dict[str, Any]) -> None:
        """Registra parâmetros da execução.

        Parameters
        ----------
        params : dict
            Parâmetros a registrar (convertidos para string pelo MLflow).

        Examples
        --------
        >>> tracker.log_params({"max_depth": 6})  # doctest: +SKIP
        """
        if not self.enabled:
            return
        assert self._mlflow is not None  # garantido por `self.enabled` acima
        try:
            # O MLflow trunca valores longos; encurtamos antes para evitar
            # que um parâmetro grande derrube o registro inteiro.
            self._mlflow.log_params({key: str(value)[:500] for key, value in params.items()})
        except Exception as error:
            logger.debug("Falha ao registrar parâmetros no MLflow: %s", error)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Registra métricas da execução.

        Parameters
        ----------
        metrics : dict of str to float
            Métricas a registrar; valores não numéricos são ignorados.
        step : int, optional
            Passo (época ou fold) ao qual as métricas se referem.

        Examples
        --------
        >>> tracker.log_metrics({"f1_macro": 0.76})  # doctest: +SKIP
        """
        if not self.enabled:
            return
        assert self._mlflow is not None  # garantido por `self.enabled` acima
        try:
            numeric = {
                key: value for key, value in metrics.items() if isinstance(value, (int, float))
            }
            self._mlflow.log_metrics(numeric, step=step)
        except Exception as error:
            logger.debug("Falha ao registrar métricas no MLflow: %s", error)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        """Registra um arquivo como artefato da execução.

        Parameters
        ----------
        path : Path
            Arquivo a anexar (relatório, figura, model card).
        artifact_path : str, optional
            Subdiretório do artefato dentro da execução.

        Examples
        --------
        >>> tracker.log_artifact(Path("reports/evaluation_report.md"))  # doctest: +SKIP
        """
        if not self.enabled or not self.config.log_artifacts:
            return
        target = Path(path)
        if not target.exists():
            return
        assert self._mlflow is not None  # garantido por `self.enabled` acima
        try:
            self._mlflow.log_artifact(str(target), artifact_path=artifact_path)
        except Exception as error:
            logger.debug("Falha ao registrar artefato no MLflow: %s", error)

    def log_dataset(self, dataset_hash: str, n_users: int, n_features: int) -> None:
        """Registra a identidade do dataset usado na execução.

        Parameters
        ----------
        dataset_hash : str
            Hash do conteúdo da matriz de atributos.
        n_users : int
            Número de usuários.
        n_features : int
            Número de atributos.

        Examples
        --------
        >>> tracker.log_dataset("a1b2c3", 900, 512)  # doctest: +SKIP
        """
        self.log_params(
            {
                "dataset_hash": dataset_hash,
                "dataset_n_users": n_users,
                "dataset_n_features": n_features,
            }
        )

    def register_model(self, model_path: Path, model_name: str) -> None:
        """Registra o modelo no MLflow Model Registry.

        O Registry existe para separar "um arquivo no disco" de "o modelo
        promovido": sem ele, não há registro de qual versão está em uso nem
        como voltar à anterior.

        Parameters
        ----------
        model_path : Path
            Caminho do modelo persistido.
        model_name : str
            Nome sob o qual registrar.

        Examples
        --------
        >>> tracker.register_model(Path("models/artifacts/hybrid.joblib"), "clf")  # doctest: +SKIP
        """
        if not self.enabled or not self.config.register_best_model:
            return

        assert self._mlflow is not None  # garantido por `self.enabled` acima
        try:
            self._mlflow.log_artifact(str(model_path), artifact_path="model")
            logger.info(
                "Modelo '%s' anexado à execução. Para promovê-lo no Registry, use a UI "
                "do MLflow ou 'mlflow models register'.",
                model_name,
            )
        except Exception as error:
            logger.warning("Falha ao registrar o modelo no MLflow: %s", error)
