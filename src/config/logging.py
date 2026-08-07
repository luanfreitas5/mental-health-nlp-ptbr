"""Configuração de logging: console via ``rich`` e arquivo com rotação diária.

Duas decisões merecem destaque:

1. **Redação de PII no próprio handler.** O filtro :class:`PIIRedactionFilter`
   é anexado a todos os handlers, de modo que menções, URLs, e-mails e
   telefones são removidos *antes* de a mensagem ser escrita. Depender apenas
   da disciplina de quem chama ``logger.info`` não é uma garantia — e um log
   com PII de pessoas em sofrimento psíquico é exatamente o vazamento que o
   projeto não pode ter (ver docs/guides/ethics.md).
2. **Console e arquivo com níveis independentes.** O console fica em ``INFO``
   (legível durante uma execução longa) e o arquivo em ``DEBUG`` (rastro
   completo para depuração posterior).

Examples
--------
>>> from config.logging import configure_logging, get_logger
>>> configure_logging()
>>> logger = get_logger(__name__)
>>> logger.info("Pipeline iniciado.")
"""

from __future__ import annotations

import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.logging import RichHandler

from constants.regex import PII_PATTERNS
from exceptions.configuration import ConfigFileNotFoundError, ConfigParsingError

#: Console compartilhado — o mesmo objeto é usado pelo logging e pelas barras
#: de progresso, senão as duas saídas competem pelo terminal e se sobrescrevem.
CONSOLE: Console = Console(stderr=False)

_CONFIGURED: bool = False


class PIIRedactionFilter(logging.Filter):
    """Remove informação pessoal identificável das mensagens de log.

    Parameters
    ----------
    patterns : list of str
        Nomes dos padrões a aplicar (``mention``, ``url``, ``email``, ``phone``).
    replacement : str, optional
        Texto que substitui a ocorrência, by default ``"[REDIGIDO]"``.

    Examples
    --------
    >>> filtro = PIIRedactionFilter(["email"])
    >>> registro = logging.LogRecord("t", logging.INFO, "f", 1, "contato: a@b.com", None, None)
    >>> filtro.filter(registro)
    True
    >>> registro.msg
    'contato: [REDIGIDO]'
    """

    def __init__(self, patterns: list[str], replacement: str = "[REDIGIDO]") -> None:
        super().__init__()
        self.replacement = replacement
        self.patterns = [PII_PATTERNS[name] for name in patterns if name in PII_PATTERNS]

    def filter(self, record: logging.LogRecord) -> bool:
        """Reescreve a mensagem do registro sem a PII.

        Returns
        -------
        bool
            Sempre ``True`` — o filtro sanitiza, não descarta registros.
        """
        message = record.getMessage()
        redacted = message
        for pattern in self.patterns:
            redacted = pattern.sub(self.replacement, redacted)

        if redacted != message:
            # Substitui msg/args de uma vez: já interpolamos em getMessage().
            record.msg = redacted
            record.args = ()
        return True


def read_logging_config(config_file: Path) -> dict[str, Any]:
    """Lê ``configs/logging.yaml``.

    Parameters
    ----------
    config_file : Path
        Caminho do arquivo de configuração de logging.

    Returns
    -------
    dict
        Conteúdo do arquivo.

    Raises
    ------
    ConfigFileNotFoundError
        Se o arquivo não existir.
    ConfigParsingError
        Se o YAML for inválido.
    """
    if not config_file.is_file():
        raise ConfigFileNotFoundError(f"Configuração de logging não encontrada: {config_file}")
    try:
        return yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise ConfigParsingError(f"YAML inválido em {config_file}: {error}") from error


def _build_file_handler(
    settings: dict[str, Any],
    logs_dir: Path,
) -> TimedRotatingFileHandler:
    """Cria o handler de arquivo com rotação diária à meia-noite."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    pattern = settings.get("filename_pattern", "log_%Y-%m-%d.log")
    filename = logs_dir / datetime.now().strftime(pattern)

    handler = TimedRotatingFileHandler(
        filename=filename,
        when=settings.get("rotation", "midnight"),
        backupCount=int(settings.get("backup_count", 30)),
        encoding=settings.get("encoding", "utf-8"),
        delay=True,
    )
    handler.setLevel(settings.get("level", "DEBUG"))
    handler.setFormatter(
        logging.Formatter(
            fmt=settings.get("format", "%(asctime)s \t %(levelname)s \t %(name)s \t %(message)s"),
            datefmt=settings.get("datefmt", "%Y-%m-%d %H:%M:%S"),
        )
    )
    return handler


def _build_console_handler(settings: dict[str, Any]) -> RichHandler:
    """Cria o handler de console com ``rich`` (cor, tracebacks e ``name:lineno``)."""
    handler = RichHandler(
        console=CONSOLE,
        rich_tracebacks=bool(settings.get("rich_tracebacks", True)),
        show_path=bool(settings.get("show_path", True)),
        show_time=bool(settings.get("show_time", True)),
        markup=bool(settings.get("markup", True)),
        omit_repeated_times=False,
    )
    handler.setLevel(settings.get("level", "INFO"))
    # O RichHandler já renderiza tempo e nível; o formatter cuida de name:lineno.
    handler.setFormatter(logging.Formatter("%(name)s:%(lineno)d \t %(message)s"))
    return handler


def _configure_pii_filter(redaction: dict[str, Any]) -> PIIRedactionFilter | None:
    """Cria o filtro de redação de PII a partir da seção ``redaction`` do YAML."""
    if not redaction.get("enabled", True):
        return None
    return PIIRedactionFilter(
        patterns=list(redaction.get("patterns", [])),
        replacement=str(redaction.get("replacement", "[REDIGIDO]")),
    )


def _attach_console_handler(
    root: logging.Logger,
    console_settings: dict[str, Any],
    pii_filter: PIIRedactionFilter | None,
    level: str | None,
    root_level: str,
) -> None:
    """Constrói e anexa o handler de console ao logger raiz, se habilitado."""
    if not console_settings.get("enabled", True):
        return
    console_handler = _build_console_handler(console_settings)
    if level:
        console_handler.setLevel(root_level)
    if pii_filter:
        console_handler.addFilter(pii_filter)
    root.addHandler(console_handler)


def _attach_file_handler(
    root: logging.Logger,
    file_settings: dict[str, Any],
    pii_filter: PIIRedactionFilter | None,
) -> None:
    """Constrói e anexa o handler de arquivo ao logger raiz, se habilitado."""
    if not file_settings.get("enabled", True):
        return

    # Import tardio: evita ciclo (config.paths não depende de logging).
    from config.paths import get_paths

    logs_dir = Path(file_settings.get("directory", "logs"))
    if not logs_dir.is_absolute():
        logs_dir = get_paths().root / logs_dir
    file_handler = _build_file_handler(file_settings, logs_dir)
    if pii_filter:
        file_handler.addFilter(pii_filter)
    root.addHandler(file_handler)


def _quiet_third_party_loggers(console_settings: dict[str, Any]) -> None:
    """Reduz a verbosidade de bibliotecas de terceiros conforme configurado."""
    for name, quiet_level in console_settings.get("quiet_loggers", {}).items():
        logging.getLogger(name).setLevel(str(quiet_level).upper())


def _reset_root_handlers(root: logging.Logger) -> None:
    """Remove todos os handlers atualmente registrados no logger raiz."""
    for handler in root.handlers.copy():
        root.removeHandler(handler)


def _resolve_logging_settings(config_file: Path | None) -> dict[str, Any]:
    """Resolve o caminho do YAML de logging (padrão ou informado) e o carrega."""
    # Import tardio: evita ciclo (config.paths não depende de logging).
    from config.paths import get_paths

    target = config_file or (get_paths().configs.root / "logging.yaml")
    return read_logging_config(Path(target))


def configure_logging(
    config_file: Path | None = None,
    level: str | None = None,
    *,
    force: bool = False,
) -> logging.Logger:
    """Configura o logging global do projeto.

    Idempotente: chamadas repetidas não duplicam handlers (o que faria cada
    mensagem aparecer N vezes no console).

    Parameters
    ----------
    config_file : Path, optional
        Arquivo de configuração, by default ``configs/logging.yaml``.
    level : str, optional
        Sobrescreve o nível do YAML (usado por ``main.py --log-level``).
    force : bool, optional
        Reconfigura mesmo que já tenha sido configurado, by default False.

    Returns
    -------
    logging.Logger
        Logger raiz já configurado.

    Examples
    --------
    >>> configure_logging(level="DEBUG")  # doctest: +ELLIPSIS
    <RootLogger root (DEBUG)>
    """
    global _CONFIGURED  # noqa: PLW0603 — estado de configuração é global por natureza

    root = logging.getLogger()
    if _CONFIGURED and not force:
        return root

    settings = _resolve_logging_settings(config_file)

    root_level = (level or settings.get("level", "INFO")).upper()
    root.setLevel(root_level)
    _reset_root_handlers(root)

    pii_filter = _configure_pii_filter(settings.get("redaction", {}))
    console_settings = settings.get("console", {})
    _attach_console_handler(root, console_settings, pii_filter, level, root_level)
    _attach_file_handler(root, settings.get("file", {}), pii_filter)
    _quiet_third_party_loggers(console_settings)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger nomeado, garantindo que o logging esteja configurado.

    Parameters
    ----------
    name : str
        Nome do logger — use sempre ``__name__``.

    Returns
    -------
    logging.Logger
        Logger pronto para uso.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.name
    'config.logging'
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
