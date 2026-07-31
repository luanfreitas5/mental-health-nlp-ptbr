"""Exceções relacionadas a configuração e variáveis de ambiente.

Configuração inválida deve falhar no startup, com erro tipado e mensagem
acionável — nunca no meio de uma coleta ou de um treinamento longo.
"""

from __future__ import annotations

from exceptions.base import MentalHealthNLPError


class ConfigurationError(MentalHealthNLPError):
    """Erro genérico de configuração."""


class ConfigFileNotFoundError(ConfigurationError):
    """Arquivo YAML de configuração inexistente."""


class ConfigParsingError(ConfigurationError):
    """YAML sintaticamente inválido ou fora do schema esperado."""


class ConfigValidationError(ConfigurationError):
    """Valores de configuração que violam as regras de negócio do projeto."""


class MissingSecretError(ConfigurationError):
    """Segredo obrigatório ausente no ``.env``.

    Examples
    --------
    >>> raise MissingSecretError(
    ...     "Variável PSEUDONYMIZATION_SALT ausente: copie .env.example para .env."
    ... )
    Traceback (most recent call last):
    ...
    exceptions.configuration.MissingSecretError: Variável PSEUDONYMIZATION_SALT ausente: ...
    """
