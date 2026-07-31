"""Exceção base do projeto.

Toda exceção customizada herda de :class:`MentalHealthNLPError`, o que permite
capturar qualquer erro previsto do domínio com um único ``except`` no
orquestrador (``main.py``) sem recorrer a ``except Exception``.
"""

from __future__ import annotations

from typing import Any


class MentalHealthNLPError(Exception):
    """Erro base de todas as exceções do projeto.

    Parameters
    ----------
    message : str
        Mensagem descritiva em pt-BR, explicando a causa e, quando possível,
        a ação corretiva.
    context : dict, optional
        Metadados adicionais (etapa, arquivo, modelo) anexados à mensagem.
        **Nunca** inclua texto bruto de tweet ou identificadores diretos aqui:
        exceções acabam em logs (ver LGPD em docs/guides/ethics.md).

    Examples
    --------
    >>> raise MentalHealthNLPError("Falha na etapa", context={"stage": "train"})
    Traceback (most recent call last):
    ...
    exceptions.base.MentalHealthNLPError: Falha na etapa (stage=train)
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        self.message = message
        self.context = context or {}
        super().__init__(self._format())

    def _format(self) -> str:
        """Compõe a mensagem final concatenando o contexto, se houver."""
        if not self.context:
            return self.message
        details = ", ".join(f"{key}={value}" for key, value in self.context.items())
        return f"{self.message} ({details})"
