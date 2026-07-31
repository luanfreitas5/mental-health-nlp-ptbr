"""Aplicação dos contratos de dados nas fronteiras do pipeline.

Todo estágio valida a entrada que recebe e a saída que produz. Validar
apenas o código não basta: a corrupção mais cara deste projeto é silenciosa
— uma coluna de data que virou string, um score de sentimento fora de
``[0, 1]``, um ``user_id`` nulo — e se propaga por todas as etapas seguintes
até aparecer como uma métrica estranha, horas depois.
"""

from __future__ import annotations

from typing import TypeVar

import pandera.polars as pa
import polars as pl
from pandera.errors import SchemaError, SchemaErrors

from config.logging import get_logger
from exceptions.data import SchemaValidationError

logger = get_logger(__name__)

SchemaType = TypeVar("SchemaType", bound=pa.DataFrameModel)


def validate_frame(
    frame: pl.DataFrame,
    schema: type[pa.DataFrameModel],
    *,
    context: str = "",
    lazy: bool = True,
) -> pl.DataFrame:
    """Valida um DataFrame contra um contrato pandera.

    Parameters
    ----------
    frame : pl.DataFrame
        DataFrame a validar.
    schema : type of pandera.DataFrameModel
        Contrato a aplicar.
    context : str, optional
        Descrição da fronteira validada (ex.: ``"saída de preprocess"``),
        incluída na mensagem de erro.
    lazy : bool, optional
        Coleta **todas** as violações antes de falhar, by default True.
        Falhar na primeira violação obrigaria a corrigir e reexecutar uma
        etapa longa uma vez por erro.

    Returns
    -------
    pl.DataFrame
        O próprio DataFrame, validado.

    Raises
    ------
    SchemaValidationError
        Se o DataFrame violar o contrato.

    Examples
    --------
    >>> from schemas.tweets import RawTweetSchema
    >>> validate_frame(frame, RawTweetSchema, context="coleta")  # doctest: +SKIP
    """
    label = context or schema.__name__
    try:
        validated = schema.validate(frame, lazy=lazy)
    except (SchemaError, SchemaErrors) as error:
        logger.exception("Contrato de dados violado em '%s'.", label)
        raise SchemaValidationError(
            f"Violação do contrato '{schema.__name__}' em {label}. Detalhes:\n{error}"
        ) from error

    logger.debug(
        "Contrato '%s' validado em '%s' (%d linhas, %d colunas).",
        schema.__name__,
        label,
        frame.height,
        frame.width,
    )
    return pl.DataFrame(validated)


def is_valid(frame: pl.DataFrame, schema: type[pa.DataFrameModel]) -> bool:
    """Informa se um DataFrame satisfaz um contrato, sem levantar exceção.

    Útil em ramos condicionais e em testes; o caminho normal do pipeline deve
    usar :func:`validate_frame`, que falha alto.

    Parameters
    ----------
    frame : pl.DataFrame
        DataFrame a verificar.
    schema : type of pandera.DataFrameModel
        Contrato a aplicar.

    Returns
    -------
    bool
        ``True`` se o DataFrame for válido.

    Examples
    --------
    >>> is_valid(frame, RawTweetSchema)  # doctest: +SKIP
    True
    """
    try:
        schema.validate(frame, lazy=True)
    except (SchemaError, SchemaErrors):
        return False
    return True
