"""Contrato da matriz final de atributos por usuário.

Diferente dos demais contratos, este não pode enumerar suas colunas: o número
de features varia com a configuração (dimensão do PCA, tamanho do vocabulário
de n-grams, grupos ativos). A validação é então **estrutural**: verifica as
chaves obrigatórias, a convenção de prefixos por grupo e a ausência de
valores não finitos — que é onde os erros reais aparecem.
"""

from __future__ import annotations

import numpy as np
import pandera.polars as pa
import polars as pl
from pandera.typing.polars import Series

from constants.columns import FEATURE_GROUP_PREFIXES, NON_FEATURE_COLUMNS, USER_ID
from exceptions.data import SchemaValidationError
from schemas.tweets import PSEUDONYM_REGEX
from schemas.users import VALID_LABELS


class UserFeatureKeySchema(pa.DataFrameModel):
    """Contrato das colunas-chave da matriz de atributos.

    ``strict = False``: as colunas de features variam com a configuração e são
    verificadas por :func:`validate_feature_matrix`.
    """

    user_id: Series[str] = pa.Field(nullable=False, unique=True, str_matches=PSEUDONYM_REGEX)
    user_label: Series[str] = pa.Field(nullable=False, isin=VALID_LABELS)
    n_tweets: Series[int] = pa.Field(gt=0, nullable=False)

    # Padrão documentado do pandera: a classe `Config` aninhada não herda de
    # `BaseConfig`, o que o type checker interpreta como um override
    # incompatível — falso positivo conhecido dos stubs do pandera.
    class Config:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Configuração do contrato."""

        strict = False
        coerce = True


def list_feature_columns(frame: pl.DataFrame, groups: list[str] | None = None) -> list[str]:
    """Lista as colunas de atributos de um DataFrame.

    A seleção é feita pelo **prefixo** da coluna (``ling_``, ``emo_``, ...),
    o que permite ao Ablation Study incluir ou remover um grupo inteiro sem
    manter uma lista explícita de centenas de nomes.

    Parameters
    ----------
    frame : pl.DataFrame
        Matriz de atributos.
    groups : list of str, optional
        Grupos desejados; ``None`` retorna todos os grupos conhecidos.

    Returns
    -------
    list of str
        Colunas de atributos, em ordem determinística.

    Raises
    ------
    KeyError
        Se algum grupo solicitado não existir em
        :data:`constants.columns.FEATURE_GROUP_PREFIXES`.

    Examples
    --------
    >>> frame = pl.DataFrame({"user_id": ["u_1"], "ling_len": [1.0], "emo_neg": [0.5]})
    >>> list_feature_columns(frame, ["linguistic"])
    ['ling_len']
    """
    selected = _resolve_selected_groups(groups)
    _validate_known_groups(selected)
    prefixes = tuple(FEATURE_GROUP_PREFIXES[group] for group in selected)
    return _select_columns_by_prefix(frame, prefixes)


def _resolve_selected_groups(groups: list[str] | None) -> list[str]:
    """Resolve os grupos solicitados, usando todos os grupos conhecidos por padrão."""
    return groups if groups is not None else list(FEATURE_GROUP_PREFIXES)


def _validate_known_groups(selected: list[str]) -> None:
    """Garante que todos os grupos solicitados são conhecidos em ``FEATURE_GROUP_PREFIXES``."""
    unknown = [group for group in selected if group not in FEATURE_GROUP_PREFIXES]
    if unknown:
        raise KeyError(
            f"Grupos de atributos desconhecidos: {unknown}. "
            f"Conhecidos: {sorted(FEATURE_GROUP_PREFIXES)}"
        )


def _select_columns_by_prefix(frame: pl.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    """Seleciona, em ordem determinística, as colunas com um dos prefixos informados."""
    return sorted(
        column
        for column in frame.columns
        if column.startswith(prefixes) and column not in NON_FEATURE_COLUMNS
    )


def validate_feature_matrix(
    frame: pl.DataFrame,
    *,
    expected_groups: list[str] | None = None,
    allow_nan: bool = False,
) -> pl.DataFrame:
    """Valida estruturalmente a matriz de atributos por usuário.

    Parameters
    ----------
    frame : pl.DataFrame
        Matriz a validar.
    expected_groups : list of str, optional
        Grupos que devem obrigatoriamente ter ao menos uma coluna.
    allow_nan : bool, optional
        Permite valores ausentes, by default False. Um ``NaN`` que chega ao
        XGBoost é tratado como categoria própria e ao scikit-learn levanta
        exceção — em ambos os casos, é melhor detectar aqui.

    Returns
    -------
    pl.DataFrame
        A própria matriz, validada.

    Raises
    ------
    SchemaValidationError
        Se faltar coluna-chave, se um grupo esperado estiver vazio ou se
        houver valores não finitos com ``allow_nan=False``.

    Examples
    --------
    >>> frame = pl.DataFrame(
    ...     {"user_id": ["u_ab"], "user_label": ["controle"], "n_tweets": [30], "ling_a": [1.0]}
    ... )
    >>> validate_feature_matrix(frame, expected_groups=["linguistic"]).height
    1
    """
    _validate_user_id_present(frame)
    _validate_expected_groups_present(frame, expected_groups)

    feature_columns = list_feature_columns(frame)
    _validate_has_feature_columns(feature_columns)

    if not allow_nan:
        _validate_finite_feature_values(frame, feature_columns)

    return frame


def _validate_user_id_present(frame: pl.DataFrame) -> None:
    """Garante que a coluna de identificação do usuário existe na matriz."""
    if USER_ID not in frame.columns:
        raise SchemaValidationError(
            f"Matriz de atributos sem a coluna '{USER_ID}': impossível associar "
            "features aos rótulos."
        )


def _validate_expected_groups_present(
    frame: pl.DataFrame, expected_groups: list[str] | None
) -> None:
    """Garante que cada grupo de atributos esperado contribui com ao menos uma coluna."""
    for group in expected_groups or []:
        if not list_feature_columns(frame, [group]):
            raise SchemaValidationError(
                f"Nenhuma coluna do grupo '{group}' foi encontrada na matriz. "
                f"Verifique se features.groups.{group} está ativo em configs/features.yaml."
            )


def _validate_has_feature_columns(feature_columns: list[str]) -> None:
    """Garante que a matriz contém ao menos uma coluna de atributos."""
    if not feature_columns:
        raise SchemaValidationError(
            "A matriz não contém nenhuma coluna de atributos. "
            f"Prefixos esperados: {sorted(set(FEATURE_GROUP_PREFIXES.values()))}"
        )


def _find_non_finite_columns(frame: pl.DataFrame, feature_columns: list[str]) -> list[str]:
    """Lista as colunas de atributos com valores ausentes ou não finitos."""
    return [
        column
        for column in feature_columns
        if frame[column].null_count() > 0
        or (
            frame[column].dtype.is_numeric()
            and not bool(np.all(np.isfinite(frame[column].fill_null(0.0).to_numpy())))
        )
    ]


def _validate_finite_feature_values(frame: pl.DataFrame, feature_columns: list[str]) -> None:
    """Garante que não há valores ausentes ou não finitos nas colunas de atributos."""
    offenders = _find_non_finite_columns(frame, feature_columns)
    if offenders:
        raise SchemaValidationError(
            f"{len(offenders)} coluna(s) com valores ausentes ou não finitos: "
            f"{offenders[:10]}{'...' if len(offenders) > 10 else ''}. "
            "Ajuste features.aggregation.missing_strategy em configs/features.yaml."
        )
