"""Validação da rotulação: revisão manual, concordância e descarte de indefinidos.

A supervisão fraca produz rótulos baratos, mas ruidosos. Sem uma estimativa
da qualidade desses rótulos, qualquer métrica reportada fica sem interpretação:
não haveria como distinguir erro do modelo de ruído do rótulo. Por isso uma
amostra estratificada é exportada para revisão manual e a concordância
(kappa de Cohen) entre supervisão fraca e revisão humana é reportada na
dissertação e no model card.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from sklearn.metrics import cohen_kappa_score

from config.logging import get_logger
from config.settings import UserLabelingSection
from constants.columns import MANUAL_LABEL, USER_ID, USER_LABEL
from constants.labels import UserLabel
from utils.validation import require_columns

logger = get_logger(__name__)


def sample_for_manual_review(
    labels: pl.DataFrame,
    sample_size: int,
    random_seed: int,
) -> pl.DataFrame:
    """Sorteia uma amostra estratificada por classe para revisão manual.

    Estratificada, e não aleatória simples: com classes desbalanceadas, uma
    amostra simples traria pouquíssimos casos de ``ideacao_suicida``, que é
    exatamente a classe cuja qualidade de rótulo mais importa verificar.

    Parameters
    ----------
    labels : pl.DataFrame
        Rótulos produzidos pela supervisão fraca.
    sample_size : int
        Tamanho total desejado da amostra.
    random_seed : int
        Semente do sorteio.

    Returns
    -------
    pl.DataFrame
        Amostra com ``user_id``, ``user_label`` e ``manual_label`` (vazia,
        para ser preenchida pelo revisor).

    Examples
    --------
    >>> sample_for_manual_review(rotulos, 200, 42)  # doctest: +SKIP
    """
    require_columns(labels, [USER_ID, USER_LABEL], context="amostra de revisão manual")

    classes = labels[USER_LABEL].unique().to_list()
    per_class = max(sample_size // max(len(classes), 1), 1)

    samples = [
        group.sample(n=min(per_class, group.height), seed=random_seed)
        for group in (labels.filter(pl.col(USER_LABEL) == label) for label in sorted(classes))
        if group.height > 0
    ]

    if not samples:
        logger.warning("Não há rótulos suficientes para montar a amostra de revisão manual.")
        return labels.head(0)

    result = (
        pl.concat(samples)
        .select([USER_ID, USER_LABEL])
        .with_columns(pl.lit(None, dtype=pl.Utf8).alias(MANUAL_LABEL))
    )

    logger.info("Amostra de revisão manual: %d usuários (%d por classe).", result.height, per_class)
    return result


def load_manual_labels(path: Path) -> pl.DataFrame | None:
    """Carrega os rótulos revisados manualmente, se existirem.

    Parameters
    ----------
    path : Path
        Arquivo CSV com ``user_id`` e ``manual_label``.

    Returns
    -------
    pl.DataFrame or None
        Rótulos manuais, ou ``None`` se o arquivo não existir.

    Examples
    --------
    >>> load_manual_labels(Path("data/external/manual_labels.csv"))  # doctest: +SKIP
    """
    target = Path(path)
    if not target.is_file():
        logger.info(
            "Nenhum arquivo de rótulos manuais em %s: usando apenas supervisão fraca.", target
        )
        return None

    frame = pl.read_csv(target)
    require_columns(frame, [USER_ID, MANUAL_LABEL], context="rótulos manuais")
    logger.info("Carregados %d rótulos revisados manualmente.", frame.height)
    return frame.select([USER_ID, MANUAL_LABEL])


def compute_agreement(labels: pl.DataFrame) -> dict[str, float]:
    """Calcula a concordância entre supervisão fraca e revisão manual.

    O kappa de Cohen é preferido à acurácia simples porque desconta a
    concordância esperada por acaso — com três classes desbalanceadas, uma
    acurácia de 70% pode não significar nada.

    Parameters
    ----------
    labels : pl.DataFrame
        Rótulos com ``user_label`` e ``manual_label``.

    Returns
    -------
    dict of str to float
        ``n_revisados``, ``concordancia_simples`` e ``kappa_cohen``. Retorna
        zeros quando não há revisão manual disponível.

    Examples
    --------
    >>> compute_agreement(rotulos)  # doctest: +SKIP
    {'n_revisados': 200, 'concordancia_simples': 0.82, 'kappa_cohen': 0.71}
    """
    if MANUAL_LABEL not in labels.columns:
        return {"n_revisados": 0.0, "concordancia_simples": 0.0, "kappa_cohen": 0.0}

    reviewed = labels.filter(pl.col(MANUAL_LABEL).is_not_null())
    if reviewed.is_empty():
        logger.info("Nenhum rótulo revisado manualmente: concordância não calculada.")
        return {"n_revisados": 0.0, "concordancia_simples": 0.0, "kappa_cohen": 0.0}

    weak = reviewed[USER_LABEL].to_list()
    manual = reviewed[MANUAL_LABEL].to_list()

    simple = sum(a == b for a, b in zip(weak, manual, strict=True)) / len(weak)
    kappa = float(cohen_kappa_score(manual, weak))

    logger.info(
        "Concordância rótulo fraco vs. manual: simples=%.3f, kappa=%.3f (n=%d).",
        simple,
        kappa,
        reviewed.height,
    )
    return {
        "n_revisados": float(reviewed.height),
        "concordancia_simples": simple,
        "kappa_cohen": kappa,
    }


def apply_manual_labels(labels: pl.DataFrame, manual: pl.DataFrame | None) -> pl.DataFrame:
    """Sobrepõe os rótulos manuais aos produzidos pela supervisão fraca.

    A revisão humana sempre vence: é a fonte de maior qualidade disponível.

    Parameters
    ----------
    labels : pl.DataFrame
        Rótulos da supervisão fraca.
    manual : pl.DataFrame or None
        Rótulos revisados manualmente.

    Returns
    -------
    pl.DataFrame
        Rótulos consolidados.

    Examples
    --------
    >>> apply_manual_labels(rotulos, manuais)  # doctest: +SKIP
    """
    if manual is None or manual.is_empty():
        return labels

    merged = labels.drop(MANUAL_LABEL, strict=False).join(manual, on=USER_ID, how="left")

    overridden = merged.filter(
        pl.col(MANUAL_LABEL).is_not_null() & (pl.col(MANUAL_LABEL) != pl.col(USER_LABEL))
    ).height

    result = merged.with_columns(
        pl.when(pl.col(MANUAL_LABEL).is_not_null())
        .then(pl.col(MANUAL_LABEL))
        .otherwise(pl.col(USER_LABEL))
        .alias(USER_LABEL)
    )

    logger.info("Rótulos manuais aplicados: %d divergências corrigidas.", overridden)
    return result


def drop_undecided(labels: pl.DataFrame, config: UserLabelingSection) -> pl.DataFrame:
    """Descarta os usuários sem consenso suficiente entre as fontes.

    Parameters
    ----------
    labels : pl.DataFrame
        Rótulos consolidados.
    config : UserLabelingSection
        Configuração da rotulação.

    Returns
    -------
    pl.DataFrame
        Rótulos sem a classe ``indefinido`` (ou inalterados, se o descarte
        estiver desativado).

    Examples
    --------
    >>> drop_undecided(rotulos, config.labeling.user_labeling)  # doctest: +SKIP
    """
    if not config.consensus.drop_undecided:
        return labels

    initial = labels.height
    result = labels.filter(pl.col(USER_LABEL) != str(UserLabel.INDEFINIDO))
    dropped = initial - result.height

    if dropped:
        logger.warning(
            "%d de %d usuários (%.1f%%) descartados por consenso abaixo de %.2f.",
            dropped,
            initial,
            100 * dropped / max(initial, 1),
            config.consensus.min_agreement,
        )
    return result
