"""Figuras temporais: evolução do sentimento e ritmo de atividade.

São as figuras que tornam visível a justificativa da abordagem centrada no
usuário: nenhuma delas pode ser produzida a partir de tweets isolados.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from config.logging import get_logger
from constants.columns import CREATED_AT, SENTIMENT_POLARITY, USER_ID, USER_LABEL
from constants.labels import CLASS_ORDER
from visualization.theme import FIGURE_SIZES, get_class_labels, get_class_palette

logger = get_logger(__name__)


def plot_sentiment_evolution(
    tweets: pl.DataFrame,
    labels: pl.DataFrame,
    freq: str = "1w",
) -> Any:
    """Plota a evolução média do sentimento por classe ao longo do tempo.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets rotulados, com ``created_at`` e ``sentiment_polarity``.
    labels : pl.DataFrame
        Rótulos por usuário.
    freq : str, optional
        Granularidade da agregação temporal, by default ``"1w"`` (semanal).

    Returns
    -------
    matplotlib.figure.Figure
        Séries temporais com banda de erro-padrão.

    Examples
    --------
    >>> plot_sentiment_evolution(tweets, rotulos)  # doctest: +SKIP
    """
    aggregated = (
        tweets.join(labels.select([USER_ID, USER_LABEL]), on=USER_ID, how="inner")
        .with_columns(pl.col(CREATED_AT).dt.truncate(freq).alias("_periodo"))
        .group_by(["_periodo", USER_LABEL])
        .agg(
            pl.col(SENTIMENT_POLARITY).mean().alias("polaridade"),
            pl.col(SENTIMENT_POLARITY).std().alias("desvio"),
            pl.len().alias("n"),
        )
        .sort("_periodo")
    )

    figure, axis = plt.subplots(figsize=FIGURE_SIZES["wide"])
    classes = [name for name in CLASS_ORDER if name in aggregated[USER_LABEL].unique().to_list()]
    colors = get_class_palette(classes)
    display = get_class_labels(classes)

    for index, class_name in enumerate(classes):
        subset = aggregated.filter(pl.col(USER_LABEL) == class_name).sort("_periodo")
        periods = subset["_periodo"].to_list()
        values = subset["polaridade"].to_numpy()
        counts = subset["n"].to_numpy()
        deviations = np.nan_to_num(subset["desvio"].to_numpy())

        axis.plot(periods, values, color=colors[index], linewidth=2, label=display[index])

        # Banda = erro-padrão da média, não desvio-padrão: a pergunta é sobre a
        # precisão da média estimada em cada período, e ela depende do n.
        standard_error = deviations / np.sqrt(np.maximum(counts, 1))
        axis.fill_between(
            periods,
            values - standard_error,
            values + standard_error,
            color=colors[index],
            alpha=0.18,
        )

    axis.axhline(0, color="gray", linestyle="--", linewidth=1)
    axis.set_title("Evolução Temporal do Sentimento por Classe")
    axis.set_xlabel("Período")
    axis.set_ylabel("Polaridade média (-1 = negativo, +1 = positivo)")
    axis.legend(title="Classe")
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def plot_circadian_activity(
    tweets: pl.DataFrame,
    labels: pl.DataFrame,
    insomnia_window: tuple[int, int] = (0, 5),
) -> Any:
    """Plota a distribuição horária das publicações por classe.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets com ``created_at``.
    labels : pl.DataFrame
        Rótulos por usuário.
    insomnia_window : tuple of int, optional
        Faixa horária destacada como madrugada, by default ``(0, 5)``.

    Returns
    -------
    matplotlib.figure.Figure
        Distribuição relativa por hora, com a janela de insônia sombreada.

    Examples
    --------
    >>> plot_circadian_activity(tweets, rotulos)  # doctest: +SKIP
    """
    hourly = (
        tweets.join(labels.select([USER_ID, USER_LABEL]), on=USER_ID, how="inner")
        .with_columns(pl.col(CREATED_AT).dt.hour().alias("_hora"))
        .group_by([USER_LABEL, "_hora"])
        .agg(pl.len().alias("n"))
        .with_columns((pl.col("n") / pl.col("n").sum().over(USER_LABEL)).alias("proporcao"))
        .sort("_hora")
    )

    figure, axis = plt.subplots(figsize=FIGURE_SIZES["wide"])
    classes = [name for name in CLASS_ORDER if name in hourly[USER_LABEL].unique().to_list()]
    colors = get_class_palette(classes)
    display = get_class_labels(classes)

    for index, class_name in enumerate(classes):
        subset = hourly.filter(pl.col(USER_LABEL) == class_name).sort("_hora")
        axis.plot(
            subset["_hora"].to_list(),
            subset["proporcao"].to_list(),
            marker="o",
            markersize=4,
            color=colors[index],
            linewidth=2,
            label=display[index],
        )

    start, end = insomnia_window
    axis.axvspan(start, end, color="#C44E52", alpha=0.10, label="Janela de insônia")

    axis.set_xlim(0, 23)
    axis.set_xticks(range(0, 24, 2))
    axis.set_title("Distribuição Circadiana das Publicações por Classe")
    axis.set_xlabel("Hora do dia (horário local)")
    axis.set_ylabel("Proporção das publicações")
    axis.legend(title="Classe")
    figure.tight_layout()
    return figure


def plot_user_timeline(tweets: pl.DataFrame, user_id: str) -> Any:
    """Plota a linha do tempo de sentimento de um único usuário.

    Útil na análise de erro: entender por que o modelo errou num caso
    específico costuma exigir olhar a trajetória, não os agregados.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets rotulados.
    user_id : str
        Identificador pseudonimizado do usuário.

    Returns
    -------
    matplotlib.figure.Figure
        Série temporal individual com média móvel.

    Raises
    ------
    ValueError
        Se o usuário não tiver tweets no conjunto.

    Examples
    --------
    >>> plot_user_timeline(tweets, "u_ab12cd34")  # doctest: +SKIP
    """
    subset = tweets.filter(pl.col(USER_ID) == user_id).sort(CREATED_AT)
    if subset.is_empty():
        raise ValueError("Nenhum tweet encontrado para o usuário informado.")

    figure, axis = plt.subplots(figsize=FIGURE_SIZES["wide"])
    dates = subset[CREATED_AT].to_list()
    polarity = subset[SENTIMENT_POLARITY].to_numpy()

    axis.scatter(dates, polarity, alpha=0.4, s=18, color="#4C72B0", label="Tweets")

    window = max(3, len(polarity) // 20)
    if len(polarity) >= window:
        smoothed = np.convolve(polarity, np.ones(window) / window, mode="valid")
        axis.plot(
            dates[window - 1 :],
            smoothed,
            color="#C44E52",
            linewidth=2,
            label=f"Média móvel ({window} tweets)",
        )

    axis.axhline(0, color="gray", linestyle="--", linewidth=1)
    # O identificador é pseudonimizado, então pode aparecer no título sem
    # expor a identidade do usuário.
    axis.set_title(f"Trajetória de Sentimento — usuário {user_id}")
    axis.set_xlabel("Data")
    axis.set_ylabel("Polaridade")
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


def plot_activity_heatmap(tweets: pl.DataFrame) -> Any:
    """Plota um mapa de calor de atividade por dia da semana e hora.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets com ``created_at``.

    Returns
    -------
    matplotlib.figure.Figure
        Mapa de calor ``dia da semana × hora``.

    Examples
    --------
    >>> plot_activity_heatmap(tweets)  # doctest: +SKIP
    """
    frame = tweets.with_columns(
        pl.col(CREATED_AT).dt.weekday().alias("_dia"),
        pl.col(CREATED_AT).dt.hour().alias("_hora"),
    )

    matrix = np.zeros((7, 24))
    for row in frame.group_by(["_dia", "_hora"]).agg(pl.len().alias("n")).iter_rows(named=True):
        # polars: weekday devolve 1 (segunda) a 7 (domingo).
        matrix[row["_dia"] - 1, row["_hora"]] = row["n"]

    figure, axis = plt.subplots(figsize=FIGURE_SIZES["wide"])
    sns.heatmap(
        matrix,
        cmap="rocket_r",
        xticklabels=[str(hour) for hour in range(24)],
        yticklabels=["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"],
        cbar_kws={"label": "Número de tweets"},
        ax=axis,
    )

    axis.set_title("Atividade por Dia da Semana e Hora")
    axis.set_xlabel("Hora do dia")
    axis.set_ylabel("Dia da semana")
    figure.tight_layout()
    return figure
