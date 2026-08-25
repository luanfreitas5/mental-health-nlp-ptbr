"""Figuras exploratórias: distribuição das classes, palavras e n-grams."""

from __future__ import annotations

from collections import Counter
from typing import Any, cast

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from config.logging import get_logger
from constants.columns import TEXT_CLEAN, USER_LABEL
from constants.labels import CLASS_DISPLAY_NAMES, CLASS_ORDER
from visualization.theme import (
    CATEGORICAL_PALETTE,
    FIGURE_SIZES,
    get_class_labels,
    get_class_palette,
)

logger = get_logger(__name__)


def _present_classes(merged: pl.DataFrame, label_column: str = USER_LABEL) -> list[str]:
    """Retorna as classes de ``CLASS_ORDER`` presentes no DataFrame combinado."""
    present = set(merged[label_column].unique().to_list())
    return [name for name in CLASS_ORDER if name in present]


def _annotate_bar_percentages(axis: Any, total: int) -> None:
    """Anota cada barra do gráfico com a contagem e o percentual sobre o total."""
    for container in axis.containers:
        # `axis.containers` é tipado como `list[Container]` genérico, mas o
        # `barplot` do seaborn sempre popula `BarContainer`.
        axis.bar_label(
            container,  # pyright: ignore[reportArgumentType]
            labels=[
                f"{int(bar.get_height())}\n({100 * bar.get_height() / total:.1f}%)".replace(
                    ".", ","
                )
                for bar in container
            ],
            padding=3,
            fontsize=9,
        )


def plot_class_distribution(labels: pl.DataFrame, column: str = USER_LABEL) -> Any:
    """Plota a distribuição de usuários por classe.

    O desbalanceamento é reportado no próprio gráfico: é o contexto que
    determina se um F1 de 0,75 é bom ou apenas reflete a classe majoritária.

    Parameters
    ----------
    labels : pl.DataFrame
        Rótulos por usuário.
    column : str, optional
        Coluna do rótulo, by default ``user_label``.

    Returns
    -------
    matplotlib.figure.Figure
        Figura com as contagens e os percentuais anotados.

    Examples
    --------
    >>> plot_class_distribution(rotulos)  # doctest: +SKIP
    """
    counts = (
        labels.group_by(column)
        .len()
        .sort(column)
        .with_columns(pl.col(column).replace(CLASS_DISPLAY_NAMES).alias("_display"))
    )

    figure, axis = plt.subplots(figsize=FIGURE_SIZES["medium"])
    order = [CLASS_DISPLAY_NAMES.get(name, name) for name in CLASS_ORDER]
    present = [name for name in order if name in counts["_display"].to_list()]

    sns.barplot(
        data=counts.to_pandas(),
        x="_display",
        y="len",
        hue="_display",
        order=present,
        hue_order=present,
        palette=get_class_palette(),
        legend=False,
        ax=axis,
    )

    total = int(counts["len"].sum())
    _annotate_bar_percentages(axis, total)

    # `Series.max()`/`.min()` são tipados de forma genérica nos stubs do
    # Polars; a coluna `len()` é sempre inteira.
    ratio = cast(int, counts["len"].max()) / max(cast(int, counts["len"].min()), 1)
    axis.set_title(
        f"Distribuição de Usuários por Classe (n = {total}; desbalanceamento {ratio:.1f}x)"
    )
    axis.set_xlabel("Classe")
    axis.set_ylabel("Número de usuários")
    figure.tight_layout()
    return figure


def _plot_word_frequency_panel(axis: Any, texts: list[str], class_name: str, top_n: int) -> None:
    """Plota o painel de palavras mais frequentes de uma classe."""
    common = Counter(token for text in texts for token in text.split()).most_common(top_n)
    if not common:
        axis.set_visible(False)
        return

    words, frequencies = zip(*common, strict=True)
    sns.barplot(
        x=list(frequencies),
        y=list(words),
        color=get_class_palette([class_name])[0],
        ax=axis,
    )
    axis.set_title(CLASS_DISPLAY_NAMES.get(class_name, class_name))
    axis.set_xlabel("Frequência")
    axis.set_ylabel("")


def plot_word_frequency(
    tweets: pl.DataFrame,
    labels: pl.DataFrame,
    top_n: int = 20,
) -> Any:
    """Plota as palavras mais frequentes por classe.

    Um painel por classe, e não um gráfico único: o que interessa é o
    contraste de vocabulário entre os grupos, e barras empilhadas esconderiam
    justamente isso.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos, com ``user_id`` e ``text_clean``.
    labels : pl.DataFrame
        Rótulos por usuário.
    top_n : int, optional
        Palavras por classe, by default 20.

    Returns
    -------
    matplotlib.figure.Figure
        Figura com um painel por classe.

    Examples
    --------
    >>> plot_word_frequency(tweets, rotulos)  # doctest: +SKIP
    """
    merged = tweets.join(labels.select(["user_id", USER_LABEL]), on="user_id", how="inner")
    classes = _present_classes(merged)

    figure, axes = plt.subplots(1, len(classes), figsize=(6 * len(classes), 6), squeeze=False)

    for index, class_name in enumerate(classes):
        texts = merged.filter(pl.col(USER_LABEL) == class_name)[TEXT_CLEAN].to_list()
        _plot_word_frequency_panel(axes[0][index], texts, class_name, top_n)

    figure.suptitle(f"{top_n} Palavras Mais Frequentes por Classe", fontweight="bold")
    figure.tight_layout()
    return figure


def _plot_ngram_panel(axis: Any, texts: list[str], class_name: str, n: int, top_n: int) -> None:
    """Plota o painel de n-grams mais frequentes de uma classe."""
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = text.split()
        counter.update(
            " ".join(tokens[position : position + n]) for position in range(len(tokens) - n + 1)
        )

    common = counter.most_common(top_n)
    if not common:
        axis.set_visible(False)
        return

    grams, frequencies = zip(*common, strict=True)
    sns.barplot(
        x=list(frequencies),
        y=list(grams),
        color=get_class_palette([class_name])[0],
        ax=axis,
    )
    axis.set_title(CLASS_DISPLAY_NAMES.get(class_name, class_name))
    axis.set_xlabel("Frequência")
    axis.set_ylabel("")


def plot_ngrams(
    tweets: pl.DataFrame,
    labels: pl.DataFrame,
    n: int = 2,
    top_n: int = 15,
) -> Any:
    """Plota os n-grams mais frequentes por classe.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos.
    labels : pl.DataFrame
        Rótulos por usuário.
    n : int, optional
        Tamanho do n-gram, by default 2 (bigramas).
    top_n : int, optional
        N-grams por classe, by default 15.

    Returns
    -------
    matplotlib.figure.Figure
        Figura com um painel por classe.

    Examples
    --------
    >>> plot_ngrams(tweets, rotulos, n=2)  # doctest: +SKIP
    """
    merged = tweets.join(labels.select(["user_id", USER_LABEL]), on="user_id", how="inner")
    classes = _present_classes(merged)

    figure, axes = plt.subplots(1, len(classes), figsize=(7 * len(classes), 6), squeeze=False)

    for index, class_name in enumerate(classes):
        texts = merged.filter(pl.col(USER_LABEL) == class_name)[TEXT_CLEAN].to_list()
        _plot_ngram_panel(axes[0][index], texts, class_name, n, top_n)

    label = {1: "Unigramas", 2: "Bigramas", 3: "Trigramas"}.get(n, f"{n}-gramas")
    figure.suptitle(f"{top_n} {label} Mais Frequentes por Classe", fontweight="bold")
    figure.tight_layout()
    return figure


def _plot_wordcloud_panel(
    axis: Any, texts: list[str], class_name: str, wordcloud_cls: type
) -> None:
    """Plota a nuvem de palavras de uma classe, se houver corpus não vazio."""
    axis.axis("off")
    corpus = " ".join(text for text in texts).strip()
    if not corpus:
        return

    cloud = wordcloud_cls(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        random_state=42,
    ).generate(corpus)

    axis.imshow(cloud, interpolation="bilinear")
    axis.set_title(CLASS_DISPLAY_NAMES.get(class_name, class_name))


def plot_wordcloud(tweets: pl.DataFrame, labels: pl.DataFrame) -> Any | None:
    """Gera nuvens de palavras por classe.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos.
    labels : pl.DataFrame
        Rótulos por usuário.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figura com uma nuvem por classe, ou ``None`` se o pacote
        ``wordcloud`` não estiver instalado.

    Notes
    -----
    A nuvem é ilustrativa: o tamanho da fonte não é uma medida estatística e
    não deve fundamentar conclusões. As frequências e os n-grams cumprem esse
    papel.

    Examples
    --------
    >>> plot_wordcloud(tweets, rotulos)  # doctest: +SKIP
    """
    try:
        from wordcloud import WordCloud  # pyright: ignore[reportMissingImports]
    except ImportError:
        logger.warning("Pacote 'wordcloud' ausente: nuvens de palavras não geradas.")
        return None

    merged = tweets.join(labels.select(["user_id", USER_LABEL]), on="user_id", how="inner")
    classes = _present_classes(merged)

    figure, axes = plt.subplots(1, len(classes), figsize=(7 * len(classes), 5), squeeze=False)

    for index, class_name in enumerate(classes):
        texts = merged.filter(pl.col(USER_LABEL) == class_name)[TEXT_CLEAN].to_list()
        _plot_wordcloud_panel(axes[0][index], texts, class_name, WordCloud)

    figure.suptitle("Nuvens de Palavras por Classe", fontweight="bold")
    figure.tight_layout()
    return figure


def plot_feature_distribution(
    features: pl.DataFrame,
    column: str,
    label_column: str = USER_LABEL,
) -> Any:
    """Compara a distribuição de um atributo entre as classes.

    Parameters
    ----------
    features : pl.DataFrame
        Matriz de atributos com o rótulo.
    column : str
        Atributo a visualizar.
    label_column : str, optional
        Coluna do rótulo, by default ``user_label``.

    Returns
    -------
    matplotlib.figure.Figure
        Figura com violino e boxplot sobrepostos.

    Raises
    ------
    KeyError
        Se o atributo não existir na matriz.

    Examples
    --------
    >>> plot_feature_distribution(matriz, "temp_night_activity_ratio")  # doctest: +SKIP
    """
    if column not in features.columns:
        raise KeyError(f"Atributo '{column}' ausente na matriz.")

    data = features.select([column, label_column]).to_pandas()
    order = [name for name in CLASS_ORDER if name in data[label_column].unique()]

    figure, axis = plt.subplots(figsize=FIGURE_SIZES["medium"])
    sns.violinplot(
        data=data,
        x=label_column,
        y=column,
        order=order,
        hue=label_column,
        hue_order=order,
        palette=get_class_palette(order),
        legend=False,
        inner="box",
        ax=axis,
    )

    axis.set_title(f"Distribuição de '{column}' por Classe")
    axis.set_xlabel("Classe")
    axis.set_ylabel(column)
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels(get_class_labels(order))
    figure.tight_layout()
    return figure


def plot_user_activity(profile: pl.DataFrame) -> Any:
    """Descreve o volume e a janela de observação dos usuários coletados.

    Parameters
    ----------
    profile : pl.DataFrame
        Colunas de perfil (``n_tweets``, ``span_days``, ``active_days``).

    Returns
    -------
    matplotlib.figure.Figure
        Painéis com as distribuições de atividade.

    Examples
    --------
    >>> plot_user_activity(perfil)  # doctest: +SKIP
    """
    columns = [name for name in ("n_tweets", "span_days", "active_days") if name in profile.columns]
    figure, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4), squeeze=False)

    titles = {
        "n_tweets": "Tweets por usuário",
        "span_days": "Janela de observação (dias)",
        "active_days": "Dias ativos",
    }

    for index, column in enumerate(columns):
        axis = axes[0][index]
        sns.histplot(
            data=profile.to_pandas(),
            x=column,
            bins=30,
            color=CATEGORICAL_PALETTE[index % len(CATEGORICAL_PALETTE)],
            ax=axis,
        )
        axis.set_title(titles.get(column, column))
        axis.set_xlabel(column)
        axis.set_ylabel("Número de usuários")

    figure.suptitle("Perfil de Atividade dos Usuários Coletados", fontweight="bold")
    figure.tight_layout()
    return figure
