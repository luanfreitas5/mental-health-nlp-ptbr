"""Projeção 2D dos embeddings de usuários (UMAP e t-SNE).

A projeção é **exploratória**: se as classes já se separam visualmente, há
sinal semântico aproveitável; se não se separam, a tarefa depende dos
atributos estruturados. Em nenhum dos casos a figura é evidência de
desempenho — separação visual em 2D não implica separabilidade no espaço
original, nem o contrário.

A projeção é ajustada sobre o conjunto inteiro, e não apenas o treino, porque
serve à visualização e nunca alimenta um modelo. Se essas coordenadas
virassem features, isso passaria a ser vazamento.
"""

from __future__ import annotations

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config.logging import get_logger
from constants.columns import USER_ID, USER_LABEL
from constants.labels import CLASS_ORDER
from schemas.features import list_feature_columns
from visualization.theme import FIGURE_SIZES, get_class_labels, get_class_palette

logger = get_logger(__name__)


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "umap",
    random_state: int = 42,
) -> tuple[np.ndarray, str]:
    """Projeta os embeddings em duas dimensões.

    Parameters
    ----------
    embeddings : np.ndarray
        Matriz ``(n_usuarios, dim)``.
    method : {'umap', 'tsne', 'pca'}, optional
        Técnica de projeção, by default ``"umap"``.
    random_state : int, optional
        Semente, by default 42.

    Returns
    -------
    tuple
        ``(coordenadas 2D, método efetivamente usado)``. O método pode
        diferir do solicitado quando a biblioteca não está instalada ou a
        amostra é pequena demais.

    Examples
    --------
    >>> coords, usado = reduce_dimensions(np.random.rand(50, 10), method="pca")
    >>> coords.shape
    (50, 2)
    """
    n_samples = embeddings.shape[0]

    if method == "umap":
        try:
            import umap
        except ImportError:
            logger.warning("Pacote 'umap-learn' ausente: usando t-SNE como alternativa.")
            method = "tsne"
        else:
            reducer = umap.UMAP(
                n_components=2,
                random_state=random_state,
                n_neighbors=min(15, max(2, n_samples - 1)),
                min_dist=0.1,
            )
            # Os stubs do umap-learn anotam `fit_transform` com um retorno
            # genérico demais (inclui `coo_matrix`, nunca produzido nesta
            # chamada, que usa `array` denso como entrada).
            coords = cast(np.ndarray, reducer.fit_transform(embeddings))
            return coords, "UMAP"

    if method == "tsne":
        # A perplexidade precisa ser menor que o número de amostras; com
        # poucos usuários o t-SNE falharia com o valor padrão de 30.
        perplexity = min(30, max(5, (n_samples - 1) // 3))
        reducer = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=perplexity,
            init="pca",
        )
        return reducer.fit_transform(embeddings), f"t-SNE (perplexidade={perplexity})"

    return PCA(n_components=2, random_state=random_state).fit_transform(embeddings), "PCA"


def plot_embedding_projection(
    features: pl.DataFrame,
    method: str = "umap",
    random_state: int = 42,
) -> Any | None:
    """Projeta e plota os embeddings de usuários, coloridos por classe.

    Parameters
    ----------
    features : pl.DataFrame
        Matriz de atributos com colunas semânticas e ``user_label``.
    method : {'umap', 'tsne', 'pca'}, optional
        Técnica de projeção, by default ``"umap"``.
    random_state : int, optional
        Semente, by default 42.

    Returns
    -------
    matplotlib.figure.Figure or None
        Dispersão 2D, ou ``None`` se não houver colunas semânticas.

    Examples
    --------
    >>> plot_embedding_projection(matriz, method="pca")  # doctest: +SKIP
    """
    columns = list_feature_columns(features, ["semantic"])
    if not columns:
        logger.warning("Nenhuma coluna semântica na matriz: projeção de embeddings não gerada.")
        return None

    embeddings = features.select(columns).to_numpy().astype(np.float64)
    coordinates, used_method = reduce_dimensions(embeddings, method, random_state)

    figure, axis = plt.subplots(figsize=FIGURE_SIZES["square"])

    if USER_LABEL in features.columns:
        labels = features[USER_LABEL].to_list()
        classes = [name for name in CLASS_ORDER if name in set(labels)]
        colors = get_class_palette(classes)
        display = get_class_labels(classes)

        for index, class_name in enumerate(classes):
            mask = np.array([value == class_name for value in labels])
            axis.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                c=colors[index],
                label=display[index],
                alpha=0.65,
                s=28,
                edgecolors="white",
                linewidths=0.4,
            )
        axis.legend(title="Classe")
    else:
        axis.scatter(coordinates[:, 0], coordinates[:, 1], alpha=0.65, s=28)

    axis.set_title(
        f"Projeção {used_method} dos Embeddings de Usuários\n"
        f"({embeddings.shape[0]} usuários, {embeddings.shape[1]} dimensões originais)"
    )
    axis.set_xlabel(f"{used_method} — dimensão 1")
    axis.set_ylabel(f"{used_method} — dimensão 2")
    figure.tight_layout()
    return figure


def plot_interaction_network(
    tweets: pl.DataFrame,
    labels: pl.DataFrame,
    max_nodes: int = 150,
) -> Any | None:
    """Plota a rede de similaridade entre usuários.

    Como as menções são removidas na etapa de anonimização (LGPD), a rede não
    pode ser construída sobre interações reais. A alternativa adotada — e a
    limitação — é conectar usuários por **similaridade de vocabulário**, o que
    responde a uma pergunta próxima ("quem escreve de forma parecida?") sem
    reintroduzir dado identificável.

    Parameters
    ----------
    tweets : pl.DataFrame
        Tweets limpos.
    labels : pl.DataFrame
        Rótulos por usuário.
    max_nodes : int, optional
        Número máximo de usuários no grafo, by default 150. Grafos maiores
        viram uma mancha ilegível.

    Returns
    -------
    matplotlib.figure.Figure or None
        Grafo de similaridade, ou ``None`` se o ``networkx`` não estiver
        instalado.

    Examples
    --------
    >>> plot_interaction_network(tweets, rotulos)  # doctest: +SKIP
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("Pacote 'networkx' ausente: rede de similaridade não gerada.")
        return None

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    from constants.columns import TEXT_CLEAN

    documents = (
        tweets.group_by(USER_ID)
        .agg(pl.col(TEXT_CLEAN).str.join(" ").alias("documento"))
        .join(labels.select([USER_ID, USER_LABEL]), on=USER_ID, how="inner")
        .head(max_nodes)
    )
    if documents.height < 3:
        logger.warning("Usuários insuficientes para construir a rede de similaridade.")
        return None

    matrix = TfidfVectorizer(max_features=2000, min_df=1).fit_transform(
        documents["documento"].to_list()
    )
    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0.0)

    # Limiar no percentil 95: mantém apenas as ligações mais fortes, senão o
    # grafo fica quase completo e não comunica estrutura alguma.
    threshold = float(np.percentile(similarity[similarity > 0], 95)) if similarity.any() else 1.0

    graph = nx.Graph()
    user_ids = documents[USER_ID].to_list()
    user_labels = documents[USER_LABEL].to_list()

    for index, user_id in enumerate(user_ids):
        graph.add_node(user_id, label=user_labels[index])

    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            if similarity[i, j] >= threshold:
                graph.add_edge(user_ids[i], user_ids[j], weight=float(similarity[i, j]))

    figure, axis = plt.subplots(figsize=FIGURE_SIZES["square"])
    positions = nx.spring_layout(graph, seed=42, k=0.35)

    color_by_class = dict(zip(CLASS_ORDER, get_class_palette(), strict=False))
    node_colors = [color_by_class.get(graph.nodes[node]["label"], "#8C8C8C") for node in graph]

    nx.draw_networkx_edges(graph, positions, alpha=0.15, ax=axis)
    nx.draw_networkx_nodes(
        graph, positions, node_color=node_colors, node_size=60, alpha=0.85, ax=axis
    )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=name)
        for name, color in zip(get_class_labels(), get_class_palette(), strict=False)
    ]
    axis.legend(handles=handles, title="Classe", loc="upper right")

    axis.set_title(
        f"Rede de Similaridade Lexical entre Usuários\n"
        f"({graph.number_of_nodes()} usuários, {graph.number_of_edges()} conexões)"
    )
    axis.axis("off")
    figure.tight_layout()
    return figure
