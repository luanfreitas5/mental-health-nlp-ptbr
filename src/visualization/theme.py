"""Tema visual compartilhado por todas as figuras do projeto.

Uma paleta única garante que a mesma classe tenha sempre a mesma cor em
qualquer figura da dissertação — cores inconsistentes entre gráficos obrigam
o leitor a reconsultar a legenda a cada página.

A paleta das classes é ordinal, não categórica: a severidade cresce de
``controle`` a ``ideacao_suicida``, e a cor acompanha essa progressão. Foi
escolhida para permanecer distinguível em impressão em tons de cinza e sob as
formas mais comuns de daltonismo (deuteranopia e protanopia).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from config.logging import get_logger
from constants.labels import CLASS_DISPLAY_NAMES, CLASS_ORDER

logger = get_logger(__name__)

#: Cor de cada classe (severidade crescente).
CLASS_COLORS: dict[str, str] = {
    "controle": "#4C72B0",  # azul
    "depressao": "#DD8452",  # laranja
    "ideacao_suicida": "#C44E52",  # vermelho
    "indefinido": "#8C8C8C",  # cinza
}

#: Paleta sequencial para mapas de calor e matrizes de confusão.
SEQUENTIAL_PALETTE: str = "rocket_r"

#: Paleta divergente para correlações e deltas do Ablation Study.
DIVERGING_PALETTE: str = "vlag"

#: Paleta categórica para séries que não representam classes.
CATEGORICAL_PALETTE: list[str] = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
]

#: Tamanhos padrão de figura (polegadas).
FIGURE_SIZES: dict[str, tuple[float, float]] = {
    "small": (6.0, 4.0),
    "medium": (10.0, 5.0),
    "large": (12.0, 7.0),
    "square": (7.0, 7.0),
    "wide": (14.0, 5.0),
}


def apply_theme(dpi: int = 300) -> None:
    """Aplica o tema visual global do projeto.

    Parameters
    ----------
    dpi : int, optional
        Resolução das figuras, by default 300 (padrão para publicação).

    Examples
    --------
    >>> apply_theme()
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    mpl.rcParams.update(
        {
            "figure.dpi": 120,  # tela
            "savefig.dpi": dpi,  # arquivo
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "grid.alpha": 0.3,
            # DejaVu Sans é a fonte padrão do matplotlib e cobre todos os
            # acentos do português, evitando o quadrado vazio no lugar de "ç".
            "font.family": "DejaVu Sans",
        }
    )


def get_class_palette(classes: list[str] | None = None) -> list[str]:
    """Retorna a paleta de cores na ordem canônica das classes.

    Parameters
    ----------
    classes : list of str, optional
        Classes desejadas, by default :data:`constants.labels.CLASS_ORDER`.

    Returns
    -------
    list of str
        Cores em hexadecimal.

    Examples
    --------
    >>> get_class_palette()[0]
    '#4C72B0'
    """
    names = classes or list(CLASS_ORDER)
    return [CLASS_COLORS.get(name, "#8C8C8C") for name in names]


def get_class_labels(classes: list[str] | None = None) -> list[str]:
    """Retorna os nomes das classes para exibição em pt-BR.

    Parameters
    ----------
    classes : list of str, optional
        Classes desejadas, by default :data:`constants.labels.CLASS_ORDER`.

    Returns
    -------
    list of str
        Nomes formatados (ex.: ``"Ideação Suicida"``).

    Examples
    --------
    >>> get_class_labels()[2]
    'Ideação Suicida'
    """
    names = classes or list(CLASS_ORDER)
    return [CLASS_DISPLAY_NAMES.get(name, name) for name in names]


def save_figure(
    figure: Any,
    directory: Path,
    name: str,
    *,
    formats: tuple[str, ...] = ("png", "svg"),
    dpi: int = 300,
    close: bool = True,
) -> list[Path]:
    """Grava uma figura nos formatos configurados.

    PNG para inserção rápida em documentos e apresentações; SVG porque é
    vetorial e não perde qualidade na diagramação final da dissertação.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figura a gravar.
    directory : Path
        Diretório de destino.
    name : str
        Nome do arquivo, sem extensão.
    formats : tuple of str, optional
        Formatos de saída, by default ``("png", "svg")``.
    dpi : int, optional
        Resolução, by default 300.
    close : bool, optional
        Fecha a figura após gravar, by default True. Manter figuras abertas
        num laço de dezenas de gráficos esgota a memória do matplotlib.

    Returns
    -------
    list of Path
        Caminhos gravados.

    Examples
    --------
    >>> save_figure(fig, Path("reports/figures"), "matriz_confusao")  # doctest: +SKIP
    """
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for extension in formats:
        path = target_dir / f"{name}.{extension}"
        figure.savefig(path, dpi=dpi, format=extension)
        written.append(path)

    if close:
        plt.close(figure)

    logger.debug("Figura '%s' gravada em %s.", name, ", ".join(fmt for fmt in formats))
    return written
