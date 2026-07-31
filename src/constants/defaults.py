"""Valores padrão e constantes estruturais do projeto.

Aqui ficam apenas constantes que **não** são hiperparâmetros de pesquisa —
essas moram em ``configs/*.yaml`` para poderem ser variadas sem alterar
código. O que está aqui é estrutura: listas de pronomes, nomes de estágios,
sufixos de arquivo.
"""

from __future__ import annotations

from typing import Final

#: Semente global (também presente em ``configs/config.yaml``; este valor é o
#: fallback usado por utilitários que rodam fora do pipeline, como testes).
RANDOM_SEED: Final[int] = 42

#: Etapas do pipeline, na ordem de dependência.
PIPELINE_STAGES: Final[tuple[str, ...]] = (
    "collect",
    "preprocess",
    "label",
    "psych",
    "embed",
    "features",
    "split",
    "train",
    "evaluate",
    "report",
)

#: Dependência direta de cada etapa (a etapa anterior que precisa ter rodado).
STAGE_DEPENDENCIES: Final[dict[str, tuple[str, ...]]] = {
    "collect": (),
    "preprocess": ("collect",),
    "label": ("preprocess",),
    "psych": ("preprocess",),
    "embed": ("preprocess",),
    "features": ("label", "embed"),
    "split": ("features",),
    "train": ("split",),
    "evaluate": ("train",),
    "report": ("evaluate",),
}

# --- Pronomes do português brasileiro ---------------------------------------
# Estrutura da língua, não parâmetro de pesquisa. O uso elevado de pronomes de
# 1ª pessoa do singular é um dos achados mais replicados na literatura de
# detecção de depressão (foco atencional em si mesmo).

FIRST_PERSON_SINGULAR: Final[frozenset[str]] = frozenset(
    {"eu", "me", "mim", "comigo", "meu", "minha", "meus", "minhas"}
)

FIRST_PERSON_PLURAL: Final[frozenset[str]] = frozenset(
    {"nós", "nos", "nosso", "nossa", "nossos", "nossas", "conosco", "a gente"}
)

SECOND_PERSON: Final[frozenset[str]] = frozenset(
    {
        "tu",
        "te",
        "ti",
        "contigo",
        "teu",
        "tua",
        "teus",
        "tuas",
        "você",
        "voce",
        "vocês",
        "voces",
        "seu",
        "sua",
        "seus",
        "suas",
    }
)

THIRD_PERSON: Final[frozenset[str]] = frozenset(
    {
        "ele",
        "ela",
        "eles",
        "elas",
        "lhe",
        "lhes",
        "dele",
        "dela",
        "deles",
        "delas",
        "si",
        "consigo",
    }
)

#: Categoria de pronome -> conjunto de formas.
PRONOUN_GROUPS: Final[dict[str, frozenset[str]]] = {
    "first_singular": FIRST_PERSON_SINGULAR,
    "first_plural": FIRST_PERSON_PLURAL,
    "second": SECOND_PERSON,
    "third": THIRD_PERSON,
}

#: Termos de negação (uso elevado também é marcador linguístico de depressão).
NEGATION_TERMS: Final[frozenset[str]] = frozenset(
    {"não", "nao", "nunca", "jamais", "nada", "ninguém", "ninguem", "nenhum", "nenhuma", "sem"}
)

# --- Formatos e sufixos de arquivo ------------------------------------------
PARQUET_SUFFIX: Final[str] = ".parquet"
JSON_SUFFIX: Final[str] = ".json"
CSV_SUFFIX: Final[str] = ".csv"
NUMPY_SUFFIX: Final[str] = ".npy"
JOBLIB_SUFFIX: Final[str] = ".joblib"

#: Compressão padrão dos parquets (bom equilíbrio tamanho/velocidade).
PARQUET_COMPRESSION: Final[str] = "zstd"

# --- Limites operacionais ---------------------------------------------------
#: Tamanho de lote padrão para inferência em GPU/CPU.
DEFAULT_BATCH_SIZE: Final[int] = 32
#: Segundos entre atualizações da barra de progresso (evita flood no log).
PROGRESS_REFRESH_SECONDS: Final[float] = 0.5
#: Número de casas decimais nas métricas reportadas.
METRIC_PRECISION: Final[int] = 4
