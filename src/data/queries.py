"""Construção das consultas de busca a partir dos arquivos ``.txt``.

Manter palavras-chave e hashtags em arquivos de texto (e não no código) é
requisito metodológico: a lista de termos é parte do método de amostragem,
precisa ser auditável no histórico do git e revisável por terceiros sem que
ninguém precise ler Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from config.logging import get_logger
from config.paths import get_paths
from config.settings import SeedSearchSection
from utils.files import read_terms_file

logger = get_logger(__name__)


@dataclass(frozen=True)
class SearchQuery:
    """Uma consulta de busca pronta para o twscrape.

    Attributes
    ----------
    query : str
        Expressão de busca completa, com operadores e filtros.
    term : str
        Termo original que originou a consulta.
    kind : str
        ``"keyword"`` ou ``"hashtag"``.
    group : str
        Grupo de coleta (``depressao``, ``ideacao_suicida``, ``controle``).
    candidate_label : str
        Rótulo candidato atribuído a quem for encontrado por esta consulta.
    """

    query: str
    term: str
    kind: str
    group: str
    candidate_label: str


def build_query_string(
    term: str,
    *,
    language: str,
    since: date,
    until: date,
    exclude_retweets: bool,
    exclude_replies: bool,
) -> str:
    """Monta a expressão de busca no dialeto do X/Twitter.

    Termos com espaço são envolvidos em aspas para busca de expressão exata —
    sem isso, ``quero morrer`` retornaria qualquer tweet com "quero" **ou**
    "morrer", o que destruiria a precisão da amostragem.

    Parameters
    ----------
    term : str
        Palavra-chave ou hashtag.
    language : str
        Código do idioma (ex.: ``"pt"``).
    since, until : date
        Janela temporal da busca.
    exclude_retweets, exclude_replies : bool
        Filtros de tipo de publicação.

    Returns
    -------
    str
        Expressão de busca completa.

    Examples
    --------
    >>> build_query_string(
    ...     "quero morrer",
    ...     language="pt",
    ...     since=date(2024, 1, 1),
    ...     until=date(2024, 2, 1),
    ...     exclude_retweets=True,
    ...     exclude_replies=False,
    ... )
    '"quero morrer" lang:pt since:2024-01-01 until:2024-02-01 -filter:retweets'
    """
    expression = f'"{term}"' if " " in term.strip() and not term.startswith("#") else term

    parts = [
        expression,
        f"lang:{language}",
        f"since:{since.isoformat()}",
        f"until:{until.isoformat()}",
    ]
    if exclude_retweets:
        parts.append("-filter:retweets")
    if exclude_replies:
        parts.append("-filter:replies")

    return " ".join(parts)


def build_queries(config: SeedSearchSection) -> list[SearchQuery]:
    """Constrói todas as consultas da busca semente.

    Parameters
    ----------
    config : SeedSearchSection
        Seção ``seed_search`` de ``configs/collection.yaml``.

    Returns
    -------
    list of SearchQuery
        Consultas de todos os grupos, em ordem determinística (grupo, tipo,
        termo) — importante para que uma coleta interrompida possa ser
        retomada exatamente do ponto onde parou.

    Raises
    ------
    FileNotFoundError
        Se algum arquivo de termos declarado não existir.

    Examples
    --------
    >>> queries = build_queries(load_config().collection.seed_search)  # doctest: +SKIP
    >>> queries[0].kind  # doctest: +SKIP
    'hashtag'
    """
    queries_dir = get_paths().queries.root
    queries: list[SearchQuery] = []

    for group_name in sorted(config.groups):
        group = config.groups[group_name]

        for kind, filename in (("hashtag", group.hashtag_file), ("keyword", group.keyword_file)):
            if not filename:
                continue

            terms = read_terms_file(queries_dir / filename)
            queries.extend(
                SearchQuery(
                    query=build_query_string(
                        term,
                        language=config.language,
                        since=config.since,
                        until=config.until,
                        exclude_retweets=config.exclude_retweets,
                        exclude_replies=config.exclude_replies,
                    ),
                    term=term,
                    kind=kind,
                    group=group_name,
                    candidate_label=group.candidate_label,
                )
                for term in terms
            )

    logger.info(
        "Construídas %d consultas em %d grupos: %s.",
        len(queries),
        len(config.groups),
        ", ".join(sorted(config.groups)),
    )
    return queries


def summarize_queries(queries: list[SearchQuery]) -> dict[str, dict[str, int]]:
    """Resume a quantidade de consultas por grupo e tipo.

    Parameters
    ----------
    queries : list of SearchQuery
        Consultas construídas.

    Returns
    -------
    dict
        ``{grupo: {"keyword": n, "hashtag": n, "total": n}}``.

    Examples
    --------
    >>> resumo = summarize_queries([SearchQuery("q", "t", "keyword", "depressao", "depressao")])
    >>> resumo["depressao"]["total"]
    1
    """
    summary: dict[str, dict[str, int]] = {}
    for query in queries:
        group = summary.setdefault(query.group, {"keyword": 0, "hashtag": 0, "total": 0})
        group[query.kind] = group.get(query.kind, 0) + 1
        group["total"] += 1
    return summary
