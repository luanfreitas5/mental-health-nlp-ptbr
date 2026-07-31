"""Coleta longitudinal de tweets via ``twscrape``.

Implementa a estratégia da proposta: busca semente por palavras-chave e
hashtags, extração dos autores, coleta retrospectiva do histórico e gravação
de um Parquet por usuário.

Três decisões estruturais:

1. **Pseudonimização na ingestão.** O ``user_id`` é convertido em hash antes
   de qualquer gravação — nenhum handle chega a tocar o disco.
2. **Coleta retomável.** Um arquivo por usuário e verificação do que já existe
   permitem continuar de onde parou: a coleta leva dias e interrupções são
   certas, não hipotéticas.
3. **Import tardio do ``twscrape``.** A biblioteca só é importada quando a
   coleta roda de fato, de modo que o restante do pipeline (e os testes)
   funcione sem ela instalada e sem contas configuradas.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from config.logging import get_logger
from config.settings import CollectionConfig
from data.queries import SearchQuery, build_queries
from data.reader import list_collected_users
from data.writer import write_parquet
from exceptions.data import CollectionError
from exceptions.model import MissingDependencyError
from utils.hashing import pseudonymize
from utils.progress import build_progress

if TYPE_CHECKING:  # pragma: no cover — apenas para o type checker
    from twscrape import API

logger = get_logger(__name__)


@dataclass(frozen=True)
class CandidateUser:
    """Usuário identificado na busca semente.

    Attributes
    ----------
    user_id : str
        Identificador pseudonimizado.
    raw_id : int
        Identificador original — mantido **apenas em memória**, para as
        chamadas de API, e nunca persistido.
    candidate_label : str
        Rótulo candidato herdado do grupo de busca.
    source_group : str
        Grupo de coleta de origem.
    """

    user_id: str
    raw_id: int
    candidate_label: str
    source_group: str


def _import_twscrape() -> Any:
    """Importa o ``twscrape`` sob demanda, com erro explicativo se ausente."""
    try:
        import twscrape
    except ImportError as error:
        raise MissingDependencyError(
            "twscrape não está instalado. Rode 'uv sync --extra collect' e configure "
            "as contas conforme docs/guides/collection.md."
        ) from error
    return twscrape


def _tweet_to_record(tweet: Any, salt: str, query: SearchQuery | None = None) -> dict[str, Any]:
    """Converte um objeto de tweet do twscrape num registro plano e pseudonimizado.

    Parameters
    ----------
    tweet : Any
        Objeto ``Tweet`` do twscrape.
    salt : str
        Segredo de pseudonimização.
    query : SearchQuery, optional
        Consulta que originou o tweet (ausente na coleta de histórico).

    Returns
    -------
    dict
        Registro compatível com :class:`schemas.tweets.RawTweetSchema`.
    """
    return {
        "user_id": pseudonymize(tweet.user.id, salt),
        "tweet_id": pseudonymize(tweet.id, salt),
        "text": tweet.rawContent,
        "created_at": tweet.date,
        "language": getattr(tweet, "lang", None),
        "is_reply": tweet.inReplyToTweetId is not None,
        "is_retweet": getattr(tweet, "retweetedTweet", None) is not None,
        "like_count": int(tweet.likeCount or 0),
        "reply_count": int(tweet.replyCount or 0),
        "retweet_count": int(tweet.retweetCount or 0),
        "quote_count": int(tweet.quoteCount or 0),
        "source_query": query.term if query else None,
        "source_group": query.group if query else None,
    }


def _user_to_record(user: Any, salt: str) -> dict[str, Any]:
    """Extrai os metadados públicos do autor, sem nenhum campo identificável."""
    return {
        "user_id": pseudonymize(user.id, salt),
        "followers_count": int(getattr(user, "followersCount", 0) or 0),
        "following_count": int(getattr(user, "friendsCount", 0) or 0),
        "statuses_count": int(getattr(user, "statusesCount", 0) or 0),
        "account_created_at": getattr(user, "created", None),
        "is_verified": bool(getattr(user, "verified", False)),
    }


class TweetCollector:
    """Coletor longitudinal de tweets.

    Parameters
    ----------
    config : CollectionConfig
        Seção validada de ``configs/collection.yaml``.
    salt : str
        Segredo de pseudonimização (do ``.env``).
    output_dir : Path
        Diretório dos históricos por usuário.
    seed_dir : Path
        Diretório dos resultados da busca semente.

    Examples
    --------
    >>> collector = TweetCollector(config, salt, historicos, sementes)  # doctest: +SKIP
    >>> collector.run()  # doctest: +SKIP
    """

    def __init__(
        self,
        config: CollectionConfig,
        salt: str,
        output_dir: Path,
        seed_dir: Path,
    ) -> None:
        self.config = config
        self.salt = salt
        self.output_dir = Path(output_dir)
        self.seed_dir = Path(seed_dir)
        self._api: API | None = None

    async def _get_api(self) -> API:
        """Inicializa (uma vez) o cliente do twscrape com o pool de contas."""
        if self._api is not None:
            return self._api

        twscrape = _import_twscrape()
        accounts_db = Path(self.config.twscrape.accounts_db)
        if self.config.twscrape.raise_when_no_account and not accounts_db.is_file():
            raise CollectionError(
                f"Banco de contas do twscrape não encontrado em {accounts_db}. "
                "Ver docs/guides/collection.md para configurar as contas."
            )

        api = twscrape.API(str(accounts_db))
        await api.pool.login_all()
        self._api = api
        return api

    async def search_seed_tweets(self, queries: list[SearchQuery]) -> pl.DataFrame:
        """Executa a busca semente e devolve os tweets encontrados.

        Parameters
        ----------
        queries : list of SearchQuery
            Consultas construídas por :func:`data.queries.build_queries`.

        Returns
        -------
        pl.DataFrame
            Tweets da busca semente, já pseudonimizados.

        Raises
        ------
        CollectionError
            Se todas as consultas falharem.

        Examples
        --------
        >>> await collector.search_seed_tweets(queries)  # doctest: +SKIP
        """
        api = await self._get_api()
        records: list[dict[str, Any]] = []
        failures = 0

        with build_progress() as progress:
            task = progress.add_task("Busca semente", total=len(queries))
            for query in queries:
                try:
                    # Não convertido para list comprehension: se a consulta
                    # falhar no meio da busca, os tweets já coletados até ali
                    # devem ser preservados em `records`, não descartados.
                    async for tweet in api.search(
                        query.query, limit=self.config.seed_search.limit_per_query
                    ):
                        records.append(  # noqa: PERF401
                            _tweet_to_record(tweet, self.salt, query)
                        )
                except (OSError, RuntimeError, ValueError) as error:
                    failures += 1
                    # Não logamos `query.query`: contém os termos de risco e
                    # poluiria o log com conteúdo sensível do protocolo.
                    logger.warning("Falha na consulta %s/%s: %s", query.group, query.kind, error)
                finally:
                    progress.advance(task)
                await asyncio.sleep(60 / self.config.rate_limit.requests_per_minute)

        if failures == len(queries):
            raise CollectionError(
                f"Todas as {failures} consultas falharam. Verifique as contas do twscrape "
                "e a conectividade."
            )

        frame = pl.DataFrame(records) if records else pl.DataFrame()
        logger.info(
            "Busca semente concluída: %d tweets, %d consultas com falha.", frame.height, failures
        )
        return frame

    def extract_candidates(self, seed_tweets: pl.DataFrame) -> list[CandidateUser]:
        """Extrai os autores únicos da busca semente.

        Quando o mesmo usuário aparece em mais de um grupo (por exemplo, em
        ``depressao`` e em ``ideacao_suicida``), prevalece o rótulo candidato
        de maior severidade — a mesma precedência usada na rotulação.

        Parameters
        ----------
        seed_tweets : pl.DataFrame
            Saída de :meth:`search_seed_tweets`.

        Returns
        -------
        list of CandidateUser
            Candidatos únicos.

        Examples
        --------
        >>> collector.extract_candidates(sementes)  # doctest: +SKIP
        """
        if seed_tweets.is_empty():
            return []

        from constants.labels import CLASS_PRECEDENCE

        severity = {label: index for index, label in enumerate(CLASS_PRECEDENCE)}
        best: dict[str, CandidateUser] = {}

        for row in seed_tweets.select(["user_id", "source_group"]).iter_rows(named=True):
            group = row["source_group"] or "controle"
            candidate = CandidateUser(
                user_id=row["user_id"],
                raw_id=0,  # preenchido na coleta do histórico
                candidate_label=group,
                source_group=group,
            )
            current = best.get(candidate.user_id)
            if current is None or severity.get(group, 99) < severity.get(
                current.candidate_label, 99
            ):
                best[candidate.user_id] = candidate

        candidates = sorted(best.values(), key=lambda item: item.user_id)
        logger.info("Identificados %d usuários candidatos únicos.", len(candidates))
        return candidates

    async def collect_user_history(self, raw_user_id: int) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Coleta o histórico retrospectivo de um usuário.

        Parameters
        ----------
        raw_user_id : int
            Identificador original do usuário na plataforma.

        Returns
        -------
        tuple
            ``(histórico, metadados)``. O histórico vem vazio quando o usuário
            não atinge o mínimo de tweets configurado.

        Examples
        --------
        >>> await collector.collect_user_history(123)  # doctest: +SKIP
        """
        api = await self._get_api()
        settings = self.config.user_history
        cutoff = datetime.now(timezone.utc) - timedelta(days=settings.window_days)

        records: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {}

        async for tweet in api.user_tweets(raw_user_id, limit=settings.max_tweets_per_user):
            if not metadata and settings.collect_user_metadata:
                metadata = _user_to_record(tweet.user, self.salt)

            created = tweet.date
            if created is not None and created < cutoff:
                continue
            if settings.exclude_retweets and getattr(tweet, "retweetedTweet", None) is not None:
                continue
            if settings.exclude_replies and tweet.inReplyToTweetId is not None:
                continue

            records.append(_tweet_to_record(tweet, self.salt))

        if len(records) < settings.min_tweets_per_user:
            logger.debug(
                "Usuário descartado: %d tweets (< %d exigidos).",
                len(records),
                settings.min_tweets_per_user,
            )
            return pl.DataFrame(), metadata

        return pl.DataFrame(records), metadata

    async def collect_all(self, candidates: list[CandidateUser]) -> pl.DataFrame:
        """Coleta o histórico de todos os candidatos ainda não coletados.

        Parameters
        ----------
        candidates : list of CandidateUser
            Candidatos identificados na busca semente.

        Returns
        -------
        pl.DataFrame
            Metadados dos usuários efetivamente coletados.

        Examples
        --------
        >>> await collector.collect_all(candidatos)  # doctest: +SKIP
        """
        already = list_collected_users(self.output_dir)
        pending = [candidate for candidate in candidates if candidate.user_id not in already]
        logger.info("Histórico: %d já coletados, %d pendentes.", len(already), len(pending))

        metadata_records: list[dict[str, Any]] = []
        with build_progress() as progress:
            task = progress.add_task("Coletando históricos", total=len(pending))
            for index, candidate in enumerate(pending, start=1):
                try:
                    history, metadata = await self.collect_user_history(candidate.raw_id)
                except (OSError, RuntimeError, ValueError) as error:
                    logger.warning("Falha ao coletar histórico de um usuário: %s", error)
                    progress.advance(task)
                    continue

                if not history.is_empty():
                    history = history.with_columns(
                        pl.lit(candidate.candidate_label).alias("candidate_label")
                    )
                    write_parquet(
                        history.drop("candidate_label"),
                        self.output_dir / f"{candidate.user_id}.parquet",
                        log_hash=False,
                    )
                    if metadata:
                        metadata_records.append(metadata)

                progress.advance(task)
                if index % self.config.rate_limit.checkpoint_every_n_users == 0:
                    logger.info("Checkpoint: %d/%d usuários processados.", index, len(pending))
                await asyncio.sleep(60 / self.config.rate_limit.requests_per_minute)

        return pl.DataFrame(metadata_records) if metadata_records else pl.DataFrame()

    async def run_async(self) -> pl.DataFrame:
        """Executa a coleta completa (busca semente + históricos).

        Returns
        -------
        pl.DataFrame
            Metadados dos usuários coletados.

        Examples
        --------
        >>> await collector.run_async()  # doctest: +SKIP
        """
        queries = build_queries(self.config.seed_search)
        seed_tweets = await self.search_seed_tweets(queries)

        if not seed_tweets.is_empty():
            write_parquet(seed_tweets, self.seed_dir / "seed_tweets.parquet")

        candidates = self.extract_candidates(seed_tweets)
        return await self.collect_all(candidates)

    def run(self) -> pl.DataFrame:
        """Executa a coleta de forma síncrona (fachada sobre :meth:`run_async`).

        Returns
        -------
        pl.DataFrame
            Metadados dos usuários coletados.

        Examples
        --------
        >>> TweetCollector(config, salt, saida, semente).run()  # doctest: +SKIP
        """
        return asyncio.run(self.run_async())
