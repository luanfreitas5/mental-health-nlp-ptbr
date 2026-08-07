"""Fixtures compartilhadas pelos testes.

Todos os dados são **sintéticos**. Dados reais deste projeto são sensíveis
(saúde mental, pessoas identificáveis na origem) e não podem entrar no
repositório sob nenhuma circunstância — nem mesmo em testes.

Os textos sintéticos imitam os padrões que o pipeline precisa detectar
(vocabulário de risco, atividade noturna, negatividade persistente) sem
reproduzir conteúdo real de ninguém.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from config.settings import Config, load_config
from constants.labels import CLASS_ORDER

RANDOM_SEED = 42

#: Frases sintéticas por classe, escritas para o teste.
SYNTHETIC_TEXTS: dict[str, list[str]] = {
    "controle": [
        "hoje o time jogou muito bem, que partida incrível",
        "terminei de ler um livro ótimo neste fim de semana",
        "aprendendo python para análise de dados, muito legal",
        "a viagem para a praia foi maravilhosa demais",
        "recomendo esse filme para todo mundo, vale a pena",
    ],
    "depressao": [
        "me sinto sozinho todos os dias e ninguém percebe",
        "sem esperança de que alguma coisa vá melhorar",
        "não consigo dormir, mais uma noite em claro",
        "cansado de tentar, nada faz sentido para mim",
        "vazio por dentro, só queria sumir um pouco",
    ],
    "ideacao_suicida": [
        "queria não ter nascido, isso tudo é demais",
        "penso em acabar com tudo quase toda noite",
        "não quero mais viver assim, estou exausto",
        "seria melhor sem mim, todo mundo ficaria bem",
        "não vejo saída nenhuma para essa situação",
    ],
}


@pytest.fixture(scope="session")
def config() -> Config:
    """Configuração real do projeto, carregada de ``configs/``.

    Returns
    -------
    Config
        Configuração validada.
    """
    return load_config()


@pytest.fixture
def rng() -> np.random.Generator:
    """Gerador aleatório com semente fixa.

    Returns
    -------
    np.random.Generator
        Gerador determinístico.
    """
    return np.random.default_rng(RANDOM_SEED)


def make_tweets(
    n_users_per_class: int = 4,
    n_tweets_per_user: int = 40,
    seed: int = RANDOM_SEED,
) -> pl.DataFrame:
    """Gera um conjunto sintético de tweets, já pseudonimizado.

    Parameters
    ----------
    n_users_per_class : int, optional
        Usuários por classe, by default 4.
    n_tweets_per_user : int, optional
        Tweets por usuário, by default 40.
    seed : int, optional
        Semente, by default 42.

    Returns
    -------
    pl.DataFrame
        Tweets conformes a :class:`schemas.tweets.RawTweetSchema`.

    Examples
    --------
    >>> make_tweets(n_users_per_class=1, n_tweets_per_user=5).height
    15
    """
    generator = np.random.default_rng(seed)
    origin = datetime(2024, 6, 1, 12, 0, 0)
    records: list[dict[str, object]] = []

    for class_index, class_name in enumerate(CLASS_ORDER):
        pool = SYNTHETIC_TEXTS[class_name]

        for user_index in range(n_users_per_class):
            user_id = f"u_{class_index}{user_index:02d}{'a' * 12}"

            for tweet_index in range(n_tweets_per_user):
                # Usuários das classes de risco recebem mais publicações de
                # madrugada, para que as features circadianas tenham sinal.
                hour = (
                    int(generator.integers(0, 5))
                    if class_name != "controle" and generator.random() < 0.4
                    else int(generator.integers(8, 23))
                )
                created = origin + timedelta(days=int(generator.integers(0, 200)), hours=hour - 12)

                # O índice do tweet entra no texto para que cada publicação
                # seja única: sem isso, o ciclo pelo pool de frases produz
                # duplicatas exatas dentro do usuário, que a deduplicação
                # remove e derrubam a contagem abaixo do filtro de atividade
                # (>= 30 tweets/usuário).
                records.append(
                    {
                        "user_id": user_id,
                        "tweet_id": f"u_{class_index}{user_index:02d}{tweet_index:04d}{'b' * 6}",
                        "text": f"{pool[tweet_index % len(pool)]} ({tweet_index})",
                        "created_at": created,
                        "language": "pt",
                        "is_reply": bool(generator.random() < 0.3),
                        "is_retweet": False,
                        "like_count": int(generator.integers(0, 50)),
                        "reply_count": int(generator.integers(0, 10)),
                        "retweet_count": int(generator.integers(0, 20)),
                        "quote_count": int(generator.integers(0, 5)),
                        "source_query": class_name,
                        "source_group": class_name,
                    }
                )

    return pl.DataFrame(records).with_columns(pl.col("created_at").cast(pl.Datetime("us")))


@pytest.fixture
def raw_tweets() -> pl.DataFrame:
    """Tweets brutos sintéticos.

    Returns
    -------
    pl.DataFrame
        Conjunto conforme :class:`schemas.tweets.RawTweetSchema`.
    """
    return make_tweets()


@pytest.fixture
def clean_tweets(raw_tweets: pl.DataFrame) -> pl.DataFrame:
    """Tweets com as colunas de texto processado.

    Returns
    -------
    pl.DataFrame
        Conjunto conforme :class:`schemas.tweets.CleanTweetSchema`.
    """
    return raw_tweets.with_columns(
        pl.col("text").alias("text_normalized"),
        pl.col("text").str.to_lowercase().alias("text_clean"),
    )


@pytest.fixture
def labeled_tweets(clean_tweets: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Tweets com sentimento sintético, correlacionado à classe de origem.

    Returns
    -------
    pl.DataFrame
        Conjunto conforme :class:`schemas.tweets.LabeledTweetSchema`.
    """
    sentiments: list[str] = []
    scores: list[float] = []

    for group in clean_tweets["source_group"].to_list():
        if group == "controle":
            sentiment = str(rng.choice(["positivo", "neutro", "negativo"], p=[0.5, 0.35, 0.15]))
        else:
            sentiment = str(rng.choice(["negativo", "neutro", "positivo"], p=[0.7, 0.25, 0.05]))
        sentiments.append(sentiment)
        scores.append(float(rng.uniform(0.6, 0.99)))

    polarity = {"positivo": 1.0, "negativo": -1.0, "neutro": 0.0}
    return clean_tweets.with_columns(
        pl.Series("sentiment", sentiments),
        pl.Series("sentiment_score", scores),
        pl.Series("sentiment_polarity", [polarity[value] for value in sentiments]),
    )


@pytest.fixture
def user_labels(raw_tweets: pl.DataFrame) -> pl.DataFrame:
    """Rótulos sintéticos por usuário.

    Returns
    -------
    pl.DataFrame
        Conjunto conforme :class:`schemas.users.UserLabelSchema`.
    """
    return (
        raw_tweets.group_by("user_id")
        .agg(pl.col("source_group").first().alias("user_label"))
        .with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("user_label_multilabel"),
            pl.col("user_label").alias("candidate_label"),
            pl.lit("weak_supervision").alias("label_source"),
            pl.lit(0.85).alias("label_agreement"),
            pl.lit(None, dtype=pl.Utf8).alias("manual_label"),
        )
        .sort("user_id")
    )


@pytest.fixture
def feature_matrix(rng: np.random.Generator) -> pl.DataFrame:
    """Matriz sintética de atributos por usuário, com sinal aprendível.

    O sinal é deliberado: as classes diferem na média de algumas colunas, para
    que os testes de modelo verifiquem aprendizado, e não apenas ausência de
    exceção.

    Returns
    -------
    pl.DataFrame
        Matriz com colunas-chave e atributos dos seis grupos.
    """
    n_per_class = 30
    records: list[dict[str, object]] = []

    for class_index, class_name in enumerate(CLASS_ORDER):
        for user_index in range(n_per_class):
            shift = class_index * 0.8
            records.append(
                {
                    "user_id": f"u_{class_index}{user_index:02d}{'c' * 12}",
                    "user_label": class_name,
                    "n_tweets": int(rng.integers(30, 400)),
                    "span_days": int(rng.integers(60, 365)),
                    "active_days": int(rng.integers(20, 200)),
                    "night_activity_ratio": float(
                        np.clip(rng.normal(0.1 + 0.1 * class_index, 0.05), 0, 1)
                    ),
                    "ling_death_ratio": float(np.clip(rng.normal(0.02 * class_index, 0.01), 0, 1)),
                    "ling_ttr": float(rng.uniform(0.2, 0.8)),
                    "ling_pronoun_first_singular": float(
                        np.clip(rng.normal(0.05 + 0.02 * class_index, 0.01), 0, 1)
                    ),
                    "emo_negativo_ratio": float(
                        np.clip(rng.normal(0.2 + 0.25 * class_index, 0.08), 0, 1)
                    ),
                    "emo_polarity_mean": float(
                        np.clip(rng.normal(0.4 - 0.4 * class_index, 0.2), -1, 1)
                    ),
                    "temp_night_activity_ratio": float(
                        np.clip(rng.normal(0.1 + 0.1 * class_index, 0.05), 0, 1)
                    ),
                    "temp_tweets_per_day": float(rng.uniform(0.5, 8.0)),
                    "behav_follower_following_ratio": float(rng.uniform(0.1, 5.0)),
                    "behav_reply_ratio": float(rng.uniform(0.0, 0.8)),
                    "psy_tristeza_mean": float(
                        np.clip(rng.normal(0.2 + 0.3 * shift / 0.8 * 0.25, 0.1), 0, 1)
                    ),
                    "psy_risco_suicida_mean": float(
                        np.clip(rng.normal(0.05 + 0.3 * class_index, 0.08), 0, 1)
                    ),
                    "sem_mean_000": float(rng.normal(shift, 0.5)),
                    "sem_mean_001": float(rng.normal(-shift, 0.5)),
                    "sem_std_000": float(abs(rng.normal(0.5, 0.1))),
                }
            )

    return pl.DataFrame(records).sort("user_id")


@pytest.fixture
def splits(feature_matrix: pl.DataFrame) -> pl.DataFrame:
    """Tabela de partições sintética, estratificada e sem vazamento.

    Returns
    -------
    pl.DataFrame
        Conjunto conforme :class:`schemas.users.SplitSchema`.
    """
    from config.settings import CrossValidationSection, SplitSection
    from data.splitter import build_split_table

    return build_split_table(
        feature_matrix.select(["user_id", "user_label"]),
        SplitSection(test_size=0.25, val_size=0.15),
        CrossValidationSection(n_splits=3),
        RANDOM_SEED,
    )


@pytest.fixture
def predictions(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predições sintéticas de um classificador razoável, mas imperfeito.

    Returns
    -------
    tuple
        ``(y_true, y_pred, y_proba)``.
    """
    n_samples, n_classes = 120, len(CLASS_ORDER)
    y_true = rng.integers(0, n_classes, size=n_samples)

    y_pred = y_true.copy()
    errors = rng.random(n_samples) < 0.25
    y_pred[errors] = rng.integers(0, n_classes, size=int(errors.sum()))

    proba = rng.random((n_samples, n_classes)) * 0.3
    proba[np.arange(n_samples), y_pred] += 1.0
    proba /= proba.sum(axis=1, keepdims=True)

    return y_true, y_pred, proba
