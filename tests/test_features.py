"""Testes da extração de atributos e da agregação tweet -> usuário."""

from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl
import pytest

from config.settings import (
    NgramsSection,
    PsychologicalSection,
    TemporalSection,
)
from constants.columns import (
    BEHAVIORAL_PREFIX,
    EMOTIONAL_PREFIX,
    LINGUISTIC_PREFIX,
    PSYCHOLOGICAL_PREFIX,
    TEMPORAL_PREFIX,
)
from features.behavioral import compute_engagement, compute_interaction_ratios
from features.builder import build_profile_columns, handle_missing_values, select_groups
from features.emotional import build_aggregations, compute_sentiment_distribution
from features.linguistic import (
    compute_lexical_diversity,
    compute_lexicon_ratios,
    compute_pronoun_usage,
    compute_text_length,
)
from features.ngrams import UserNgramVectorizer, build_user_documents
from features.psychological import build_psychological_features
from features.semantic import aggregate_embeddings
from features.temporal import compute_circadian, compute_volume
from schemas.features import list_feature_columns


class TestLinguisticos:
    """Testes dos atributos linguísticos."""

    def test_razoes_lexicais_por_usuario(self, clean_tweets: pl.DataFrame) -> None:
        """Uma linha por usuário, com as razões dos léxicos pedidos."""
        result = compute_lexicon_ratios(clean_tweets, ["death", "loneliness"])

        assert result.height == clean_tweets["user_id"].n_unique()
        assert f"{LINGUISTIC_PREFIX}death_ratio" in result.columns

    def test_razoes_ficam_entre_zero_e_um(self, clean_tweets: pl.DataFrame) -> None:
        """A razão é uma proporção de tweets e não pode sair de [0, 1]."""
        result = compute_lexicon_ratios(clean_tweets, ["loneliness"])
        values = result[f"{LINGUISTIC_PREFIX}loneliness_ratio"].to_numpy()

        assert np.all((values >= 0) & (values <= 1))

    def test_lexico_inexistente_e_ignorado(self, clean_tweets: pl.DataFrame) -> None:
        """Um léxico ausente degrada as features, mas não derruba a etapa."""
        result = compute_lexicon_ratios(clean_tweets, ["lexico_que_nao_existe"])
        assert result.height > 0

    def test_diversidade_lexical(self) -> None:
        """A type-token ratio é calculada sobre o histórico concatenado."""
        frame = pl.DataFrame({"user_id": ["u_a"], "text_clean": ["oi oi tudo bem"]})
        result = compute_lexical_diversity(frame)

        assert result[f"{LINGUISTIC_PREFIX}ttr"][0] == pytest.approx(0.75)
        assert result[f"{LINGUISTIC_PREFIX}vocabulary_size"][0] == 3

    def test_guiraud_corrige_por_comprimento(self) -> None:
        """O índice de Guiraud é menos sensível ao tamanho do texto que a TTR."""
        curto = pl.DataFrame({"user_id": ["u_a"], "text_clean": ["um dois três"]})
        longo = pl.DataFrame(
            {"user_id": ["u_b"], "text_clean": [" ".join(f"palavra{i}" for i in range(300))]}
        )

        ttr_curto = compute_lexical_diversity(curto)[f"{LINGUISTIC_PREFIX}ttr"][0]
        ttr_longo = compute_lexical_diversity(longo)[f"{LINGUISTIC_PREFIX}ttr"][0]
        guiraud_longo = compute_lexical_diversity(longo)[f"{LINGUISTIC_PREFIX}guiraud"][0]

        assert ttr_curto == pytest.approx(ttr_longo)
        assert guiraud_longo > 1.0

    def test_pronomes_de_primeira_pessoa(self) -> None:
        """O uso de 1ª pessoa do singular é detectado e normalizado."""
        frame = pl.DataFrame({"user_id": ["u_a"], "text_clean": ["eu me sinto mal"]})
        result = compute_pronoun_usage(frame)

        assert result[f"{LINGUISTIC_PREFIX}pronoun_first_singular"][0] > 0

    def test_razao_eu_nos_e_finita_sem_plural(self) -> None:
        """Sem 1ª pessoa do plural, a razão precisa continuar finita."""
        frame = pl.DataFrame({"user_id": ["u_a"], "text_clean": ["eu eu eu"]})
        value = compute_pronoun_usage(frame)[f"{LINGUISTIC_PREFIX}pronoun_i_we_ratio"][0]

        assert np.isfinite(value)

    def test_comprimento_dos_textos(self, clean_tweets: pl.DataFrame) -> None:
        """Média e desvio de caracteres e tokens são calculados por usuário."""
        result = compute_text_length(clean_tweets)
        assert f"{LINGUISTIC_PREFIX}chars_mean" in result.columns
        assert (result[f"{LINGUISTIC_PREFIX}chars_mean"] > 0).all()


class TestEmocionais:
    """Testes dos atributos emocionais."""

    def test_distribuicao_de_sentimento(self, labeled_tweets: pl.DataFrame) -> None:
        """As proporções de cada sentimento são calculadas por usuário."""
        result = compute_sentiment_distribution(labeled_tweets)

        assert f"{EMOTIONAL_PREFIX}negativo_ratio" in result.columns
        assert result.height == labeled_tweets["user_id"].n_unique()

    def test_razao_negativo_positivo_e_finita(self, labeled_tweets: pl.DataFrame) -> None:
        """Usuários sem tweets positivos são comuns; a razão não pode virar inf."""
        result = compute_sentiment_distribution(labeled_tweets)
        values = result[f"{EMOTIONAL_PREFIX}negative_positive_ratio"].to_numpy()

        assert np.all(np.isfinite(values))

    def test_agregacoes_desconhecidas_sao_rejeitadas(self) -> None:
        """Uma agregação inexistente no YAML falha com erro claro."""
        with pytest.raises(KeyError, match="não suportadas"):
            build_aggregations("x", ["mediana_ponderada"], "emo_")

    def test_agregacoes_geram_uma_coluna_cada(self) -> None:
        """Cada agregação configurada vira exatamente uma coluna."""
        assert len(build_aggregations("x", ["mean", "std", "max"], "emo_")) == 3


class TestTemporais:
    """Testes dos atributos temporais e circadianos."""

    def test_volume_por_usuario(self, clean_tweets: pl.DataFrame) -> None:
        """As métricas de volume são calculadas por usuário."""
        result = compute_volume(clean_tweets)

        assert f"{TEMPORAL_PREFIX}tweets_per_day" in result.columns
        assert (result[f"{TEMPORAL_PREFIX}tweets_per_day"] > 0).all()

    def test_atividade_noturna_entre_zero_e_um(self, clean_tweets: pl.DataFrame) -> None:
        """A razão de atividade noturna é uma proporção."""
        result = compute_circadian(clean_tweets, TemporalSection())
        values = result[f"{TEMPORAL_PREFIX}night_activity_ratio"].to_numpy()

        assert np.all((values >= 0) & (values <= 1))

    def test_entropia_circadiana_normalizada(self, clean_tweets: pl.DataFrame) -> None:
        """A entropia é normalizada por log(24) e fica em [0, 1]."""
        result = compute_circadian(clean_tweets, TemporalSection())
        values = result[f"{TEMPORAL_PREFIX}circadian_entropy"].to_numpy()

        assert np.all((values >= 0) & (values <= 1.001))


class TestComportamentais:
    """Testes dos atributos comportamentais."""

    def test_engajamento_agregado(self, clean_tweets: pl.DataFrame) -> None:
        """O engajamento recebido é agregado por usuário."""
        result = compute_engagement(clean_tweets, ["mean", "max"])
        assert any(column.startswith(BEHAVIORAL_PREFIX) for column in result.columns)

    def test_transformacao_log_reduz_cauda(self, clean_tweets: pl.DataFrame) -> None:
        """Sem log1p, um tweet viral domina a média do usuário."""
        com_log = compute_engagement(clean_tweets, ["mean"], log_transform=True)
        sem_log = compute_engagement(clean_tweets, ["mean"], log_transform=False)

        coluna = f"{BEHAVIORAL_PREFIX}like_count_mean"
        assert cast(float, com_log[coluna].max()) < cast(float, sem_log[coluna].max())

    def test_razoes_de_interacao(self, clean_tweets: pl.DataFrame) -> None:
        """As razões de resposta e conteúdo original somam corretamente."""
        result = compute_interaction_ratios(clean_tweets)
        values = result[f"{BEHAVIORAL_PREFIX}reply_ratio"].to_numpy()

        assert np.all((values >= 0) & (values <= 1))


class TestSemanticos:
    """Testes da agregação de embeddings."""

    def test_agrega_por_usuario(self) -> None:
        """Vários tweets de um usuário produzem uma única linha."""
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = aggregate_embeddings(embeddings, ["u_a", "u_a"], ["mean"])

        assert result.height == 1
        assert result["sem_mean_000"][0] == pytest.approx(2.0)

    def test_desvio_zero_com_um_unico_tweet(self) -> None:
        """Com um só tweet, o desvio é 0 — nunca NaN."""
        result = aggregate_embeddings(np.array([[1.0, 2.0]]), ["u_a"], ["std"])
        assert result["sem_std_000"][0] == pytest.approx(0.0)

    def test_incompatibilidade_de_tamanho_e_rejeitada(self) -> None:
        """Vetores e identificadores desalinhados corromperiam a matriz."""
        with pytest.raises(ValueError, match="Incompatibilidade"):
            aggregate_embeddings(np.ones((3, 2)), ["u_a"], ["mean"])

    def test_agregacao_desconhecida_e_rejeitada(self) -> None:
        """Uma agregação não suportada falha com erro claro."""
        with pytest.raises(ValueError, match="não suportada"):
            aggregate_embeddings(np.ones((1, 2)), ["u_a"], ["mediana"])


class TestPsicologicos:
    """Testes da agregação do vetor psicológico."""

    def test_agrega_dimensoes(self) -> None:
        """As dimensões configuradas viram colunas agregadas por usuário."""
        scores = pl.DataFrame(
            {
                "user_id": ["u_a", "u_a"],
                "tristeza": [0.8, 0.6],
                "isolamento": [0.5, 0.5],
                "esperanca": [0.2, 0.4],
                "ansiedade": [0.7, 0.7],
                "risco_suicida": [0.3, 0.5],
            }
        )
        config = PsychologicalSection(
            dimensions=["tristeza", "isolamento", "esperanca", "ansiedade", "risco_suicida"],
            aggregations=["mean", "max"],
        )
        result = build_psychological_features(scores, config)

        assert result[f"{PSYCHOLOGICAL_PREFIX}tristeza_mean"][0] == pytest.approx(0.7)
        assert result[f"{PSYCHOLOGICAL_PREFIX}tristeza_max"][0] == pytest.approx(0.8)

    def test_indice_composto_desconta_esperanca(self) -> None:
        """A esperança é a única dimensão positiva e entra com sinal invertido."""
        scores = pl.DataFrame(
            {
                "user_id": ["u_a"],
                "tristeza": [0.8],
                "isolamento": [0.8],
                "esperanca": [0.9],
                "ansiedade": [0.8],
                "risco_suicida": [0.8],
            }
        )
        config = PsychologicalSection(
            dimensions=["tristeza", "isolamento", "esperanca", "ansiedade", "risco_suicida"],
            aggregations=["mean"],
        )
        result = build_psychological_features(scores, config)

        assert result[f"{PSYCHOLOGICAL_PREFIX}risk_index"][0] < 0.8

    def test_scores_vazios_nao_derrubam(self) -> None:
        """Sem a etapa 'psych', o grupo fica vazio mas o pipeline segue."""
        config = PsychologicalSection(dimensions=["tristeza"], aggregations=["mean"])
        assert build_psychological_features(pl.DataFrame(), config).height == 0

    def test_dimensao_ausente_e_rejeitada(self) -> None:
        """Uma dimensão configurada mas ausente nos scores é erro de contrato."""
        scores = pl.DataFrame({"user_id": ["u_a"], "tristeza": [0.5]})
        config = PsychologicalSection(dimensions=["tristeza", "inexistente"], aggregations=["mean"])

        with pytest.raises(KeyError, match="ausentes"):
            build_psychological_features(scores, config)


class TestNgrams:
    """Testes da vetorização de n-grams."""

    def test_documento_por_usuario(self) -> None:
        """Os tweets de um usuário viram um único documento."""
        frame = pl.DataFrame({"user_id": ["u_a", "u_a"], "text_clean": ["oi", "tudo bem"]})
        assert build_user_documents(frame)["document"][0] == "oi tudo bem"

    def test_vetorizador_ajusta_e_transforma(self) -> None:
        """O TF-IDF é ajustado apenas nos documentos vistos no fit."""
        vectorizer = UserNgramVectorizer(NgramsSection(min_df=1))
        matrix = vectorizer.fit_transform(["eu não estou bem", "dia bom hoje"])

        assert matrix.shape[0] == 2
        assert matrix.shape[1] > 0

    def test_transform_antes_de_fit_falha(self) -> None:
        """Transformar sem ajustar é erro de uso e falha alto."""
        with pytest.raises(RuntimeError, match="antes de fit"):
            UserNgramVectorizer(NgramsSection()).transform(["texto"])

    def test_nomes_de_atributo_seguem_o_prefixo(self) -> None:
        """Os n-grams entram no grupo linguístico pela convenção de prefixo."""
        vectorizer = UserNgramVectorizer(NgramsSection(min_df=1))
        vectorizer.fit(["um dois três", "dois três quatro"])

        assert all(name.startswith("ling_ngram_") for name in vectorizer.get_feature_names_out())


class TestConstrucaoDaMatriz:
    """Testes da montagem final da matriz de atributos."""

    def test_colunas_de_perfil(self, clean_tweets: pl.DataFrame) -> None:
        """As colunas descritivas do perfil são calculadas por usuário."""
        result = build_profile_columns(clean_tweets)

        assert {"n_tweets", "active_days", "span_days"}.issubset(set(result.columns))
        assert (result["n_tweets"] > 0).all()

    def test_indicadores_de_ausencia(self, config) -> None:
        """A ausência não aleatória vira feature explícita antes da imputação."""
        frame = pl.DataFrame({"user_id": ["u_a", "u_b"], "temp_polarity_slope": [1.0, None]})
        result = handle_missing_values(frame, config.features)

        assert "temp_polarity_slope_is_missing" in result.columns
        assert result["temp_polarity_slope"].null_count() == 0

    def test_selecao_por_grupo(self, feature_matrix: pl.DataFrame) -> None:
        """A seleção por prefixo é o que sustenta o Ablation Study."""
        result = select_groups(feature_matrix, ["emotional"])
        features = list_feature_columns(result)

        assert features
        assert all(name.startswith(EMOTIONAL_PREFIX) for name in features)

    def test_selecao_preserva_colunas_chave(self, feature_matrix: pl.DataFrame) -> None:
        """Identificador e rótulo sobrevivem a qualquer seleção de grupo."""
        result = select_groups(feature_matrix, ["temporal"])
        assert {"user_id", "user_label"}.issubset(set(result.columns))

    def test_grupo_desconhecido_e_rejeitado(self, feature_matrix: pl.DataFrame) -> None:
        """Um grupo inexistente é erro de configuração, não seleção vazia."""
        with pytest.raises(KeyError, match="desconhecidos"):
            list_feature_columns(feature_matrix, ["grupo_inexistente"])
