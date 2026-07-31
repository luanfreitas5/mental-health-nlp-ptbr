"""Testes de normalização, limpeza e filtros dos tweets."""

from __future__ import annotations

import polars as pl
import pytest

from config.settings import (
    CleaningSection,
    CollectionFiltersSection,
    DeduplicationSection,
    NormalizationSection,
    PreprocessingFiltersSection,
    TokenizationSection,
)
from preprocessing.cleaning import (
    deduplicate,
    filter_after_cleaning,
    filter_automated_accounts,
    filter_by_quality,
    filter_users_by_activity,
)
from preprocessing.text import (
    clean_text,
    contains_pii,
    extract_hashtags,
    normalize_text,
    strip_accents,
    tokenize,
)
from preprocessing.tokenization import Tokenizer


@pytest.fixture
def normalization() -> NormalizationSection:
    """Configuração padrão de normalização."""
    return NormalizationSection()


@pytest.fixture
def cleaning() -> CleaningSection:
    """Configuração padrão de limpeza."""
    return CleaningSection()


class TestNormalizacao:
    """Testes da normalização que preserva a semântica."""

    def test_remove_url(self, normalization: NormalizationSection) -> None:
        """URLs viram placeholder — são PII potencial e ruído semântico."""
        assert "http" not in normalize_text("veja https://exemplo.com/x", normalization)

    def test_substitui_mencao(self, normalization: NormalizationSection) -> None:
        """Menções são anonimizadas preservando a estrutura da frase."""
        assert normalize_text("oi @fulano", normalization) == "oi @user"

    def test_remove_email(self, normalization: NormalizationSection) -> None:
        """E-mails são substituídos antes das menções (ordem importa)."""
        result = normalize_text("contato a.b@dominio.com", normalization)
        assert "@dominio.com" not in result
        assert "EMAIL" in result

    def test_desempacota_hashtag(self, normalization: NormalizationSection) -> None:
        """A hashtag carrega conteúdo: o termo é preservado, o '#' não."""
        assert normalize_text("dia difícil #desabafo", normalization) == "dia difícil desabafo"

    def test_colapsa_repeticoes(self, normalization: NormalizationSection) -> None:
        """A ênfase é preservada, mas a esparsidade é reduzida."""
        assert normalize_text("muitooooo triste", normalization) == "muitoo triste"

    def test_preserva_acentos_e_caixa(self, normalization: NormalizationSection) -> None:
        """Transformers dependem de caixa e acento: nada disso é removido aqui."""
        assert normalize_text("Não Estou Bem", normalization) == "Não Estou Bem"

    def test_remove_marcador_de_retweet(self, normalization: NormalizationSection) -> None:
        """O prefixo 'RT @fulano:' é metadado, não conteúdo."""
        assert normalize_text("RT @alguem: texto original", normalization) == "texto original"

    def test_colapsa_espacos(self, normalization: NormalizationSection) -> None:
        """Quebras de linha e espaços múltiplos viram espaço simples."""
        assert normalize_text("a\n\n  b", normalization) == "a b"

    def test_texto_vazio(self, normalization: NormalizationSection) -> None:
        """Entrada vazia devolve string vazia, sem exceção."""
        assert normalize_text("", normalization) == ""


class TestLimpeza:
    """Testes da limpeza agressiva."""

    def test_aplica_minusculas_e_remove_pontuacao(self, cleaning: CleaningSection) -> None:
        """Caixa e pontuação são descartadas para o TF-IDF."""
        assert clean_text("Olá, Mundo!", cleaning, frozenset()) == "olá mundo"

    def test_remove_stopwords(self, cleaning: CleaningSection) -> None:
        """Stopwords configuradas são removidas."""
        assert clean_text("eu estou bem", cleaning, frozenset({"estou"})) == "eu bem"

    def test_preserva_pronomes_da_whitelist(self) -> None:
        """Pronomes de 1ª pessoa são feature central e não podem ser removidos."""
        config = CleaningSection(stopwords_whitelist=["eu", "não"])
        result = clean_text("eu não estou bem", config, frozenset({"eu", "não", "estou"}))

        assert "eu" in result
        assert "não" in result
        assert "estou" not in result

    def test_remove_tokens_curtos(self) -> None:
        """Tokens abaixo do comprimento mínimo são descartados."""
        config = CleaningSection(min_token_length=3)
        assert clean_text("eu vou ao mar", config, frozenset()) == "vou mar"

    def test_stopword_sem_acento_tambem_e_removida(self, cleaning: CleaningSection) -> None:
        """A comparação com stopwords é robusta a variação ortográfica."""
        assert clean_text("voce esta bem", cleaning, frozenset({"você", "esta"})) == "bem"

    def test_texto_vazio(self, cleaning: CleaningSection) -> None:
        """Entrada vazia devolve string vazia."""
        assert clean_text("", cleaning, frozenset()) == ""


class TestFuncoesAuxiliares:
    """Testes das funções de apoio ao processamento de texto."""

    def test_strip_accents(self) -> None:
        """Acentos são removidos preservando as letras."""
        assert strip_accents("ação") == "acao"

    def test_tokenize_ignora_pontuacao(self) -> None:
        """O fallback por regex extrai apenas tokens alfabéticos."""
        assert tokenize("não estou bem, hoje!") == ["não", "estou", "bem", "hoje"]

    def test_extract_hashtags(self) -> None:
        """As hashtags são extraídas em minúsculas, sem o '#'."""
        assert extract_hashtags("dia #Difícil #saudemental") == ["difícil", "saudemental"]

    def test_contains_pii_detecta_email(self) -> None:
        """A salvaguarda antes do LLM detecta e-mail residual."""
        assert contains_pii("meu email é a@b.com")

    def test_contains_pii_ignora_texto_limpo(self) -> None:
        """Texto já higienizado não gera falso positivo."""
        assert not contains_pii("hoje foi um dia difícil")


class TestTokenizador:
    """Testes do tokenizador com fallback."""

    def test_fallback_por_regex(self) -> None:
        """Sem spaCy, a tokenização por regex mantém o pipeline funcional."""
        tokenizer = Tokenizer(TokenizationSection(backend="regex"))
        assert tokenizer.tokenize("não estou bem") == ["não", "estou", "bem"]

    def test_lote_preserva_ordem(self) -> None:
        """A tokenização em lote devolve os resultados na ordem da entrada."""
        tokenizer = Tokenizer(TokenizationSection(backend="regex"))
        assert tokenizer.tokenize_batch(["a b", "c"]) == [["a", "b"], ["c"]]

    def test_modelo_inexistente_nao_derruba(self) -> None:
        """Um modelo spaCy ausente degrada para o fallback, sem exceção."""
        tokenizer = Tokenizer(TokenizationSection(spacy_model="modelo_inexistente_xyz"))
        assert tokenizer.tokenize("teste") == ["teste"]


class TestDeduplicacao:
    """Testes da remoção de duplicatas."""

    def test_remove_texto_repetido_do_mesmo_usuario(self) -> None:
        """Repost do próprio conteúdo é ruído."""
        frame = pl.DataFrame(
            {
                "user_id": ["u_a", "u_a"],
                "tweet_id": ["t1", "t2"],
                "text": ["mesmo texto", "mesmo texto"],
            }
        )
        assert deduplicate(frame, DeduplicationSection()).height == 1

    def test_preserva_texto_repetido_entre_usuarios(self) -> None:
        """Texto igual entre pessoas é sinal (correntes, letras), não ruído."""
        frame = pl.DataFrame(
            {
                "user_id": ["u_a", "u_b"],
                "tweet_id": ["t1", "t2"],
                "text": ["mesmo texto", "mesmo texto"],
            }
        )
        assert deduplicate(frame, DeduplicationSection()).height == 2

    def test_remove_identificador_duplicado(self) -> None:
        """O mesmo tweet coletado duas vezes aparece uma só."""
        frame = pl.DataFrame(
            {"user_id": ["u_a", "u_a"], "tweet_id": ["t1", "t1"], "text": ["a", "b"]}
        )
        assert deduplicate(frame, DeduplicationSection()).height == 1


class TestFiltros:
    """Testes dos filtros de qualidade e de atividade."""

    def test_filtra_por_comprimento(self) -> None:
        """Textos curtos demais não sustentam nenhuma feature."""
        frame = pl.DataFrame({"text": ["oi", "um texto suficientemente longo aqui"]})
        assert filter_by_quality(frame, CollectionFiltersSection()).height == 1

    def test_filtra_por_idioma(self) -> None:
        """Tweets fora do idioma alvo são descartados."""
        frame = pl.DataFrame(
            {
                "text": ["um texto suficientemente longo", "a text long enough here"],
                "language": ["pt", "en"],
            }
        )
        assert filter_by_quality(frame, CollectionFiltersSection()).height == 1

    def test_remove_vazios_apos_limpeza(self) -> None:
        """Tweets só com emoji/menção viram texto vazio e são descartados."""
        frame = pl.DataFrame({"text_clean": ["", "um dois três"]})
        assert filter_after_cleaning(frame, PreprocessingFiltersSection()).height == 1

    def test_filtra_por_minimo_de_tokens(self) -> None:
        """Abaixo do mínimo de tokens, o tweet não contribui."""
        frame = pl.DataFrame({"text_clean": ["um dois", "um dois três quatro"]})
        config = PreprocessingFiltersSection(min_tokens_per_tweet=3)
        assert filter_after_cleaning(frame, config).height == 1

    def test_remove_conta_automatizada(self, raw_tweets: pl.DataFrame) -> None:
        """Volume incompatível com uso pessoal indica bot ou divulgação."""
        result = filter_automated_accounts(raw_tweets, max_tweets_per_day=1000)
        assert result.height == raw_tweets.height

        agressivo = filter_automated_accounts(raw_tweets, max_tweets_per_day=0)
        assert agressivo.height < raw_tweets.height

    def test_filtra_usuarios_com_pouca_atividade(self, raw_tweets: pl.DataFrame) -> None:
        """Histórico curto não sustenta as features longitudinais."""
        result = filter_users_by_activity(raw_tweets, min_tweets=10_000, min_active_days=1)
        assert result.is_empty()

    def test_mantem_usuarios_ativos(self, raw_tweets: pl.DataFrame) -> None:
        """Usuários que atendem aos critérios permanecem integralmente."""
        result = filter_users_by_activity(raw_tweets, min_tweets=5, min_active_days=1)
        assert result["user_id"].n_unique() == raw_tweets["user_id"].n_unique()
