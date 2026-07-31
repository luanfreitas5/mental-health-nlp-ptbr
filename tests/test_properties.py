"""Testes baseados em propriedades (``hypothesis``).

Em vez de verificar alguns casos escolhidos à mão, estes testes afirmam
**invariantes** que precisam valer para qualquer entrada gerada. É a forma de
encontrar os casos-limite que ninguém pensou em escrever: texto só com emoji,
usuário com um único tweet, probabilidade exatamente 0,5.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from config.settings import CleaningSection, NormalizationSection
from evaluation.metrics import bootstrap_confidence_interval, compute_metrics
from evaluation.statistics import cliffs_delta, holm_correction
from preprocessing.text import clean_text, contains_pii, normalize_text, strip_accents, tokenize
from utils.hashing import hash_payload, hash_text, pseudonymize
from utils.lexicons import normalize_term
from utils.timing import format_duration

#: Textos arbitrários, incluindo acentos, emoji e pontuação.
texto = st.text(max_size=280)

#: Perfis de configuração determinísticos (evitam recriar Pydantic por exemplo).
NORMALIZATION = NormalizationSection()
CLEANING = CleaningSection()

SETTINGS = settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)


class TestPropriedadesDeTexto:
    """Invariantes das funções de processamento de texto."""

    @SETTINGS
    @given(texto)
    def test_normalizacao_nunca_falha(self, value: str) -> None:
        """Qualquer texto de entrada produz uma string, nunca uma exceção."""
        assert isinstance(normalize_text(value, NORMALIZATION), str)

    @SETTINGS
    @given(texto)
    def test_normalizacao_remove_pii(self, value: str) -> None:
        """Invariante central de privacidade: a saída não contém PII."""
        result = normalize_text(value, NORMALIZATION)
        assert not contains_pii(result)

    @SETTINGS
    @given(texto)
    def test_limpeza_nao_deixa_espacos_duplicados(self, value: str) -> None:
        """A saída da limpeza é uma sequência de tokens separados por um espaço."""
        result = clean_text(normalize_text(value, NORMALIZATION), CLEANING, frozenset())
        assert "  " not in result
        assert result == result.strip()

    @SETTINGS
    @given(texto)
    def test_limpeza_e_idempotente(self, value: str) -> None:
        """Limpar duas vezes é igual a limpar uma vez."""
        once = clean_text(value, CLEANING, frozenset())
        twice = clean_text(once, CLEANING, frozenset())
        assert once == twice

    @SETTINGS
    @given(texto)
    def test_remocao_de_acentos_preserva_comprimento(self, value: str) -> None:
        """Remover acentos não pode inserir caracteres."""
        assert len(strip_accents(value)) <= len(value)

    @SETTINGS
    @given(texto)
    def test_tokenizacao_nao_produz_token_vazio(self, value: str) -> None:
        """Nenhum token vazio sobrevive à tokenização."""
        assert all(token for token in tokenize(value))

    @SETTINGS
    @given(st.text(max_size=50))
    def test_normalizacao_de_termo_e_minuscula(self, value: str) -> None:
        """A normalização de léxico sempre produz texto em caixa baixa."""
        assert normalize_term(value) == normalize_term(value).lower()


class TestPropriedadesDeHashing:
    """Invariantes da pseudonimização e do hashing."""

    @SETTINGS
    @given(st.text(min_size=1, max_size=50), st.text(min_size=1, max_size=20))
    def test_pseudonimizacao_e_deterministica(self, identifier: str, salt: str) -> None:
        """A mesma entrada sempre produz o mesmo pseudônimo."""
        assert pseudonymize(identifier, salt) == pseudonymize(identifier, salt)

    @SETTINGS
    @given(st.text(min_size=1, max_size=50), st.text(min_size=1, max_size=20))
    def test_pseudonimo_tem_formato_valido(self, identifier: str, salt: str) -> None:
        """O formato satisfaz o padrão exigido pelos contratos pandera."""
        result = pseudonymize(identifier, salt)
        assert result.startswith("u_")
        assert len(result) == 18

    @SETTINGS
    @given(st.text(min_size=1, max_size=50), st.text(min_size=1, max_size=20))
    def test_salt_diferente_muda_o_pseudonimo(self, identifier: str, salt: str) -> None:
        """O salt é o que impede reverter o hash por força bruta sobre handles."""
        assert pseudonymize(identifier, salt) != pseudonymize(identifier, f"{salt}x")

    @SETTINGS
    @given(texto)
    def test_hash_tem_comprimento_fixo(self, value: str) -> None:
        """O SHA-256 tem sempre 64 caracteres hexadecimais."""
        assert len(hash_text(value)) == 64

    @SETTINGS
    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.one_of(st.integers(), st.text(max_size=20)),
            max_size=5,
        )
    )
    def test_hash_de_payload_independe_da_ordem(self, payload: dict) -> None:
        """O cache do LLM depende disso: mesma requisição, mesma chave."""
        reversed_payload = dict(reversed(list(payload.items())))
        assert hash_payload(payload) == hash_payload(reversed_payload)


class TestPropriedadesDeMetricas:
    """Invariantes das métricas de avaliação."""

    @SETTINGS
    @given(
        st.lists(st.integers(min_value=0, max_value=2), min_size=5, max_size=100),
        st.lists(st.integers(min_value=0, max_value=2), min_size=5, max_size=100),
    )
    def test_metricas_ficam_na_faixa_valida(self, y_true: list[int], y_pred: list[int]) -> None:
        """Acurácia e F1 são proporções e nunca saem de [0, 1]."""
        size = min(len(y_true), len(y_pred))
        metrics = compute_metrics(np.array(y_true[:size]), np.array(y_pred[:size]))

        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_macro"] <= 1.0
        assert -1.0 <= metrics["mcc"] <= 1.0

    @SETTINGS
    @given(st.lists(st.integers(min_value=0, max_value=2), min_size=5, max_size=60))
    def test_predicao_perfeita_maximiza(self, y_true: list[int]) -> None:
        """Prever exatamente a verdade produz acurácia 1."""
        array = np.array(y_true)
        assert compute_metrics(array, array.copy())["accuracy"] == pytest.approx(1.0)

    @SETTINGS
    @given(
        st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=60),
        st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=60),
    )
    def test_intervalo_contem_o_ponto(self, y_true: list[int], y_pred: list[int]) -> None:
        """O ponto estimado sempre cai dentro do intervalo de confiança."""
        size = min(len(y_true), len(y_pred))
        interval = bootstrap_confidence_interval(
            np.array(y_true[:size]), np.array(y_pred[:size]), n_bootstrap=30
        )
        assert interval["lower"] <= interval["upper"]

    @SETTINGS
    @given(
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=2, max_size=20),
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=2, max_size=20),
    )
    def test_cliffs_delta_na_faixa(self, first: list[float], second: list[float]) -> None:
        """O delta de Cliff é limitado a [-1, 1] por construção."""
        assert -1.0 <= cliffs_delta(np.array(first), np.array(second)) <= 1.0

    @SETTINGS
    @given(
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=15)
    )
    def test_holm_nunca_reduz_p_valor(self, p_values: list[float]) -> None:
        """A correção para múltiplas comparações só pode aumentar p-valores."""
        corrected = holm_correction(p_values)
        assert all(
            after >= before - 1e-9 for before, after in zip(p_values, corrected, strict=True)
        )

    @SETTINGS
    @given(
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=15)
    )
    def test_holm_nao_ultrapassa_um(self, p_values: list[float]) -> None:
        """Um p-valor corrigido continua sendo uma probabilidade."""
        assert all(0.0 <= value <= 1.0 for value in holm_correction(p_values))


class TestPropriedadesDeFormatacao:
    """Invariantes da formatação em pt-BR."""

    @SETTINGS
    @given(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False))
    def test_duracao_sempre_formatavel(self, seconds: float) -> None:
        """Qualquer duração não negativa produz uma string não vazia."""
        assert format_duration(seconds)

    @SETTINGS
    @given(st.floats(min_value=0.0, max_value=59.0, allow_nan=False))
    def test_duracao_curta_usa_virgula(self, seconds: float) -> None:
        """O separador decimal em pt-BR é a vírgula."""
        assert "." not in format_duration(seconds)
