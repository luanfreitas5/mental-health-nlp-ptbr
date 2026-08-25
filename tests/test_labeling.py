"""Testes da rotulação por supervisão fraca e das salvaguardas do LLM."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import polars as pl
import pytest

from config.settings import EmotionSection, SentimentSection, UserLabelingSection
from constants.labels import UserLabel
from exceptions.model import LLMResponseError
from labeling.emotion import EmotionLabeler
from labeling.llm import PsychologicalVector, UserClassification, extract_json_object
from labeling.prompt import build_psychological_prompt, format_tweets
from labeling.sentiment import SentimentLabeler, _apply_dynamic_quantization, _resolve_fp16
from labeling.validation import (
    apply_manual_labels,
    compute_agreement,
    drop_undecided,
    load_manual_labels,
    sample_for_manual_review,
)
from labeling.weak_supervision import (
    LabelVote,
    assign_user_labels,
    compute_lexical_evidence,
    compute_temporal_persistence,
    label_from_lexical_evidence,
    resolve_consensus,
)


def _fake_transformers_module(pipeline_fn: Any) -> types.ModuleType:
    """Cria um módulo ``transformers`` falso cujo ``pipeline`` é ``pipeline_fn``."""
    module = types.ModuleType("transformers")
    module.pipeline = pipeline_fn  # type: ignore[attr-defined]
    return module


def _fake_torch_module() -> tuple[types.ModuleType, list[dict[str, Any]]]:
    """Cria um módulo ``torch`` falso, com CUDA sempre disponível.

    Retorna também a lista de chamadas a ``quantize_dynamic``, para as
    asserções de :class:`TestQuantizacaoDinamica` e
    :class:`TestCarregamentoComPrecisaoReduzida`.
    """
    module = types.ModuleType("torch")
    module.float16 = "float16-sentinel"  # type: ignore[attr-defined]
    module.qint8 = "qint8-sentinel"  # type: ignore[attr-defined]
    module.cuda = SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    module.nn = SimpleNamespace(Linear=object())  # type: ignore[attr-defined]

    quantize_calls: list[dict[str, Any]] = []

    def quantize_dynamic(model: Any, layers: Any, dtype: Any) -> str:
        quantize_calls.append({"model": model, "layers": layers, "dtype": dtype})
        return f"quantized({model})"

    module.quantization = SimpleNamespace(quantize_dynamic=quantize_dynamic)  # type: ignore[attr-defined]
    return module, quantize_calls


@pytest.fixture
def labeling_config(config) -> UserLabelingSection:
    """Configuração real de rotulação por usuário."""
    return config.labeling.user_labeling


class TestEvidenciaLexical:
    """Testes da evidência léxica agregada por usuário."""

    def test_uma_linha_por_usuario(self, labeled_tweets: pl.DataFrame) -> None:
        """A evidência é agregada no nível do usuário."""
        result = compute_lexical_evidence(labeled_tweets)
        assert result.height == labeled_tweets["user_id"].n_unique()

    def test_inclui_razao_de_negatividade(self, labeled_tweets: pl.DataFrame) -> None:
        """Com sentimento disponível, a razão de negatividade é calculada."""
        assert "negative_ratio" in compute_lexical_evidence(labeled_tweets).columns

    def test_classe_de_risco_tem_mais_evidencia(self, labeled_tweets: pl.DataFrame) -> None:
        """Os textos sintéticos de risco têm densidade léxica maior que o controle."""
        evidence = compute_lexical_evidence(labeled_tweets).join(
            labeled_tweets.group_by("user_id").agg(pl.col("source_group").first()),
            on="user_id",
        )

        controle = evidence.filter(pl.col("source_group") == "controle")
        risco = evidence.filter(pl.col("source_group") == "ideacao_suicida")

        risco_mean = cast(float, risco["death_ratio"].mean())
        controle_mean = cast(float, controle["death_ratio"].mean())
        assert risco_mean > controle_mean


class TestPersistenciaTemporal:
    """Testes do critério de persistência temporal."""

    def test_calcula_janelas_com_sinal(
        self, labeled_tweets: pl.DataFrame, labeling_config: UserLabelingSection
    ) -> None:
        """O número de janelas distintas com sinal é contado por usuário."""
        result = compute_temporal_persistence(labeled_tweets, labeling_config)

        assert "windows_with_signal" in result.columns
        assert "has_persistence" in result.columns

    def test_amplitude_nao_negativa(
        self, labeled_tweets: pl.DataFrame, labeling_config: UserLabelingSection
    ) -> None:
        """A amplitude do histórico é sempre não negativa."""
        result = compute_temporal_persistence(labeled_tweets, labeling_config)
        assert (result["span_days"] >= 0).all()


class TestConsenso:
    """Testes da combinação ponderada das fontes de rotulação."""

    def test_maioria_ponderada_vence(self, labeling_config: UserLabelingSection) -> None:
        """A classe com maior peso acumulado é escolhida."""
        votes = [
            LabelVote("a", str(UserLabel.DEPRESSAO), 0.7),
            LabelVote("b", str(UserLabel.CONTROLE), 0.3),
        ]
        label, agreement = resolve_consensus(votes, labeling_config)

        assert label == str(UserLabel.DEPRESSAO)
        assert agreement == pytest.approx(0.7)

    def test_consenso_baixo_vira_indefinido(self, labeling_config: UserLabelingSection) -> None:
        """Sem concordância mínima, o usuário é marcado como indefinido."""
        votes = [
            LabelVote("a", str(UserLabel.DEPRESSAO), 0.34),
            LabelVote("b", str(UserLabel.CONTROLE), 0.33),
            LabelVote("c", str(UserLabel.IDEACAO_SUICIDA), 0.33),
        ]
        label, _ = resolve_consensus(votes, labeling_config)

        assert label == str(UserLabel.INDEFINIDO)

    def test_empate_resolvido_por_severidade(self, labeling_config: UserLabelingSection) -> None:
        """Em empate, prevalece a classe mais grave — não a ordem de inserção."""
        votes = [
            LabelVote("a", str(UserLabel.DEPRESSAO), 0.5),
            LabelVote("b", str(UserLabel.IDEACAO_SUICIDA), 0.5),
        ]
        label, _ = resolve_consensus(votes, labeling_config)

        assert label == str(UserLabel.IDEACAO_SUICIDA)

    def test_sem_votos_retorna_indefinido(self, labeling_config: UserLabelingSection) -> None:
        """Sem nenhuma fonte disponível, o rótulo é indefinido."""
        assert resolve_consensus([], labeling_config)[0] == str(UserLabel.INDEFINIDO)


class TestRegrasLexicais:
    """Testes da derivação de classe a partir da evidência léxica."""

    def test_evidencia_de_morte_indica_ideacao(self, labeling_config: UserLabelingSection) -> None:
        """Densidade alta de termos de morte aponta ideação suicida."""
        row = {"death_ratio": 0.9, "death_hits": 100}
        assert label_from_lexical_evidence(row, labeling_config) == str(UserLabel.IDEACAO_SUICIDA)

    def test_negatividade_isolada_nao_basta(self, labeling_config: UserLabelingSection) -> None:
        """Negatividade sem marcador específico descreve irritação, não depressão."""
        row = {"negative_ratio": 0.95, "hopelessness_ratio": 0.0, "loneliness_ratio": 0.0}
        assert label_from_lexical_evidence(row, labeling_config) == str(UserLabel.CONTROLE)

    def test_negatividade_com_desesperanca_indica_depressao(
        self, labeling_config: UserLabelingSection
    ) -> None:
        """Negatividade acompanhada de desesperança aponta depressão."""
        row = {"negative_ratio": 0.95, "hopelessness_ratio": 0.5, "loneliness_ratio": 0.0}
        assert label_from_lexical_evidence(row, labeling_config) == str(UserLabel.DEPRESSAO)

    def test_sem_evidencia_e_controle(self, labeling_config: UserLabelingSection) -> None:
        """Ausência de evidência resulta em controle."""
        assert label_from_lexical_evidence({}, labeling_config) == str(UserLabel.CONTROLE)


class TestAtribuicaoDeRotulos:
    """Testes da rotulação completa por usuário."""

    def test_uma_linha_por_usuario(
        self, labeled_tweets: pl.DataFrame, labeling_config: UserLabelingSection
    ) -> None:
        """Cada usuário recebe exatamente um rótulo."""
        result = assign_user_labels(labeled_tweets, labeling_config)
        assert result.height == labeled_tweets["user_id"].n_unique()

    def test_colunas_do_contrato(
        self, labeled_tweets: pl.DataFrame, labeling_config: UserLabelingSection
    ) -> None:
        """A saída satisfaz as colunas exigidas pelo contrato de rótulos."""
        result = assign_user_labels(labeled_tweets, labeling_config)

        assert {"user_id", "user_label", "label_agreement"}.issubset(set(result.columns))

    def test_concordancia_entre_zero_e_um(
        self, labeled_tweets: pl.DataFrame, labeling_config: UserLabelingSection
    ) -> None:
        """A concordância é uma fração do peso total."""
        result = assign_user_labels(labeled_tweets, labeling_config)
        values = result["label_agreement"].to_numpy()

        assert all(0.0 <= value <= 1.0 for value in values)


class TestValidacaoDeRotulos:
    """Testes da revisão manual e do descarte de indefinidos."""

    def test_amostra_estratificada(self, user_labels: pl.DataFrame) -> None:
        """A amostra cobre todas as classes, não só a majoritária."""
        sample = sample_for_manual_review(user_labels, sample_size=6, random_seed=42)
        assert sample["user_label"].n_unique() == user_labels["user_label"].n_unique()

    def test_amostra_tem_coluna_para_o_revisor(self, user_labels: pl.DataFrame) -> None:
        """A coluna de rótulo manual vem vazia, para preenchimento humano."""
        sample = sample_for_manual_review(user_labels, sample_size=3, random_seed=42)
        assert sample["manual_label"].null_count() == sample.height

    def test_rotulo_manual_sobrepoe_o_automatico(self, user_labels: pl.DataFrame) -> None:
        """A revisão humana é a fonte de maior qualidade e sempre vence."""
        target = user_labels["user_id"][0]
        manual = pl.DataFrame({"user_id": [target], "manual_label": ["ideacao_suicida"]})

        result = apply_manual_labels(user_labels, manual)
        assert result.filter(pl.col("user_id") == target)["user_label"][0] == "ideacao_suicida"

    def test_sem_rotulos_manuais_nada_muda(self, user_labels: pl.DataFrame) -> None:
        """Ausência de revisão manual preserva os rótulos automáticos."""
        assert apply_manual_labels(user_labels, None).equals(user_labels)

    def test_concordancia_sem_revisao_e_zero(self, user_labels: pl.DataFrame) -> None:
        """Sem revisão manual, a concordância não é inventada."""
        assert compute_agreement(user_labels)["n_revisados"] == 0.0

    def test_kappa_perfeito_quando_tudo_concorda(self, user_labels: pl.DataFrame) -> None:
        """Concordância total produz kappa 1."""
        frame = user_labels.with_columns(pl.col("user_label").alias("manual_label"))
        assert compute_agreement(frame)["kappa_cohen"] == pytest.approx(1.0)

    def test_descarta_indefinidos(
        self, user_labels: pl.DataFrame, labeling_config: UserLabelingSection
    ) -> None:
        """Rótulos sem consenso são removidos antes do treino."""
        frame = pl.concat(
            [
                user_labels,
                user_labels.head(1).with_columns(
                    pl.lit("u_indefinido_x").alias("user_id"),
                    pl.lit(str(UserLabel.INDEFINIDO)).alias("user_label"),
                ),
            ]
        )
        result = drop_undecided(frame, labeling_config)

        assert str(UserLabel.INDEFINIDO) not in result["user_label"].to_list()

    def test_arquivo_manual_ausente_retorna_none(self, tmp_path: Path) -> None:
        """Sem arquivo de revisão, o pipeline segue com supervisão fraca."""
        assert load_manual_labels(tmp_path / "inexistente.csv") is None


class TestSaidaDoLLM:
    """Testes das salvaguardas sobre a resposta do LLM."""

    def test_extrai_json_de_resposta_verborragica(self) -> None:
        """Modelos menores envolvem o JSON em texto; o objeto é recuperado."""
        assert extract_json_object('Claro! {"a": 1} espero ter ajudado') == {"a": 1}

    def test_resposta_sem_json_e_rejeitada(self) -> None:
        """Sem objeto JSON, a resposta é descartada em vez de adivinhada."""
        with pytest.raises(LLMResponseError, match="não contém"):
            extract_json_object("desculpe, não posso ajudar com isso")

    def test_json_malformado_e_rejeitado(self) -> None:
        """JSON sintaticamente inválido falha com erro tipado."""
        with pytest.raises(LLMResponseError, match="inválido"):
            extract_json_object('{"a": }')

    def test_vetor_psicologico_valida_faixa(self) -> None:
        """Valores fora de [0, 1] indicam resposta inválida do modelo."""
        with pytest.raises(ValueError):
            PsychologicalVector(
                tristeza=1.5, isolamento=0.5, esperanca=0.5, ansiedade=0.5, risco_suicida=0.5
            )

    def test_vetor_psicologico_aceita_valores_validos(self) -> None:
        """Um vetor dentro da faixa é aceito e serializável."""
        vector = PsychologicalVector(
            tristeza=0.9, isolamento=0.8, esperanca=0.1, ansiedade=0.7, risco_suicida=0.6
        )
        assert set(vector.model_dump()) == {
            "tristeza",
            "isolamento",
            "esperanca",
            "ansiedade",
            "risco_suicida",
        }

    def test_classificacao_tem_valores_padrao(self) -> None:
        """Campos opcionais têm padrão, evitando falha por resposta incompleta."""
        result = UserClassification(**json.loads('{"classe": "controle"}'))
        assert result.confianca == pytest.approx(0.5)


class TestPrompts:
    """Testes da construção dos prompts."""

    def test_tweets_numerados(self) -> None:
        """A numeração ajuda o modelo a tratar as publicações como sequência."""
        assert format_tweets(["primeiro", "segundo"], max_chars=1000) == "1. primeiro\n2. segundo"

    def test_trunca_no_orcamento(self) -> None:
        """O orçamento de caracteres é respeitado, truncando pelo fim."""
        result = format_tweets(["a" * 100, "b" * 100], max_chars=50)
        assert "b" * 100 not in result

    def test_prompt_sem_tweets_e_rejeitado(self, config) -> None:
        """Não faz sentido montar prompt psicológico sem conteúdo."""
        from exceptions.model import LLMError

        with pytest.raises(LLMError, match="sem tweets"):
            build_psychological_prompt([], config.llm)

    def test_prompt_com_pii_e_rejeitado(self, config) -> None:
        """A salvaguarda impede enviar PII residual ao LLM."""
        from exceptions.model import LLMError

        with pytest.raises(LLMError, match="PII"):
            build_psychological_prompt(["meu email é a@b.com"], config.llm)


class TestResolucaoDeFp16:
    """``_resolve_fp16`` decide se a inferência usa precisão mista."""

    def test_e_ignorado_fora_de_cuda(self, caplog: pytest.LogCaptureFixture) -> None:
        """fp16 só acelera em Tensor Cores CUDA; em CPU ele é desativado com aviso."""
        with caplog.at_level("WARNING"):
            assert _resolve_fp16(True, "cpu", "modelo-x") is False
        assert "modelo-x" in caplog.text

    def test_e_mantido_em_cuda(self) -> None:
        """Em GPU CUDA, a preferência do usuário é respeitada."""
        assert _resolve_fp16(True, "cuda", "modelo-x") is True

    def test_desativado_nao_gera_aviso(self, caplog: pytest.LogCaptureFixture) -> None:
        """Sem solicitação de fp16, nada precisa ser avisado."""
        with caplog.at_level("WARNING"):
            assert _resolve_fp16(False, "cpu", "modelo-x") is False
        assert caplog.text == ""


class TestQuantizacaoDinamica:
    """``_apply_dynamic_quantization`` aplica int8 dinâmico só em CPU."""

    def test_e_ignorada_em_cuda(self) -> None:
        """Quantização dinâmica não é suportada em CUDA; fp16 cobre esse caso."""
        pipeline = SimpleNamespace(model="modelo-original")
        _apply_dynamic_quantization(pipeline, "cuda", "modelo-x")
        assert pipeline.model == "modelo-original"

    def test_substitui_o_modelo_em_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Em CPU, ``pipeline.model`` passa a apontar para a versão quantizada."""
        fake_torch, calls = _fake_torch_module()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        pipeline = SimpleNamespace(model="modelo-original")

        _apply_dynamic_quantization(pipeline, "cpu", "modelo-x")

        assert pipeline.model == "quantized(modelo-original)"
        assert calls == [
            {
                "model": "modelo-original",
                "layers": {fake_torch.nn.Linear},
                "dtype": fake_torch.qint8,
            }
        ]


class TestCarregamentoComPrecisaoReduzida:
    """``SentimentLabeler``/``EmotionLabeler`` repassam fp16/quantize ao pipeline."""

    def test_sentiment_passa_torch_dtype_quando_fp16_habilitado(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """fp16 habilitado em GPU deve virar ``torch_dtype=torch.float16``."""
        fake_torch, _ = _fake_torch_module()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        captured_kwargs: dict[str, Any] = {}

        def fake_pipeline(**kwargs: Any) -> SimpleNamespace:
            captured_kwargs.update(kwargs)
            return SimpleNamespace(model="modelo-carregado")

        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers_module(fake_pipeline))

        config = SentimentSection(
            model_name="fake/sentiment", label_mapping={}, device="cuda", fp16=True
        )
        labeler = SentimentLabeler(config)

        labeler._load()

        assert captured_kwargs["torch_dtype"] == fake_torch.float16

    def test_sentiment_nao_passa_torch_dtype_quando_fp16_desabilitado(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sem fp16, nenhum ``torch_dtype`` é repassado ao pipeline."""
        captured_kwargs: dict[str, Any] = {}

        def fake_pipeline(**kwargs: Any) -> SimpleNamespace:
            captured_kwargs.update(kwargs)
            return SimpleNamespace(model="modelo-carregado")

        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers_module(fake_pipeline))

        config = SentimentSection(
            model_name="fake/sentiment", label_mapping={}, device="cpu", fp16=False
        )
        labeler = SentimentLabeler(config)

        labeler._load()

        assert "torch_dtype" not in captured_kwargs

    def test_sentiment_aplica_quantizacao_quando_configurado(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """quantize habilitado em CPU quantiza o modelo carregado pelo pipeline."""
        fake_torch, calls = _fake_torch_module()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(
            sys.modules,
            "transformers",
            _fake_transformers_module(lambda **_: SimpleNamespace(model="modelo-original")),
        )

        config = SentimentSection(
            model_name="fake/sentiment", label_mapping={}, device="cpu", quantize=True
        )
        labeler = SentimentLabeler(config)

        pipeline = labeler._load()

        assert pipeline.model == "quantized(modelo-original)"
        assert len(calls) == 1

    def test_emotion_passa_torch_dtype_quando_fp16_habilitado(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """O mesmo contrato de fp16 vale para o classificador de emoções."""
        fake_torch, _ = _fake_torch_module()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        captured_kwargs: dict[str, Any] = {}

        def fake_pipeline(**kwargs: Any) -> SimpleNamespace:
            captured_kwargs.update(kwargs)
            return SimpleNamespace(model="modelo-carregado")

        monkeypatch.setitem(sys.modules, "transformers", _fake_transformers_module(fake_pipeline))

        config = EmotionSection(model_name="fake/emotion", target_emotions=["tristeza"], fp16=True)
        labeler = EmotionLabeler(config)

        labeler._load()

        assert captured_kwargs["torch_dtype"] == fake_torch.float16

    def test_emotion_aplica_quantizacao_quando_configurado(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """quantize habilitado quantiza o modelo de emoções quando o device é CPU."""
        fake_torch, calls = _fake_torch_module()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(
            sys.modules,
            "transformers",
            _fake_transformers_module(lambda **_: SimpleNamespace(model="modelo-original")),
        )

        config = EmotionSection(
            model_name="fake/emotion", target_emotions=["tristeza"], quantize=True
        )
        labeler = EmotionLabeler(config)
        labeler.device = "cpu"

        pipeline = labeler._load()

        assert pipeline.model == "quantized(modelo-original)"
        assert len(calls) == 1

    def test_prompt_valido_carrega_versao(self, config) -> None:
        """A versão do prompt acompanha o resultado, para rastreabilidade."""
        prompt = build_psychological_prompt(["hoje foi um dia difícil"], config.llm)
        assert prompt.version == config.llm.prompts.version
