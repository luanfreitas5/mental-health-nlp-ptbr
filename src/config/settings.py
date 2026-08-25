"""Carrega e valida todas as configurações do projeto com Pydantic.

Cada arquivo YAML de ``configs/`` tem um modelo correspondente aqui. A
validação acontece no *startup*: uma chave com nome errado, um limiar fora de
faixa ou um peso que não soma 1 derrubam a execução imediatamente, com erro
tipado, em vez de corromper silenciosamente um treinamento de horas.

Segredos (salt de pseudonimização, credenciais do twscrape, id de aprovação
ética) vêm do ``.env`` via :class:`Secrets` e **nunca** de YAML versionado.

Examples
--------
>>> from config.settings import load_config
>>> config = load_config()
>>> config.general.target.classes
['controle', 'depressao', 'ideacao_suicida']
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.paths import ROOT, get_paths
from exceptions.configuration import (
    ConfigFileNotFoundError,
    ConfigParsingError,
    ConfigValidationError,
)

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class _Section(BaseModel):
    """Modelo base das seções de configuração (rejeita chaves desconhecidas)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# config.yaml
# ---------------------------------------------------------------------------


class ProjectSection(_Section):
    """Identificação e parâmetros globais do projeto."""

    name: str
    description: str
    version: str
    language: str = "pt-BR"
    timezone: str = "America/Sao_Paulo"
    random_seed: int = Field(ge=0, default=42)


class TargetSection(_Section):
    """Definição da variável-alvo (rótulo do usuário)."""

    column: str
    classes: list[str] = Field(min_length=2)
    risk_classes: list[str]
    main_metric: str
    multilabel_analysis: bool = True

    @model_validator(mode="after")
    def check_risk_classes(self) -> TargetSection:
        """Garante que toda classe de risco também exista em ``classes``."""
        unknown = set(self.risk_classes) - set(self.classes)
        if unknown:
            raise ValueError(
                f"Classes de risco desconhecidas: {sorted(unknown)}. "
                f"Devem estar em target.classes: {self.classes}"
            )
        return self


class SplitSection(_Section):
    """Particionamento treino/validação/teste agrupado por usuário."""

    group_column: str = "user_id"
    stratify_column: str = "user_label"
    test_size: float = Field(gt=0, lt=1, default=0.20)
    val_size: float = Field(ge=0, lt=1, default=0.10)
    shuffle: bool = True

    @model_validator(mode="after")
    def check_sizes(self) -> SplitSection:
        """Impede que teste + validação consumam todo o conjunto."""
        if self.test_size + self.val_size >= 0.9:
            raise ValueError(
                f"test_size + val_size = {self.test_size + self.val_size:.2f} "
                "deixa menos de 10% dos dados para treino."
            )
        return self


class CrossValidationSection(_Section):
    """Parâmetros da validação cruzada."""

    n_splits: int = Field(ge=2, default=5)
    n_repeats: int = Field(ge=1, default=1)
    strategy: Literal["stratified_group_kfold", "stratified_kfold", "group_kfold"] = (
        "stratified_group_kfold"
    )


class ExperimentSection(_Section):
    """Rastreamento de experimentos (MLflow)."""

    enabled: bool = True
    tracking_uri: str = "mlruns"
    experiment_name: str
    register_best_model: bool = True
    registered_model_name: str
    log_artifacts: bool = True


class PrivacySection(_Section):
    """Salvaguardas de privacidade aplicadas em todo o pipeline (LGPD)."""

    pseudonymize_user_ids: bool = True
    drop_display_names: bool = True
    scrub_mentions: bool = True
    scrub_urls: bool = True
    scrub_emails: bool = True
    scrub_phone_numbers: bool = True
    forbid_text_in_logs: bool = True
    salt_env_var: str = "PSEUDONYMIZATION_SALT"


class GeneralConfig(_Section):
    """Conteúdo validado de ``configs/config.yaml``."""

    project: ProjectSection
    target: TargetSection
    split: SplitSection
    cross_validation: CrossValidationSection
    experiment: ExperimentSection
    privacy: PrivacySection


# ---------------------------------------------------------------------------
# collection.yaml
# ---------------------------------------------------------------------------


class QueryGroup(_Section):
    """Grupo de consulta da busca semente (palavras-chave + hashtags)."""

    keyword_file: str | None = None
    hashtag_file: str | None = None
    candidate_label: str

    @model_validator(mode="after")
    def check_any_file(self) -> QueryGroup:
        """Um grupo sem nenhum arquivo de termos não coletaria nada."""
        if not self.keyword_file and not self.hashtag_file:
            raise ValueError(
                f"O grupo '{self.candidate_label}' não define keyword_file nem hashtag_file."
            )
        return self


class SeedSearchSection(_Section):
    """Busca semente que identifica os usuários candidatos."""

    language: str = "pt"
    since: date
    until: date
    limit_per_query: int = Field(gt=0, default=500)
    exclude_retweets: bool = True
    exclude_replies: bool = False
    groups: dict[str, QueryGroup]

    @model_validator(mode="after")
    def check_window(self) -> SeedSearchSection:
        """A janela de coleta precisa ser cronologicamente válida."""
        if self.since >= self.until:
            raise ValueError(f"Janela inválida: since={self.since} >= until={self.until}.")
        return self


class UserHistorySection(_Section):
    """Coleta retrospectiva do histórico de cada usuário."""

    window_days: int = Field(gt=0, default=365)
    max_tweets_per_user: int = Field(gt=0, default=1000)
    min_tweets_per_user: int = Field(gt=0, default=30)
    min_active_days: int = Field(gt=0, default=15)
    exclude_retweets: bool = True
    exclude_replies: bool = False
    collect_user_metadata: bool = True

    @model_validator(mode="after")
    def check_bounds(self) -> UserHistorySection:
        """O teto de tweets por usuário precisa ser maior que o piso."""
        if self.min_tweets_per_user > self.max_tweets_per_user:
            raise ValueError(
                f"min_tweets_per_user ({self.min_tweets_per_user}) > "
                f"max_tweets_per_user ({self.max_tweets_per_user})."
            )
        return self


class SamplingSection(_Section):
    """Amostragem e controle de balanceamento entre classes."""

    min_users_per_class: int = Field(gt=0, default=300)
    max_users_per_class: int = Field(gt=0, default=1500)
    max_class_imbalance_ratio: float = Field(ge=1.0, default=3.0)
    random_seed: int = 42


class CollectionFiltersSection(_Section):
    """Filtros de qualidade aplicados na ingestão."""

    min_chars_per_tweet: int = Field(ge=0, default=10)
    max_chars_per_tweet: int = Field(gt=0, default=1000)
    drop_duplicated_text: bool = True
    max_tweets_per_day: int = Field(gt=0, default=80)
    min_account_age_days: int = Field(ge=0, default=90)
    require_language: str | None = "pt"


class RateLimitSection(_Section):
    """Controle de vazão e resiliência da coleta."""

    requests_per_minute: int = Field(gt=0, default=30)
    retry_attempts: int = Field(ge=0, default=3)
    backoff_seconds: float = Field(gt=0, default=15)
    backoff_multiplier: float = Field(ge=1.0, default=2.0)
    checkpoint_every_n_users: int = Field(gt=0, default=25)


class TwscrapeSection(_Section):
    """Configuração do cliente twscrape."""

    accounts_db: str
    raise_when_no_account: bool = True


class CollectionConfig(_Section):
    """Conteúdo validado de ``configs/collection.yaml``."""

    seed_search: SeedSearchSection
    user_history: UserHistorySection
    sampling: SamplingSection
    filters: CollectionFiltersSection
    rate_limit: RateLimitSection
    twscrape: TwscrapeSection


# ---------------------------------------------------------------------------
# preprocessing.yaml
# ---------------------------------------------------------------------------


class DeduplicationSection(_Section):
    """Remoção de tweets duplicados."""

    by_tweet_id: bool = True
    by_text_within_user: bool = True
    by_text_global: bool = False
    keep: Literal["first", "last"] = "first"


class NormalizationSection(_Section):
    """Normalização que preserva a semântica (entrada de Transformers/LLM)."""

    unicode_form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC"
    strip_control_chars: bool = True
    collapse_whitespace: bool = True
    replace_urls: str | None = "URL"
    replace_mentions: str | None = "@user"
    replace_emails: str | None = "EMAIL"
    replace_phone_numbers: str | None = "TELEFONE"
    replace_numbers: str | None = None
    unpack_hashtags: bool = True
    collapse_repeated_chars: int | None = 2
    demojize: bool = False


class CleaningSection(_Section):
    """Limpeza agressiva (entrada de TF-IDF, n-grams e léxicos)."""

    lowercase: bool = True
    remove_punctuation: bool = True
    remove_emojis: bool = True
    remove_accents: bool = False
    remove_stopwords: bool = True
    stopwords_file: str = "stopwords_ptbr.txt"
    stopwords_whitelist: list[str] = Field(default_factory=list)
    min_token_length: int = Field(ge=1, default=2)


class TokenizationSection(_Section):
    """Tokenização e lematização."""

    backend: Literal["spacy", "regex"] = "spacy"
    spacy_model: str = "pt_core_news_sm"
    lemmatize: bool = True
    batch_size: int = Field(gt=0, default=256)
    n_process: int = Field(ge=1, default=1)


class PreprocessingFiltersSection(_Section):
    """Filtros aplicados após a limpeza."""

    min_tokens_per_tweet: int = Field(ge=0, default=3)
    drop_empty_after_cleaning: bool = True


class PreprocessingConfig(_Section):
    """Conteúdo validado de ``configs/preprocessing.yaml``."""

    deduplication: DeduplicationSection
    normalization: NormalizationSection
    cleaning: CleaningSection
    tokenization: TokenizationSection
    filters: PreprocessingFiltersSection


# ---------------------------------------------------------------------------
# labeling.yaml
# ---------------------------------------------------------------------------


class SentimentSection(_Section):
    """Rotulação de sentimento por encoder Transformer."""

    enabled: bool = True
    model_name: str
    fallback_model_name: str | None = None
    revision: str = "main"
    batch_size: int = Field(gt=0, default=32)
    max_length: int = Field(gt=0, default=128)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    label_mapping: dict[str, str]
    min_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    cache_predictions: bool = True
    # fp16 só acelera em Tensor Cores de GPU CUDA; fora disso é ignorado (com
    # aviso). quantize aplica quantização dinâmica int8 às camadas lineares
    # e só faz sentido em CPU — as duas flags são mutuamente exclusivas na
    # prática (fp16 para GPU, quantize para CPU) e reduzem o tempo de
    # inferência às custas de uma leve perda de precisão numérica.
    fp16: bool = False
    quantize: bool = False


class EmotionSection(_Section):
    """Classificação de emoções finas por encoder Transformer."""

    enabled: bool = True
    model_name: str
    batch_size: int = Field(gt=0, default=32)
    max_length: int = Field(gt=0, default=128)
    target_emotions: list[str]
    fp16: bool = False
    quantize: bool = False


class LabelSourceSection(_Section):
    """Uma fonte de rótulo da supervisão fraca."""

    enabled: bool = True
    weight: float = Field(ge=0.0, le=1.0)


class TemporalPersistenceSection(_Section):
    """Critério de persistência temporal do sinal."""

    window_days: int = Field(gt=0, default=30)
    min_windows_with_signal: int = Field(ge=1, default=2)
    min_span_days: int = Field(ge=0, default=60)


class ConsensusSection(_Section):
    """Combinação das fontes de rótulo e revisão manual."""

    min_agreement: float = Field(ge=0.0, le=1.0, default=0.6)
    drop_undecided: bool = True
    manual_review_sample_size: int = Field(ge=0, default=200)
    manual_review_file: str
    manual_labels_file: str


class UserLabelingSection(_Section):
    """Rotulação do usuário por supervisão fraca."""

    strategy: Literal["weak_supervision", "manual", "hybrid"] = "weak_supervision"
    sources: dict[str, LabelSourceSection]
    lexical_thresholds: dict[str, dict[str, float]]
    temporal_persistence: TemporalPersistenceSection
    class_precedence: list[str]
    consensus: ConsensusSection
    control_is_screening_negative_only: bool = True

    @model_validator(mode="after")
    def check_weights(self) -> UserLabelingSection:
        """Os pesos das fontes ativas devem somar 1 (voto ponderado)."""
        total = sum(source.weight for source in self.sources.values() if source.enabled)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Os pesos das fontes de rotulação ativas somam {total:.3f}, e não 1.0. "
                "Ajuste user_labeling.sources em configs/labeling.yaml."
            )
        return self


class LabelingConfig(_Section):
    """Conteúdo validado de ``configs/labeling.yaml``."""

    sentiment: SentimentSection
    emotion: EmotionSection
    user_labeling: UserLabelingSection


# ---------------------------------------------------------------------------
# features.yaml
# ---------------------------------------------------------------------------


class NgramsSection(_Section):
    """Vetorização de n-grams."""

    enabled: bool = True
    vectorizer: Literal["tfidf", "count"] = "tfidf"
    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = Field(gt=0, default=3000)
    min_df: int = Field(ge=1, default=5)
    max_df: float = Field(gt=0.0, le=1.0, default=0.85)
    sublinear_tf: bool = True


class LinguisticSection(_Section):
    """Atributos linguísticos."""

    lexicon_ratios: list[str]
    text_length: bool = True
    lexical_diversity: bool = True
    pronouns: bool = True
    ngrams: NgramsSection


class EmotionalSection(_Section):
    """Atributos emocionais agregados por usuário."""

    sentiment_distribution: bool = True
    sentiment_confidence: bool = True
    emotion_intensity: bool = True
    aggregations: list[str]


class ReductionSection(_Section):
    """Redução de dimensionalidade dos embeddings."""

    method: Literal["pca", "none"] = "pca"
    n_components: int = Field(gt=0, default=64)
    random_state: int = 42


class SemanticSection(_Section):
    """Embeddings semânticos e sua agregação por usuário."""

    enabled: bool = True
    primary_model: str
    models: dict[str, str]
    pooling: Literal["mean", "cls"] = "mean"
    max_length: int = Field(gt=0, default=128)
    batch_size: int = Field(gt=0, default=32)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    normalize: bool = True
    # Autocast (fp16/bf16) só acelera em Tensor Cores de GPU CUDA; fora disso
    # é ignorado (com aviso) e o encoder roda em fp32. bf16 tem o mesmo
    # alcance de expoente do fp32 (sem risco de overflow em ativações), mas
    # exige GPU Ampere+; fp16 roda em qualquer CUDA, mas satura em ativações
    # muito grandes. Extração de embedding é só forward pass — sem
    # otimizador/gradiente — então o ganho vira tempo por lote menor e
    # memória livre para lotes maiores, sem GradScaler.
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    user_aggregations: list[str]
    reduction: ReductionSection


class TemporalSection(_Section):
    """Atributos temporais e circadianos."""

    volume: bool = True
    night_activity: bool = True
    insomnia_window: tuple[int, int] = (0, 5)
    circadian_entropy: bool = True
    sentiment_trend: bool = True
    recent_window_days: int = Field(gt=0, default=30)
    polarity_shift: bool = True
    negative_persistence: bool = True
    risk_intensification: bool = True
    min_days_for_trend: int = Field(gt=0, default=14)

    @field_validator("insomnia_window")
    @classmethod
    def check_window(cls, value: tuple[int, int]) -> tuple[int, int]:
        """A janela de insônia deve ser um intervalo horário válido."""
        start, end = value
        if not (0 <= start <= 23 and 0 <= end <= 23):
            raise ValueError(f"insomnia_window fora de 0..23: {value}")
        return value


class BehavioralSection(_Section):
    """Atributos comportamentais de engajamento e audiência."""

    engagement: bool = True
    audience: bool = True
    follower_following_ratio: bool = True
    reply_ratio: bool = True
    log_transform: bool = True
    aggregations: list[str]


class PsychologicalSection(_Section):
    """Vetor psicológico extraído por LLM."""

    enabled: bool = True
    dimensions: list[str]
    granularity: Literal["batch", "tweet"] = "batch"
    batch_size_tweets: int = Field(gt=0, default=20)
    aggregations: list[str]


class AggregationSection(_Section):
    """Agregação tweet -> usuário e tratamento de ausentes."""

    min_tweets_per_user: int = Field(gt=0, default=30)
    add_missing_indicators: bool = True
    missing_strategy: Literal["median", "zero", "keep_nan"] = "median"


class ScalingSection(_Section):
    """Escalonamento das features numéricas."""

    method: Literal["standard", "robust", "none"] = "standard"
    fit_on_train_only: bool = True


class FeaturesConfig(_Section):
    """Conteúdo validado de ``configs/features.yaml``."""

    groups: dict[str, bool]
    linguistic: LinguisticSection
    emotional: EmotionalSection
    semantic: SemanticSection
    temporal: TemporalSection
    behavioral: BehavioralSection
    psychological: PsychologicalSection
    aggregation: AggregationSection
    scaling: ScalingSection

    def enabled_groups(self) -> list[str]:
        """Retorna os grupos de atributos ativos.

        Returns
        -------
        list of str
            Nomes dos grupos com valor ``true`` em ``features.groups``.
        """
        return [name for name, enabled in self.groups.items() if enabled]


# ---------------------------------------------------------------------------
# model_params.yaml
# ---------------------------------------------------------------------------


class ModelSpec(_Section):
    """Especificação de um modelo: escopo, estimador e hiperparâmetros.

    Attributes
    ----------
    scope : str
        ``baseline`` (sempre roda), ``comparison`` (escopo garantido da
        dissertação) ou ``exploratory`` (extensão, roda sob demanda).
    estimator : str
        Chave registrada em :mod:`models.factory`.
    params : dict
        Hiperparâmetros repassados ao estimador.
    feature_groups : list of str, optional
        Restringe o modelo a um subconjunto de grupos de atributos.
    """

    scope: Literal["baseline", "comparison", "exploratory"]
    estimator: str
    params: dict[str, Any] = Field(default_factory=dict)
    feature_groups: list[str] | None = None


class ClassImbalanceSection(_Section):
    """Estratégia de tratamento de desbalanceamento."""

    strategy: Literal["class_weight", "none", "smote"] = "class_weight"


class ModelParamsConfig(_Section):
    """Conteúdo validado de ``configs/model_params.yaml``."""

    baseline: dict[str, ModelSpec]
    traditional: dict[str, ModelSpec]
    deep: dict[str, ModelSpec]
    transformer: dict[str, ModelSpec]
    llm: dict[str, ModelSpec]
    hybrid: dict[str, ModelSpec]
    class_imbalance: ClassImbalanceSection

    def all_models(self) -> dict[str, ModelSpec]:
        """Achata todas as famílias num único dicionário ``nome -> spec``.

        Returns
        -------
        dict of str to ModelSpec
            Todos os modelos declarados, independentemente da família.

        Raises
        ------
        ConfigValidationError
            Se houver nome de modelo duplicado entre famílias.
        """
        merged: dict[str, ModelSpec] = {}
        for family in (
            self.baseline,
            self.traditional,
            self.deep,
            self.transformer,
            self.llm,
            self.hybrid,
        ):
            for name, spec in family.items():
                if name in merged:
                    raise ConfigValidationError(
                        f"Nome de modelo duplicado em model_params.yaml: '{name}'."
                    )
                merged[name] = spec
        return merged

    def select(self, *, include_exploratory: bool = False) -> dict[str, ModelSpec]:
        """Seleciona os modelos a executar conforme o escopo.

        Parameters
        ----------
        include_exploratory : bool, optional
            Inclui a extensão exploratória, by default False.

        Returns
        -------
        dict of str to ModelSpec
            Baseline + comparação principal (+ exploratórios, se solicitado).

        Examples
        --------
        >>> config = load_config()
        >>> sorted(config.models.select())  # doctest: +SKIP
        ['bertimbau', 'bilstm', 'dummy', 'hybrid_xgboost', ...]
        """
        allowed = {"baseline", "comparison"}
        if include_exploratory:
            allowed.add("exploratory")
        return {name: spec for name, spec in self.all_models().items() if spec.scope in allowed}


# ---------------------------------------------------------------------------
# evaluation.yaml
# ---------------------------------------------------------------------------


class MetricsSection(_Section):
    """Métricas calculadas e destacadas."""

    primary: str
    compute: list[str]
    per_class: bool = True
    highlight: list[str] = Field(default_factory=list)


class UncertaintySection(_Section):
    """Quantificação de incerteza das métricas."""

    method: Literal["bootstrap", "cv_std"] = "bootstrap"
    n_bootstrap: int = Field(gt=0, default=1000)
    confidence_level: float = Field(gt=0.0, lt=1.0, default=0.95)
    random_state: int = 42


class CalibrationSection(_Section):
    """Avaliação de calibração das probabilidades."""

    enabled: bool = True
    n_bins: int = Field(gt=1, default=10)
    metrics: list[str]
    reliability_curve: bool = True


class TestToggle(_Section):
    """Liga/desliga um teste estatístico com parâmetros opcionais."""

    enabled: bool = True
    correction: bool | None = None
    alternative: str | None = None
    posthoc: str | None = None


class StatisticsSection(_Section):
    """Testes de significância entre modelos."""

    mcnemar: TestToggle
    wilcoxon: TestToggle
    friedman: TestToggle
    alpha: float = Field(gt=0.0, lt=1.0, default=0.05)
    multiple_comparison_correction: Literal["holm", "bonferroni", "none"] = "holm"
    effect_size: Literal["cliffs_delta", "cohens_d", "none"] = "cliffs_delta"


class SliceDefinition(_Section):
    """Definição de uma fatia de avaliação."""

    column: str
    bins: list[float]
    labels: list[str]

    @model_validator(mode="after")
    def check_lengths(self) -> SliceDefinition:
        """``bins`` precisa ter exatamente um elemento a mais que ``labels``."""
        if len(self.bins) != len(self.labels) + 1:
            raise ValueError(
                f"Fatia com {len(self.bins)} bins e {len(self.labels)} rótulos: "
                "esperado len(bins) == len(labels) + 1."
            )
        return self


class SlicesSection(_Section):
    """Avaliação por fatias (subgrupos comportamentais)."""

    enabled: bool = True
    definitions: dict[str, SliceDefinition]
    min_samples_per_slice: int = Field(gt=0, default=20)
    max_acceptable_gap: float = Field(ge=0.0, le=1.0, default=0.15)


class AblationSection(_Section):
    """Ablation Study sobre os grupos de atributos."""

    enabled: bool = True
    base_model: str
    groups: list[str]
    mode: Literal["leave_one_out", "only_one", "both"] = "leave_one_out"
    include_only_one: bool = True
    n_repeats: int = Field(ge=1, default=5)


class GranularityComparisonSection(_Section):
    """Comparação tweet-level vs. user-level (hipótese H5)."""

    enabled: bool = True
    tweet_level_aggregation: Literal["majority_vote", "mean_proba"] = "majority_vote"
    models: list[str]


class ShapSection(_Section):
    """Configuração do SHAP."""

    enabled: bool = True
    explainer: Literal["tree", "kernel", "linear"] = "tree"
    max_display: int = Field(gt=0, default=25)
    sample_size: int = Field(gt=0, default=500)
    random_state: int = 42


class PermutationImportanceSection(_Section):
    """Configuração da importância por permutação."""

    enabled: bool = True
    n_repeats: int = Field(ge=1, default=10)
    random_state: int = 42
    scoring: str = "f1_macro"


class InterpretabilitySection(_Section):
    """Interpretabilidade do modelo final."""

    shap: ShapSection
    permutation_importance: PermutationImportanceSection


class ReportingSection(_Section):
    """Formatos e artefatos dos relatórios gerados."""

    formats: list[str]
    figure_formats: list[str]
    figure_dpi: int = Field(gt=0, default=300)
    save_predictions: bool = True
    generate_model_card: bool = True


class EvaluationConfig(_Section):
    """Conteúdo validado de ``configs/evaluation.yaml``."""

    metrics: MetricsSection
    uncertainty: UncertaintySection
    calibration: CalibrationSection
    statistics: StatisticsSection
    slices: SlicesSection
    ablation: AblationSection
    granularity_comparison: GranularityComparisonSection
    interpretability: InterpretabilitySection
    regression_thresholds: dict[str, dict[str, float]]
    reporting: ReportingSection


# ---------------------------------------------------------------------------
# llm.yaml
# ---------------------------------------------------------------------------


class OllamaSection(_Section):
    """Conexão com o servidor Ollama local."""

    host: str = "http://localhost:11434"
    timeout_seconds: float = Field(gt=0, default=120)
    keep_alive: str = "5m"
    auto_pull: bool = False
    retry_attempts: int = Field(ge=0, default=3)
    backoff_seconds: float = Field(gt=0, default=5)


class LLMCacheSection(_Section):
    """Cache de respostas do LLM, indexado por hash do prompt."""

    enabled: bool = True
    path: str


class PsychologicalFeaturesSection(_Section):
    """Extração do vetor psicológico via LLM."""

    enabled: bool = True
    model: str
    temperature: float = Field(ge=0.0, le=2.0, default=0.0)
    seed: int = 42
    top_p: float = Field(gt=0.0, le=1.0, default=1.0)
    num_ctx: int = Field(gt=0, default=8192)
    response_format: Literal["json", "text"] = "json"
    max_repairs: int = Field(ge=0, default=2)
    dimensions: list[str]
    batch_size_tweets: int = Field(gt=0, default=20)
    max_concurrency: int = Field(ge=1, default=2)
    cache: LLMCacheSection


class FewShotSection(_Section):
    """Amostragem dos exemplos few-shot."""

    n_examples_per_class: int = Field(ge=1, default=1)
    source_split: Literal["train"] = "train"
    random_state: int = 42


class LLMClassifierSection(_Section):
    """O LLM atuando como classificador da comparação principal."""

    enabled: bool = True
    model: str
    mode: Literal["zero_shot", "few_shot"] = "few_shot"
    temperature: float = Field(ge=0.0, le=2.0, default=0.0)
    seed: int = 42
    num_ctx: int = Field(gt=0, default=8192)
    max_tweets_per_prompt: int = Field(gt=0, default=20)
    few_shot: FewShotSection
    response_format: Literal["json", "text"] = "json"


class PromptsSection(_Section):
    """Prompts versionados (mudança de prompt = mudança de método)."""

    version: str
    language: str = "pt-BR"
    psychological_system: str
    psychological_user: str
    classifier_system: str
    classifier_user: str


class SafeguardsSection(_Section):
    """Salvaguardas aplicadas antes de qualquer chamada ao LLM."""

    require_pii_scrubbed_input: bool = True
    max_prompt_chars: int = Field(gt=0, default=24000)
    log_prompt_content: bool = False


class LLMConfig(_Section):
    """Conteúdo validado de ``configs/llm.yaml``."""

    provider: Literal["ollama"] = "ollama"
    ollama: OllamaSection
    psychological_features: PsychologicalFeaturesSection
    classifier: LLMClassifierSection
    prompts: PromptsSection
    safeguards: SafeguardsSection


# ---------------------------------------------------------------------------
# .env (segredos)
# ---------------------------------------------------------------------------


class Secrets(BaseSettings):
    """Segredos e variáveis de ambiente, carregados de ``.env``.

    Nunca versionado. Ver ``.env.example`` para o modelo.

    Attributes
    ----------
    pseudonymization_salt : str
        Salt do hash SHA-256 que pseudonimiza os identificadores de usuário.
        Sem ele, o hash seria reversível por força bruta sobre handles públicos.
    ethics_approval_id : str, optional
        Número do CAAE da aprovação CEP/CONEP. A etapa de coleta é bloqueada
        enquanto estiver vazio (ver :mod:`pipelines.collection`).
    """

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    pseudonymization_salt: str = "salt-de-desenvolvimento-troque-no-env"
    ethics_approval_id: str = ""
    twscrape_accounts_db: str = ""
    mlflow_tracking_uri: str = ""
    ollama_host: str = ""


# ---------------------------------------------------------------------------
# Agregador
# ---------------------------------------------------------------------------


class Config(BaseModel):
    """Configuração completa do projeto, já validada.

    Attributes
    ----------
    general, collection, preprocessing, labeling, features, models, evaluation, llm
        Conteúdo validado de cada arquivo YAML de ``configs/``.
    secrets
        Variáveis sensíveis carregadas do ``.env``.
    """

    model_config = ConfigDict(frozen=True)

    general: GeneralConfig
    collection: CollectionConfig
    preprocessing: PreprocessingConfig
    labeling: LabelingConfig
    features: FeaturesConfig
    models: ModelParamsConfig
    evaluation: EvaluationConfig
    llm: LLMConfig
    secrets: Secrets

    @property
    def random_seed(self) -> int:
        """Semente global de reprodutibilidade."""
        return self.general.project.random_seed

    @property
    def classes(self) -> list[str]:
        """Classes da variável-alvo, na ordem canônica."""
        return self.general.target.classes


def read_yaml(path: Path) -> dict[str, Any]:
    """Lê um YAML de configuração e devolve um dicionário.

    Parameters
    ----------
    path : Path
        Caminho do arquivo.

    Returns
    -------
    dict
        Conteúdo do arquivo (dicionário vazio se o arquivo estiver vazio).

    Raises
    ------
    ConfigFileNotFoundError
        Se o arquivo não existir.
    ConfigParsingError
        Se o YAML for inválido ou não representar um mapeamento.
    """
    if not path.is_file():
        raise ConfigFileNotFoundError(f"Arquivo de configuração não encontrado: {path}")

    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigParsingError(f"YAML inválido em {path}: {error}") from error

    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ConfigParsingError(f"Esperado um mapeamento no topo de {path}, veio {type(content)}.")
    return content


def _build(model: type[BaseModel], path: Path) -> Any:
    """Instancia um modelo Pydantic a partir de um YAML, com erro contextualizado."""
    payload = read_yaml(path)
    try:
        return model(**payload)
    except ValueError as error:
        raise ConfigValidationError(f"Configuração inválida em {path.name}:\n{error}") from error


@lru_cache(maxsize=1)
def load_config(configs_dir: Path | None = None) -> Config:
    """Carrega e valida todas as configurações do projeto.

    O resultado é memoizado: os YAMLs são lidos uma única vez por processo.

    Parameters
    ----------
    configs_dir : Path, optional
        Diretório alternativo de configurações, by default ``configs/``.

    Returns
    -------
    Config
        Configuração completa e validada.

    Raises
    ------
    ConfigFileNotFoundError
        Se algum arquivo obrigatório estiver ausente.
    ConfigValidationError
        Se algum valor violar o schema ou as regras de negócio.

    Examples
    --------
    >>> config = load_config()
    >>> config.general.project.name
    'mental-health-nlp-ptbr'
    """
    directory = Path(configs_dir) if configs_dir else get_paths().configs.root

    return Config(
        general=_build(GeneralConfig, directory / "config.yaml"),
        collection=_build(CollectionConfig, directory / "collection.yaml"),
        preprocessing=_build(PreprocessingConfig, directory / "preprocessing.yaml"),
        labeling=_build(LabelingConfig, directory / "labeling.yaml"),
        features=_build(FeaturesConfig, directory / "features.yaml"),
        models=_build(ModelParamsConfig, directory / "model_params.yaml"),
        evaluation=_build(EvaluationConfig, directory / "evaluation.yaml"),
        llm=_build(LLMConfig, directory / "llm.yaml"),
        secrets=Secrets(),
    )
