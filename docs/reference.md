# Referência da API

Documentação gerada automaticamente a partir das docstrings NumPy (em pt-BR).

---

## Configuração

::: config.settings
    options:
      members:
        - Config
        - load_config
        - read_yaml

::: config.paths
    options:
      members:
        - ProjectPaths
        - get_paths
        - resolve_path

::: config.logging
    options:
      members:
        - configure_logging
        - get_logger
        - PIIRedactionFilter

::: config.environment
    options:
      members:
        - seed_everything
        - resolve_device
        - describe_environment

---

## Constantes

::: constants.labels

::: constants.metrics
    options:
      members:
        - is_higher_better

---

## Contratos de dados

::: schemas.tweets

::: schemas.users

::: schemas.features
    options:
      members:
        - list_feature_columns
        - validate_feature_matrix

::: schemas.validation
    options:
      members:
        - validate_frame
        - is_valid

---

## Dados

::: data.queries

::: data.collector
    options:
      members:
        - TweetCollector
        - CandidateUser

::: data.splitter

::: data.reader
    options:
      members:
        - read_parquet
        - read_user_histories

::: data.writer
    options:
      members:
        - write_parquet
        - write_partitioned

::: data.catalog
    options:
      members:
        - build_catalog
        - write_dataset_manifest
        - compare_manifest

---

## Pré-processamento

::: preprocessing.text

::: preprocessing.cleaning

::: preprocessing.tokenization
    options:
      members:
        - Tokenizer

::: preprocessing.pipeline
    options:
      members:
        - run_preprocessing

---

## Rotulação

::: labeling.sentiment
    options:
      members:
        - SentimentLabeler
        - SentimentPrediction

::: labeling.emotion
    options:
      members:
        - EmotionLabeler

::: labeling.weak_supervision

::: labeling.llm
    options:
      members:
        - OllamaClient
        - PsychologicalExtractor
        - PsychologicalVector
        - extract_json_object

::: labeling.prompt

::: labeling.validation

---

## Atributos

::: features.linguistic

::: features.emotional

::: features.semantic
    options:
      members:
        - EmbeddingEncoder
        - aggregate_embeddings
        - build_semantic_features

::: features.temporal

::: features.behavioral

::: features.psychological

::: features.ngrams
    options:
      members:
        - UserNgramVectorizer
        - build_user_documents

::: features.builder

---

## Modelos

::: models.base
    options:
      members:
        - BaseUserClassifier
        - UserDataset

::: models.traditional
    options:
      members:
        - TabularClassifier
        - build_estimator

::: models.deep
    options:
      members:
        - SequenceClassifier
        - build_sequence_batch

::: models.transformer
    options:
      members:
        - TransformerClassifier
        - aggregate_user_probabilities

::: models.llm
    options:
      members:
        - LLMClassifier

::: models.hybrid
    options:
      members:
        - HybridClassifier
        - split_feature_blocks

::: models.factory

::: models.persistence

---

## Treinamento

::: training.trainer

::: training.cross_validation

---

## Avaliação

::: evaluation.metrics

::: evaluation.calibration

::: evaluation.statistics

::: evaluation.slices

::: evaluation.ablation

::: evaluation.evaluator
    options:
      members:
        - Evaluator
        - EvaluationResult

::: evaluation.reports

---

## Interpretabilidade

::: interpretability.importance

::: interpretability.shap_values

---

## Visualização

::: visualization.theme

::: visualization.distributions

::: visualization.evaluation_plots

::: visualization.temporal_plots

::: visualization.embeddings

::: visualization.interpretability_plots

---

## Pipelines

::: pipelines.base
    options:
      members:
        - PipelineStage
        - StageContext

::: pipelines.workflow

---

## IA responsável

::: reports_templates.model_card

::: reports_templates.datasheet

---

## Utilitários

::: utils.hashing

::: utils.lexicons

::: utils.files

::: utils.validation

::: utils.progress

::: utils.timing

---

## Exceções

::: exceptions.base

::: exceptions.configuration

::: exceptions.data

::: exceptions.model

::: exceptions.pipeline
