"""Rotulação automática em dois níveis: sentimento por tweet e classe por usuário.

Os dois níveis são constructos distintos e não intercambiáveis. O sentimento
(``positivo``/``negativo``/``neutro``) é uma feature e uma triagem; o rótulo do
usuário (``controle``/``depressao``/``ideacao_suicida``) é a variável-alvo.
Tratar o primeiro como proxy do segundo seria um erro de validade de
constructo — e é justamente a confusão que a proposta se propõe a evitar.

Modules
-------
sentiment
    :class:`SentimentLabeler` — encoder Transformer fine-tuned para pt-BR.
emotion
    :class:`EmotionLabeler` — emoções finas (tristeza, raiva, medo, ...).
llm
    :class:`OllamaClient` e :class:`PsychologicalExtractor` — vetor psicológico
    extraído por LLM local, com saída validada e cache.
prompt
    Construção e versionamento dos prompts.
weak_supervision
    :func:`assign_user_labels` — voto ponderado entre grupo de coleta,
    evidência léxica e persistência temporal.
validation
    Amostra de revisão manual, kappa de Cohen e descarte de indefinidos.
"""

from labeling.emotion import EmotionLabeler
from labeling.llm import OllamaClient, PsychologicalExtractor, PsychologicalVector
from labeling.prompt import Prompt, build_classifier_prompt, build_psychological_prompt
from labeling.sentiment import SentimentLabeler, SentimentPrediction
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
    resolve_consensus,
)

__all__ = [
    "EmotionLabeler",
    "LabelVote",
    "OllamaClient",
    "Prompt",
    "PsychologicalExtractor",
    "PsychologicalVector",
    "SentimentLabeler",
    "SentimentPrediction",
    "apply_manual_labels",
    "assign_user_labels",
    "build_classifier_prompt",
    "build_psychological_prompt",
    "compute_agreement",
    "compute_lexical_evidence",
    "compute_temporal_persistence",
    "drop_undecided",
    "load_manual_labels",
    "resolve_consensus",
    "sample_for_manual_review",
]
