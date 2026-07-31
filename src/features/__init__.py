"""Extração de atributos e agregação tweet -> usuário.

Os seis grupos correspondem às Seções 1 a 6 da proposta e são a unidade do
Ablation Study: cada um pode ser ativado ou removido individualmente, o que
permite medir a contribuição marginal de cada família de atributos (H2, H3, H4).

Convenção de prefixos, usada para selecionar grupos sem manter listas de
colunas: ``ling_``, ``emo_``, ``sem_``, ``temp_``, ``behav_``, ``psy_``.

Modules
-------
linguistic
    Léxicos de risco, comprimento, diversidade lexical e pronomes.
emotional
    Distribuição de sentimento, confiança e emoções finas.
semantic
    Embeddings de Transformers e agregação por usuário.
temporal
    Volume, ritmo circadiano, tendência de humor e intensificação de risco.
behavioral
    Engajamento recebido, audiência e razões de interação.
psychological
    Agregação do vetor psicológico extraído por LLM.
ngrams
    :class:`UserNgramVectorizer` — TF-IDF por usuário, ajustado dentro do
    ``Pipeline`` para não vazar vocabulário do teste.
builder
    :func:`build_user_features` — monta a matriz final e trata ausentes.
"""

from features.behavioral import build_behavioral_features
from features.builder import build_profile_columns, build_user_features, select_groups
from features.emotional import build_emotional_features
from features.linguistic import build_linguistic_features
from features.ngrams import UserNgramVectorizer, build_user_documents
from features.psychological import build_psychological_features
from features.semantic import EmbeddingEncoder, aggregate_embeddings, build_semantic_features
from features.temporal import build_temporal_features

__all__ = [
    "EmbeddingEncoder",
    "UserNgramVectorizer",
    "aggregate_embeddings",
    "build_behavioral_features",
    "build_emotional_features",
    "build_linguistic_features",
    "build_profile_columns",
    "build_psychological_features",
    "build_semantic_features",
    "build_temporal_features",
    "build_user_documents",
    "build_user_features",
    "select_groups",
]
