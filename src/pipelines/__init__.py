"""Etapas do pipeline, executáveis de forma independente.

O acoplamento entre etapas é o sistema de arquivos, não a memória: cada uma lê
artefatos do disco e grava outros. Isso permite executar uma etapa isolada,
retomar uma execução interrompida e inspecionar qualquer resultado
intermediário — essencial num projeto em que a coleta leva dias e o
fine-tuning leva horas.

Modules
-------
base
    :class:`PipelineStage` e :class:`StageContext`.
collection
    Etapa 1 — coleta, protegida por barreira ética (CEP/CONEP).
preprocessing
    Etapa 2 — limpeza, normalização e filtros.
labeling
    Etapa 3 — sentimento por tweet e classe por usuário.
psychological
    Etapa 4 — vetor psicológico via LLM local.
embedding
    Etapa 5 — embeddings semânticos por tweet.
features
    Etapa 6 — matriz de atributos por usuário.
splitting
    Etapa 7 — partições e folds agrupados por usuário.
training
    Etapa 8 — validação cruzada e treinamento final.
evaluation
    Etapa 9 — teste, significância estatística, ablação e SHAP.
reporting
    Etapa 10 — figuras, model card e datasheet.
workflow
    :func:`run_stage` e :func:`run_pipeline` — registro e orquestração.
"""

from pipelines.base import PipelineStage, StageContext
from pipelines.workflow import (
    DEFAULT_PIPELINE,
    STAGES,
    describe_stages,
    get_stage,
    run_pipeline,
    run_stage,
)

__all__ = [
    "DEFAULT_PIPELINE",
    "STAGES",
    "PipelineStage",
    "StageContext",
    "describe_stages",
    "get_stage",
    "run_pipeline",
    "run_stage",
]
