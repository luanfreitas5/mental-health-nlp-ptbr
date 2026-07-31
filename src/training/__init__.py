"""Treinamento final e validação cruzada.

Modules
-------
trainer
    Montagem dos :class:`~models.base.UserDataset` por partição, treinamento
    e persistência dos modelos.
cross_validation
    Validação cruzada sobre os folds fixados em ``splits.parquet`` — os
    mesmos blocos para todos os modelos, requisito dos testes pareados.
"""

from training.cross_validation import (
    build_fold_datasets,
    cross_validate_all,
    cross_validate_model,
    extract_fold_scores,
)
from training.trainer import (
    build_dataset,
    load_training_inputs,
    load_user_sequences,
    load_user_texts,
    split_features,
    train_all,
    train_model,
)

__all__ = [
    "build_dataset",
    "build_fold_datasets",
    "cross_validate_all",
    "cross_validate_model",
    "extract_fold_scores",
    "load_training_inputs",
    "load_user_sequences",
    "load_user_texts",
    "split_features",
    "train_all",
    "train_model",
]
