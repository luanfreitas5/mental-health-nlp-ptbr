"""Contratos de dados (pandera) aplicados nas fronteiras do pipeline.

Cada estágio valida o que recebe e o que produz. Os contratos são versionados
junto do código: uma mudança de schema aparece no diff do PR, o que torna
explícita uma alteração que, de outro modo, só apareceria como métrica
estranha semanas depois.

Modules
-------
tweets
    :class:`RawTweetSchema`, :class:`CleanTweetSchema`,
    :class:`LabeledTweetSchema` e :class:`PsychologicalScoreSchema`.
users
    :class:`UserMetadataSchema`, :class:`UserLabelSchema`,
    :class:`UserProfileSchema` e :class:`SplitSchema`.
features
    Validação estrutural da matriz de atributos e seleção por grupo.
validation
    :func:`validate_frame` — aplica um contrato traduzindo o erro do pandera
    em :class:`~exceptions.data.SchemaValidationError`.
"""

from schemas.features import (
    UserFeatureKeySchema,
    list_feature_columns,
    validate_feature_matrix,
)
from schemas.tweets import (
    CleanTweetSchema,
    LabeledTweetSchema,
    PsychologicalScoreSchema,
    RawTweetSchema,
)
from schemas.users import (
    SplitSchema,
    UserLabelSchema,
    UserMetadataSchema,
    UserProfileSchema,
)
from schemas.validation import is_valid, validate_frame

__all__ = [
    "CleanTweetSchema",
    "LabeledTweetSchema",
    "PsychologicalScoreSchema",
    "RawTweetSchema",
    "SplitSchema",
    "UserFeatureKeySchema",
    "UserLabelSchema",
    "UserMetadataSchema",
    "UserProfileSchema",
    "is_valid",
    "list_feature_columns",
    "validate_feature_matrix",
    "validate_frame",
]
