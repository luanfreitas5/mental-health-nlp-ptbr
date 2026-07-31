"""Rótulos, enums e mapeamentos das variáveis-alvo.

Duas variáveis distintas convivem no projeto e **não** devem ser confundidas:

* :class:`UserLabel` — o constructo principal, no nível do usuário.
* :class:`Sentiment` — constructo auxiliar, no nível do tweet, usado como
  triagem e como feature. Não é proxy de risco clínico.

Os enums herdam de ``(str, Enum)`` em vez de ``StrEnum`` porque o projeto
suporta Python 3.10, onde ``StrEnum`` ainda não existe. O ``__str__``
explícito garante que a interpolação em f-strings produza o valor
(``"controle"``) e não a representação do membro, em qualquer versão.
"""

from __future__ import annotations

from enum import Enum
from typing import Final


class _ValueStr(str, Enum):
    """Enum de strings cuja representação textual é sempre o próprio valor."""

    def __str__(self) -> str:
        """Retorna o valor da constante (ex.: ``'controle'``)."""
        return str(self.value)


class UserLabel(_ValueStr):
    """Classe do usuário (variável-alvo principal).

    Notes
    -----
    ``CONTROLE`` significa "sem sinais **detectados** no recorte coletado" —
    nunca "ausência clínica confirmada". Ver a limitação de viés de seleção em
    ``reports/datasheets/``.
    """

    CONTROLE = "controle"
    DEPRESSAO = "depressao"
    IDEACAO_SUICIDA = "ideacao_suicida"
    #: Usado apenas na rotulação: consenso insuficiente entre as fontes.
    INDEFINIDO = "indefinido"


class Sentiment(_ValueStr):
    """Sentimento do tweet (constructo auxiliar)."""

    POSITIVO = "positivo"
    NEGATIVO = "negativo"
    NEUTRO = "neutro"
    #: Confiança do encoder abaixo de ``labeling.sentiment.min_confidence``.
    INDEFINIDO = "indefinido"


class Emotion(_ValueStr):
    """Emoções finas previstas pelo encoder multi-rótulo."""

    TRISTEZA = "tristeza"
    RAIVA = "raiva"
    MEDO = "medo"
    ALEGRIA = "alegria"
    NOJO = "nojo"
    SURPRESA = "surpresa"


class PsychologicalDimension(_ValueStr):
    """Dimensões do vetor psicológico extraído por LLM."""

    TRISTEZA = "tristeza"
    ISOLAMENTO = "isolamento"
    ESPERANCA = "esperanca"
    ANSIEDADE = "ansiedade"
    RISCO_SUICIDA = "risco_suicida"


class FeatureGroup(_ValueStr):
    """Grupos de atributos — unidade do Ablation Study."""

    LINGUISTIC = "linguistic"
    EMOTIONAL = "emotional"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    PSYCHOLOGICAL = "psychological"


class Split(_ValueStr):
    """Partições do dataset."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


#: Ordem canônica das classes. Fixa a ordem das colunas de probabilidade, das
#: linhas da matriz de confusão e dos rótulos das figuras — sem isso, gráficos
#: e métricas de execuções diferentes não seriam comparáveis.
CLASS_ORDER: Final[tuple[str, ...]] = (
    UserLabel.CONTROLE.value,
    UserLabel.DEPRESSAO.value,
    UserLabel.IDEACAO_SUICIDA.value,
)

#: Classes consideradas de risco (usadas na binarização "risco vs. controle").
RISK_CLASSES: Final[tuple[str, ...]] = (
    UserLabel.DEPRESSAO.value,
    UserLabel.IDEACAO_SUICIDA.value,
)

#: Precedência em caso de coocorrência: severidade decrescente.
CLASS_PRECEDENCE: Final[tuple[str, ...]] = (
    UserLabel.IDEACAO_SUICIDA.value,
    UserLabel.DEPRESSAO.value,
    UserLabel.CONTROLE.value,
)

#: Rótulo -> índice inteiro (entrada dos modelos).
LABEL_TO_INDEX: Final[dict[str, int]] = {label: index for index, label in enumerate(CLASS_ORDER)}

#: Índice inteiro -> rótulo (saída legível dos modelos).
INDEX_TO_LABEL: Final[dict[int, str]] = {index: label for label, index in LABEL_TO_INDEX.items()}

#: Polaridade numérica do sentimento, usada nas séries temporais de humor.
SENTIMENT_POLARITY: Final[dict[str, float]] = {
    Sentiment.POSITIVO.value: 1.0,
    Sentiment.NEGATIVO.value: -1.0,
    Sentiment.NEUTRO.value: 0.0,
    Sentiment.INDEFINIDO.value: 0.0,
}

#: Nomes das classes para exibição em figuras e relatórios (pt-BR).
CLASS_DISPLAY_NAMES: Final[dict[str, str]] = {
    UserLabel.CONTROLE.value: "Controle",
    UserLabel.DEPRESSAO.value: "Depressão",
    UserLabel.IDEACAO_SUICIDA.value: "Ideação Suicida",
    UserLabel.INDEFINIDO.value: "Indefinido",
}
