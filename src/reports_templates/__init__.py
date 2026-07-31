"""Documentos de IA responsável gerados a partir dos resultados.

Modules
-------
model_card
    :func:`build_model_card` — uso pretendido, usos fora de escopo,
    desempenho por subgrupo, limitações e considerações éticas
    (Mitchell et al., 2019).
datasheet
    :func:`build_datasheet` — motivação, composição, coleta, privacidade e
    manutenção do dataset (Gebru et al., 2021).
"""

from reports_templates.datasheet import build_datasheet
from reports_templates.model_card import build_model_card

__all__ = ["build_datasheet", "build_model_card"]
