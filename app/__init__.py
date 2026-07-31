"""Dashboard Streamlit de inspeção dos resultados.

Modules
-------
dashboard
    Interface que lê os artefatos de ``reports/`` — comparação entre modelos,
    detalhamento por modelo, testes estatísticos, figuras e documentos de IA
    responsável. Não treina nem recalcula nada: um dashboard que dispara
    processamento vira uma segunda implementação do pipeline, com risco de
    divergir dele.
"""
