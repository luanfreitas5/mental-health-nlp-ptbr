"""Suíte de testes do projeto.

Organização:

- ``test_config`` — carregamento e validação da configuração.
- ``test_utils`` — pseudonimização, hashing, léxicos e verificações defensivas.
- ``test_preprocessing`` — normalização, limpeza e filtros.
- ``test_features`` — os seis grupos de atributos e a agregação por usuário.
- ``test_labeling`` — supervisão fraca e salvaguardas do LLM.
- ``test_models`` — modelos, fábrica e persistência.
- ``test_evaluation`` — métricas, incerteza, calibração e testes estatísticos.
- ``test_data`` — particionamento, IO e consultas de coleta.
- ``test_schemas`` — contratos de dados (pandera).
- ``test_properties`` — invariantes verificadas com ``hypothesis``.
- ``test_integration`` — encadeamento das etapas, comportamento do modelo e
  regressão de métrica.

Todos os dados usados aqui são **sintéticos** (ver ``conftest.py``): dados
reais deste projeto são sensíveis e não entram no repositório.
"""
