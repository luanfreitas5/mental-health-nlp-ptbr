"""Código-fonte do projeto ``mental-health-nlp-ptbr``.

Detecção longitudinal de sinais de depressão e ideação suicida em redes
sociais com Transformers e Modelos de Linguagem.

Este diretório é a **raiz de importação**: os módulos são importados sem o
prefixo ``src.`` (ex.: ``from config.settings import load_config``), conforme
o padrão do projeto. ``src/main.py`` insere este diretório em ``sys.path``
quando executado diretamente, e ``pyproject.toml`` faz o mesmo para o pytest.

Packages
--------
config
    Configuração validada com Pydantic, caminhos, logging e reprodutibilidade.
constants
    Constantes estruturais: colunas, rótulos, métricas e padrões regex.
exceptions
    Hierarquia de exceções do domínio.
schemas
    Contratos de dados (pandera) aplicados nas fronteiras do pipeline.
utils
    Utilitários transversais: hashing/pseudonimização, léxicos, progresso.
data
    Coleta, leitura, escrita e particionamento agrupado por usuário.
preprocessing
    Limpeza, normalização e tokenização.
labeling
    Sentimento por tweet, vetor psicológico por LLM e supervisão fraca.
features
    Seis grupos de atributos e agregação tweet -> usuário.
models
    Classificadores tabulares, recorrentes, Transformer, LLM e híbrido.
training
    Treinamento final e validação cruzada.
evaluation
    Métricas com incerteza, testes estatísticos, fatias e ablação.
interpretability
    Importância por permutação e valores SHAP.
visualization
    Figuras com tema e paleta compartilhados.
experiment
    Rastreamento com MLflow.
reports_templates
    Model Card e Datasheet.
pipelines
    Etapas do pipeline e orquestração.
"""

__all__: list[str] = []
