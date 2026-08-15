## v0.3.0 (2026-08-15)

### Feat

- **data**: garantir a unicidade do usuário na matriz de características
- **data**: suporte a artefatos de dados particionados
- **project**: Implementar um pipeline de detecção de saúde mental por meio de PNL (Processamento de Linguagem Natural)
- **dashboard**: Adicionar aplicativo de inspeção de resultados do Streamlit
- **config**: Adiciona uma estrutura de configuração abrangente e modelos de coleta de dados.

### Fix

- restringir o Python e corrigir o processamento de texto
- **models**: lidar com classes ausentes em folds do XGBoost
- **models**: reindexar rótulos para folds do XGBoost
- **builder**: normalizar valores NaN para nulo em colunas float
- **reader**: remover tweets duplicados por ID em diferentes arquivos de histórico
- **types**: converter tipos de retorno da biblioteca

### Refactor

- **data**: adicionar funções de leitura de partições por usuário
- usar encadeamento de métodos e remover chamadas redundantes a `str()`
- extrair funções auxiliares e melhorar a modularidade em toda a base de código
- **pipeline**: processamento de usuários em fluxo contínuo com persistência de resultados por usuário
- **models**: extrair RecurrentClassifier para uma fábrica em nível de módulo
- usar a sintaxe de união PEP 604 para verificação de tipos

### Perf

- **config**: otimizar tamanhos de lote e concorrência para V100S-32GB
- **llm**: Paralelização da extração em lote
