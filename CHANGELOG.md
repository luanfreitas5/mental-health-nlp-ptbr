# Changelog

Todas as mudanças relevantes deste projeto são documentadas neste arquivo.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e o
versionamento segue [Semantic Versioning](https://semver.org/lang/pt-BR/).
As entradas são geradas pelo `commitizen` a partir dos Conventional Commits
(`cz bump --changelog`).

## v0.2.0 (2026-07-31)

### Feat

- **project**: Implementar um pipeline de detecção de saúde mental por meio de PNL (Processamento de Linguagem Natural)
- **dashboard**: Adicionar aplicativo de inspeção de resultados do Streamlit
- **config**: Adiciona uma estrutura de configuração abrangente e modelos de coleta de dados.

### Fix

- **types**: converter tipos de retorno da biblioteca

### Refactor

- usar a sintaxe de união PEP 604 para verificação de tipos

## 0.1.0 (2026-07-30)

### Adicionado

**Pipeline** — dez etapas independentes, orquestradas por `src/main.py --stage`:
coleta, pré-processamento, rotulação, vetor psicológico, embeddings, atributos,
particionamento, treinamento, avaliação e relatórios. O acoplamento entre etapas
é o sistema de arquivos, o que permite reexecutar apenas o que mudou.

**Configuração** — YAMLs em `configs/` validados com Pydantic no *startup*:
geral, caminhos, logging, coleta, pré-processamento, rotulação, atributos,
hiperparâmetros, avaliação e LLM. Palavras-chave, hashtags e léxicos
psicolinguísticos versionados como `.txt`.

**Contratos de dados** — schemas `pandera` aplicados na entrada e na saída de
cada estágio (`raw → interim → processed`), incluindo validação estrutural da
matriz de atributos.

**Atributos** — seis grupos agregados no nível do usuário: linguísticos,
emocionais, semânticos, temporais, comportamentais e psicológicos. Convenção de
prefixos que sustenta o Ablation Study.

**Modelos** — interface única (`BaseUserClassifier`) para famílias distintas:
tabulares (Dummy, Regressão Logística, Random Forest, XGBoost, LightGBM), BiLSTM
sobre sequências de embeddings, Transformer com agregação por usuário, LLM local
via Ollama e o modelo híbrido proposto.

**Avaliação** — métricas com intervalo de confiança por bootstrap, métricas por
classe, calibração (Brier e ECE), avaliação por fatias comportamentais, testes de
McNemar, Wilcoxon e Friedman/Nemenyi com correção de Holm e delta de Cliff, e
Ablation Study nos modos *leave-one-out* e *only-one*.

**Interpretabilidade** — valores SHAP e importância por permutação, agregadas por
grupo de atributos.

**Visualização** — tema e paleta compartilhados; distribuição de classes, nuvens
de palavras, n-grams, matriz de confusão, curvas ROC e PR, calibração, evolução
temporal do sentimento, ritmo circadiano, projeção UMAP/t-SNE, rede de
similaridade, SHAP e diagrama de diferença crítica.

**IA responsável** — geração automática de Model Card (Mitchell et al., 2019) e
Datasheet for Datasets (Gebru et al., 2021).

**Privacidade (LGPD)** — pseudonimização SHA-256 com salt na ingestão, remoção de
PII do texto, filtro de redação em todos os handlers de logging, processamento de
LLM 100% local e barreira ética que bloqueia a coleta sem aprovação CEP/CONEP.

**Testes** — unitários, baseados em propriedades (`hypothesis`), comportamentais
(invariâncias direcionais), de regressão de métrica e de integração. Todos os
dados usados nos testes são sintéticos.

**Infraestrutura** — MLflow para rastreamento, `Makefile` com alvos por etapa,
workflows de CI/CD, documentação MkDocs Material e dashboard Streamlit.

[Não publicado]: https://github.com/luanfreitas5/mental-health-nlp-ptbr/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/luanfreitas5/mental-health-nlp-ptbr/releases/tag/v0.1.0

## v0.2.0 (2026-07-31)

### Feat

- **project**: Implementar um pipeline de detecção de saúde mental por meio de PNL (Processamento de Linguagem Natural)
- **dashboard**: Adicionar aplicativo de inspeção de resultados do Streamlit
- **config**: Adiciona uma estrutura de configuração abrangente e modelos de coleta de dados.

### Fix

- **types**: converter tipos de retorno da biblioteca

### Refactor

- usar a sintaxe de união PEP 604 para verificação de tipos
