# Setup

## Pré-requisitos

| Requisito | Versão | Necessário para |
|---|---|---|
| Python | 3.10+ | Tudo |
| [uv](https://docs.astral.sh/uv/) | recente | Gerenciamento de dependências |
| GPU NVIDIA + CUDA 12 | opcional | Transformers e embeddings (viável em CPU, porém lento) |
| [Ollama](https://ollama.com) | opcional | Atributos psicológicos e o LLM classificador |
| Contas do X/Twitter | opcional | Apenas para a etapa de coleta |

## Instalação

```bash
git clone https://github.com/luanfreitas5/mental-health-nlp-ptbr.git
cd mental-health-nlp-ptbr

make install   # dependências principais + ferramentas de desenvolvimento
make hooks     # instala os hooks do pre-commit
```

### Dependências opcionais

Os extras são separados porque cada um resolve um problema distinto — e porque
o PyTorch sozinho pesa vários GB. Instale só o que for usar.

=== "LLM e Transformers"

    ```bash
    make install-llm
    ```

    PyTorch, Transformers, Accelerate e o cliente do Ollama. Necessário para as
    etapas `label` (encoders de sentimento e emoção), `embed` (embeddings
    semânticos), `psych` (vetor psicológico) e para os modelos BERTimbau,
    BiLSTM e LLM.

=== "Coleta"

    ```bash
    make install-collect
    ```

    `twscrape`. Isolado porque a coleta só roda com aprovação CEP/CONEP e contas
    configuradas — o restante do pipeline não depende dele.

=== "PLN (lematização)"

    ```bash
    make install-nlp   # instala spaCy e baixa pt_core_news_sm
    ```

    Sem o modelo do spaCy, a tokenização cai num fallback por regex e a
    lematização é desativada, com aviso no log. O pipeline continua funcional,
    apenas com features linguísticas um pouco mais ruidosas.

=== "Figuras opcionais"

    ```bash
    uv sync --extra viz --dev
    ```

    `wordcloud`, `networkx` e `umap-learn`. Ausentes, as figuras
    correspondentes são puladas e as demais continuam sendo geradas.

## Configuração dos segredos

```bash
cp .env.example .env
```

### `PSEUDONYMIZATION_SALT` (obrigatório)

Salt do hash que converte identificadores diretos em pseudônimos.

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

!!! danger "Não troque o salt no meio da coleta"
    A pseudonimização precisa ser determinística: o mesmo usuário tem de receber
    o mesmo pseudônimo em coletas diferentes, senão o histórico longitudinal se
    fragmenta em dois usuários que o pipeline tratará como pessoas distintas.

!!! warning "Por que o salt existe"
    Sem ele, o hash seria reversível por força bruta: o espaço de handles do
    X/Twitter é público e enumerável, bastaria calcular o hash de cada um e
    comparar. O salt é o que torna a pseudonimização irreversível na prática.

### `ETHICS_APPROVAL_ID` (obrigatório para coletar)

Número do CAAE da aprovação do Comitê de Ética em Pesquisa. A etapa `collect` é
bloqueada enquanto estiver vazio. Ver [Ética e LGPD](ethics.md).

## Verificação da instalação

```bash
make status   # situação dos artefatos de cada etapa
make smoke    # verificações rápidas
make test     # suíte completa com cobertura
```

Saída esperada de `make status` num projeto recém-clonado — todos os artefatos
ainda por produzir:

```
[--] seed_tweets           etapa=collect       0.00 MB
[--] user_metadata         etapa=collect       0.00 MB
[--] tweets_clean          etapa=preprocess    0.00 MB
...
```

## Ollama (opcional)

Necessário para a etapa `psych` e para o modelo LLM da comparação principal.

```bash
# Instale o Ollama a partir de https://ollama.com
ollama serve         # sobe o servidor local
make ollama-pull     # baixa o llama3.2
```

!!! info "Por que local"
    Os textos são dados pessoais sensíveis de saúde. Enviá-los a uma API remota
    configuraria transferência internacional de dado sensível sem base legal
    adequada (LGPD, arts. 11 e 33). Todo o processamento fica na máquina.

O download **não** é automático: `configs/llm.yaml` define `auto_pull: false`
de propósito, para que o pipeline não baixe vários GB no meio de uma execução
longa sem que ninguém perceba.

## Estrutura de diretórios

Criada automaticamente no primeiro `python src/main.py`:

```
data/
├── raw/          # dados originais — nunca modificados
├── external/     # fontes externas (contas do twscrape, rótulos manuais)
├── interim/      # intermediários (tweets limpos, rotulados, embeddings)
└── processed/    # matriz final e partições
models/           # modelos treinados e checkpoints
reports/          # figuras, métricas, model cards, datasheets
logs/             # log_AAAA-MM-DD.log, com rotação diária
```

## Solução de problemas

??? question "`ConfigValidationError` ao iniciar"
    A configuração é validada no *startup*. A mensagem indica o arquivo, a chave
    e a regra violada. É intencional: melhor falhar imediatamente do que
    corromper uma execução de horas.

??? question "`MissingDependencyError: PyTorch não está instalado`"
    Rode `make install-llm`. As etapas que dependem de Transformers são as
    únicas afetadas; as demais continuam funcionando.

??? question "`LLMUnavailableError: Servidor Ollama inacessível`"
    Verifique se `ollama serve` está em execução e se o host em
    `configs/llm.yaml` está correto. Sem Ollama, a etapa `psych` é pulada com
    aviso e o grupo `psychological` fica ausente da matriz.

??? question "Modelo spaCy não encontrado"
    Rode `make spacy-model`. Sem ele, a tokenização usa o fallback por regex.

??? question "CUDA não detectada"
    `python src/main.py --status` registra o ambiente no log, incluindo
    disponibilidade de GPU. Sem GPU, tudo roda em CPU — as etapas `embed` e o
    fine-tuning do BERTimbau ficam significativamente mais lentas.
