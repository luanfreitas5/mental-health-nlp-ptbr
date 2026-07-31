# Arquitetura

Este documento registra as decisões de projeto e **por que** foram tomadas. Cada
uma foi um trade-off; a alternativa descartada está registrada junto.

## Princípio organizador: etapas acopladas por arquivos

Cada etapa lê artefatos do disco e grava outros. Nenhuma passa objetos em memória
para a seguinte.

**Por quê.** A coleta leva dias; o fine-tuning, horas. Um pipeline monolítico
exigiria reexecutar tudo a cada ajuste. Com acoplamento por arquivos, é possível
reexecutar só a etapa que mudou, inspecionar qualquer resultado intermediário e
retomar uma execução interrompida.

**Custo aceito.** Serialização e desserialização repetidas. Irrelevante diante
do custo das etapas de modelagem.

## Estrutura de importação

`src/` é a **raiz de importação**: os módulos são importados sem o prefixo
`src.` (`from config.settings import load_config`), conforme o padrão do projeto.

Duas consequências práticas:

1. `pyproject.toml` define `[tool.uv] package = false` — o projeto não é
   instalado no `site-packages`, o que exigiria o prefixo `src.`.
2. Nenhum pacote de topo pode ter nome de módulo da biblioteca padrão. Por isso
   não existem `src/io/` nem `src/logging/`: sombreariam `io` e `logging` para
   todo o processo. A configuração de logging vive em `src/config/logging.py`,
   que é aninhado e portanto seguro.

## Configuração validada no *startup*

Todos os YAMLs de `configs/` têm um modelo Pydantic correspondente em
`src/config/settings.py`, com `extra="forbid"`.

**Por quê.** Uma chave com nome errado seria silenciosamente ignorada e o
pipeline rodaria com o valor padrão — o pior tipo de bug, porque produz
resultados plausíveis e errados. Com `forbid`, um erro de digitação derruba a
execução no primeiro segundo.

As regras de negócio também são validadas: os pesos das fontes de rotulação
precisam somar 1, a janela de coleta precisa ser cronologicamente válida,
`test_size + val_size` não pode consumir o conjunto inteiro.

## Contratos de dados nas fronteiras

Schemas `pandera` validam entrada e saída de cada estágio.

**Por quê.** A corrupção mais cara deste projeto é silenciosa: uma coluna de data
que virou string, um score fora de `[0, 1]`, um `user_id` nulo. Sem validação na
fronteira, o erro só aparece como métrica estranha, horas depois, e a
investigação começa do zero.

Todos usam `strict=True`: coluna inesperada quase sempre significa que duas
fontes foram unidas por engano.

## Pseudonimização na ingestão

O `user_id` é convertido em SHA-256 com salt **dentro do coletor**, antes de
qualquer gravação.

**Alternativa descartada:** coletar os identificadores reais e anonimizar depois.
Isso criaria uma janela em que dados identificáveis existem em disco — e essa
janela é exatamente onde vazamentos acontecem.

O salt é obrigatório e vem do `.env`. Sem ele, o hash seria reversível por força
bruta sobre o espaço público de handles.

## Duas colunas de texto

`text_normalized` (PII removida, semântica preservada) e `text_clean`
(agressivamente reduzido).

**Por quê.** Transformers extraem sinal de caixa, pontuação e emoji — foram
pré-treinados com eles. TF-IDF e contagens lexicais, ao contrário, só ganham
esparsidade com essa variação. Um único texto forçaria escolher qual dos dois
degradar.

## Modelo híbrido: PCA assimétrico

Os embeddings passam por PCA antes da concatenação; os atributos estruturados
entram inteiros.

**Por quê.** Com ~1.500 dimensões de embedding contra algumas dezenas de
atributos estruturados, as árvores escolheriam quase sempre uma dimensão
semântica — não por ser mais informativa, mas por haver muito mais candidatas.
O desequilíbrio inviabilizaria o Ablation Study, que é justamente o experimento
que testa H2, H3 e H4.

**Custo aceito.** As componentes do PCA não são interpretáveis diretamente. Por
isso aparecem como `sem_pca_*` nas figuras SHAP, para impedir a leitura
equivocada de que correspondem a termos específicos.

## BiLSTM sobre sequências de embeddings

A rede recorrente consome a **sequência cronológica de embeddings de tweets**,
não tokens de um texto único.

**Alternativa descartada:** concatenar todos os tweets num único documento e
passá-lo por uma LSTM de tokens. Isso apagaria a fronteira entre publicações e a
ordem entre elas — exatamente a informação que a abordagem centrada no usuário
quer explorar.

## Transformer hierárquico em duas etapas

O BERTimbau é ajustado no nível do **tweet**, herdando o rótulo do usuário, e as
probabilidades são agregadas por **média**.

**Alternativa descartada:** um Transformer hierárquico consumindo o histórico
inteiro. Exigiria contexto muito maior que 512 tokens e um orçamento de GPU
incompatível com o prazo do mestrado.

**Por que média e não voto majoritário.** O rótulo herdado é ruidoso: nem todo
tweet de um usuário com depressão expressa depressão. A média das probabilidades
é muito mais robusta a rótulos individuais ruidosos que a contagem de votos.

## Particionamento agrupado por usuário

A partição é atribuída ao usuário, nunca ao tweet, e a ausência de interseção é
verificada explicitamente em dois lugares.

**Por quê.** É o vazamento mais grave e mais fácil de cometer neste projeto. Com
tweets da mesma pessoa nas duas partições, o modelo reconheceria o autor em vez
do sinal clínico, e a métrica de teste ficaria inflada de forma indetectável por
inspeção de código.

## TF-IDF dentro do `Pipeline`

Os n-grams **não** entram na matriz pré-calculada; o vetorizador vive dentro do
`Pipeline` do scikit-learn.

**Por quê.** Vocabulário e pesos IDF precisam ser estimados só no treino. Ajustar
sobre o dataset inteiro antes da divisão faria termos exclusivos do teste
influenciarem a representação — um vazamento sutil, invisível em revisão de
código.

O mesmo vale para o escalonamento e para o PCA do modelo híbrido.

**Exceção conhecida.** A imputação de ausentes em `features.builder` usa a
mediana de todo o conjunto. Para as features estruturais deste projeto isso é
aceitável e amplamente praticado, mas é, a rigor, um vazamento leve de
estatística descritiva. Mover a imputação para dentro do `Pipeline` eliminaria
o problema ao custo de perder os indicadores de ausência calculados antes do
split — a decisão está documentada aqui em vez de escondida.

## Indicadores de ausência

Antes de imputar, cria-se `<coluna>_is_missing`.

**Por quê.** A ausência aqui **não é aleatória**: uma tendência temporal só falta
quando o histórico é curto demais para estimá-la. Imputar em silêncio jogaria
fora essa informação; o indicador deixa o modelo aprender que a ausência
significa algo.

## Interface única de modelo

Todas as famílias implementam `BaseUserClassifier` e consomem `UserDataset`.

**Por quê.** Sem isso, o avaliador precisaria de um ramo condicional por modelo,
e qualquer diferença de tratamento entre eles tornaria a comparação injusta. O
`UserDataset` carrega as várias representações do mesmo conjunto de usuários
(matriz tabular, textos, sequências) para que o pipeline não precise saber quem
consome o quê.

## Ablação em dois modos

*Leave-one-out* mede contribuição marginal; *only-one* mede contribuição
absoluta.

**Por quê.** Só o primeiro leva a conclusões erradas com frequência. Grupos
correlacionados — emocional e psicológico, por exemplo — têm contribuição
marginal quase nula porque um substitui o outro. Um *leave-one-out* isolado
sugeriria que ambos são dispensáveis, quando remover os dois juntos derrubaria o
desempenho.

## Correção para múltiplas comparações

Comparando oito modelos par a par são 28 testes. Ao nível de 5%, esperar-se-ia
ao menos um "significativo" por puro acaso. A correção de Holm é aplicada por
padrão, e o tamanho de efeito (delta de Cliff) acompanha cada p-valor — o
p-valor responde "a diferença é real?", não "a diferença importa?".

## Degradação em vez de interrupção

Dependências opcionais ausentes (spaCy, Ollama, `wordcloud`, `umap-learn`)
produzem aviso e caminho alternativo, não exceção.

**Por quê.** Interromper uma execução de horas por causa de uma nuvem de palavras
seria desproporcional. As dependências **essenciais**, ao contrário, falham alto
e cedo.

**Exceção deliberada:** o modelo do Ollama não é baixado automaticamente
(`auto_pull: false`). Baixar vários GB no meio de um pipeline longo, sem que
ninguém perceba, é pior do que interromper com instrução clara.

## Barreira ética na coleta

A etapa `collect` verifica `ETHICS_APPROVAL_ID` antes de qualquer requisição.

**Por quê.** Coletar publicações de pessoas em sofrimento psíquico sem aprovação
prévia é violação de protocolo de pesquisa. O código não deve tornar isso fácil.
A barreira é contornável (`--allow-collection-without-ethics`) para teste
técnico, e o uso fica registrado no log com aviso explícito.

## Redação de PII no handler de logging

O filtro é anexado a **todos** os handlers, não aplicado nas chamadas.

**Por quê.** Depender da disciplina de quem escreve `logger.info` não é garantia.
Um log com PII de pessoas em sofrimento psíquico é exatamente o vazamento que o
projeto não pode ter, e a proteção precisa estar no lugar por onde toda mensagem
passa obrigatoriamente.
