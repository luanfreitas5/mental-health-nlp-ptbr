# Design — Performance das etapas 2–6 do pipeline

Data: 2026-08-24

## Contexto e motivação

O pedido original listava 4 mudanças de performance para as etapas 2–6:

1. Paralelizar o loop por usuário (multiprocessing) nas etapas 2 e 6.
2. Gravar em lotes de 50–200 usuários em vez de 1 parquet por usuário, nas
   etapas 2 e 6.
3. Perfilar (scalene/memray) as etapas 2, 5 e 6 antes de otimizar.
4. Migrar pandas → polars nas operações vetorizadas de texto/features.

Exploração do código mostrou dois achados que mudam o desenho literal do
pedido:

- **`pandas` não é usado em `src/`** — é dependência transitiva só do
  seaborn/sklearn. O item 4 foi reinterpretado (aprovado pelo usuário) como
  **vetorizar as chamadas `.map_elements()`** (UDF Python linha-a-linha, o
  equivalente polars do `pd.apply`, com o mesmo custo), que são o real
  análogo do padrão "pandas lento" no código atual em polars.
- **As funções de negócio das etapas 2 e 6 (`run_preprocessing`,
  `build_user_features_raw`) já operam corretamente sobre múltiplos usuários
  de uma vez** — usam `group_by(user_id)` internamente em todos os pontos que
  agregam por usuário (confirmado em `preprocessing/cleaning.py`,
  `features/builder.py` e nos 6 construtores de grupo de
  `src/features/*.py`). Isso significa que os itens 1 e 2 não são
  independentes: processar em lote (ler N usuários → uma chamada de negócio
  sobre o lote → gravar um parquet por lote) já vetoriza o trabalho *entre*
  usuários, sem qualquer mudança de assinatura nessas funções. A
  paralelização então distribui **lotes** entre processos, não usuários
  individuais.

## Decisões já validadas com o usuário

- Item 4 reinterpretado como vetorização de `map_elements` (não há migração
  pandas→polars a fazer).
- Sequenciamento: perfilar primeiro (Fase 0), decidir prioridade/escopo das
  Fases 1 e 2 com base nos achados.
- Retomada em diretórios em lote: sem manifesto novo — `list_collected_users`
  passa a agregar `user_id` via `pl.scan_parquet(dir/*.parquet).select(...).unique()`
  quando o diretório é particionado por lote em vez de por usuário.
- `memray` será adicionado como dependência dev, ao lado do `scalene` já
  existente.

## Fase 0 — Profiling

**Objetivo:** medir onde tempo e memória realmente vão nas etapas 2, 5 e 6,
antes de decidir o desenho fino das Fases 1–2.

- Adicionar `memray` a `pyproject.toml` (grupo de dependências dev).
- Criar um script de profiling isolado (`scripts/profile_stage.py`) que:
  - recebe o nome da etapa (`preprocess`, `embed`, `features`) e um
    `--limit-users` para rodar sobre uma amostra realista (sugestão inicial:
    200–500 usuários já coletados/rotulados, ajustável);
  - roda a etapa via o mesmo `StageContext`/`run()` usado pelo pipeline real
    (não reimplementa lógica), para que o profile reflita o código de
    produção;
  - é instrumentado para rodar sob `scalene` (CPU + memória) e, para as
    etapas 2 e 6, também sob `memray` (perfil de alocação, já que essas
    etapas materializam DataFrames grandes por lote/usuário);
  - grava a saída em `reports/profiling/<etapa>_<timestamp>/` (HTML do
    scalene, `.bin`/flamegraph do memray).
- Rodar o profiling nas etapas 2, 5 e 6 e produzir um resumo curto (Markdown,
  no mesmo diretório) respondendo: o custo domina em I/O de parquet, em
  `map_elements`, no overhead fixo de chamada por usuário, ou em outra coisa
  (ex.: tokenização spaCy, TF-IDF, inferência do encoder)?
- **Critério de saída da fase:** o resumo aponta, para cada etapa, se vale a
  pena prosseguir com a Fase 1 (vetorização), a Fase 2 (lote +
  multiprocessing), as duas, ou nenhuma — a Fase 0 pode reduzir o escopo das
  fases seguintes se o gargalo real for outro (ex.: a etapa 5 já é dominada
  pela GPU e não precisa de nenhuma das duas mudanças, confirmando a decisão
  de design original de excluí-la dos itens 1–2).

## Fase 1 — Vetorização (`map_elements` → expressões polars nativas)

**Escopo confirmado** (todas as ocorrências de `map_elements`/`.apply` em
`src/`, restrito ao que toca as etapas 2 e 6 — os demais usos, fora de
escopo, estão listados em "Não-objetivos"):

- `src/preprocessing/pipeline.py::apply_text_processing` — hoje aplica
  `normalize_text`/`clean_text` (funções Python puras de
  `src/preprocessing/text.py`) via `map_elements`. Reescrever como cadeias de
  expressões `pl.col(...).str.*` (`replace_all`, `to_lowercase`,
  `strip_chars`, `split`, etc.), preservando a ordem de operações documentada
  (e-mail antes de menção, telefone antes de número, dedupe antes de
  normalizar). Onde uma operação não tiver equivalente vetorizado direto no
  polars (ex.: colapso de caracteres repetidos com contagem variável,
  desempacotamento de hashtag preservando grupo de captura), avaliar
  `str.replace_all` com regex compatível antes de manter `map_elements` como
  fallback pontual — o objetivo é eliminar o máximo de UDFs linha-a-linha,
  não zerar 100% deles a qualquer custo de legibilidade.
- `src/features/linguistic.py::_annotate_lexicon_hits` — `lexicon.contains`/
  `lexicon.count` via `map_elements`. Reescrever como `str.contains` (regex
  alternada dos termos do léxico) e `str.count_matches`, eliminando a UDF por
  linha.
- `src/labeling/weak_supervision.py` (linhas ~98, ~101, ~170) — mesmo padrão
  de `lexicon.contains`/`.count`. **Fora do escopo direto** (é etapa 3, não
  2/6), mas como usa a mesma classe de léxico (`utils.lexicons`), a
  vetorização de `linguistic.py` deve ser feita na classe/utilitário
  compartilhado para não duplicar duas implementações divergentes — o reuso
  em `weak_supervision.py` é um efeito colateral bem-vindo, não um objetivo
  adicional desta fase.

**Fora de escopo desta fase:** `preprocessing/cleaning.py` (já 100%
vetorizado, sem `map_elements`), `features/temporal.py:326`,
`interpretability/*.py`, `utils/hashing.py::pseudonymize` — não pertencem às
etapas 2/6 ou têm volume de chamadas irrelevante (uma vez por execução, não
por tweet).

**Critério de aceite:** testes existentes de `tests/test_preprocessing.py` e
`tests/test_features.py` continuam passando sem alteração de asserção —
mudança de implementação, não de comportamento. Novo teste de regressão de
performance é opcional e não bloqueante (vetorização já é validada por
corretude; ganho de performance é avaliado via re-profiling, não via teste).

## Fase 2 — Processamento em lote + multiprocessing (etapas 2 e 6)

**Modelo unificado:** os pendentes (`select_pending_users`) são fatiados em
*chunks* de tamanho configurável (`batch_size`, 50–200, default a calibrar
pela Fase 0). Cada chunk é processado por um worker de um
`concurrent.futures.ProcessPoolExecutor` (start method `spawn` — obrigatório
no Windows, também mais seguro no Linux/macOS com bibliotecas nativas como
spaCy/torch carregadas no processo pai):

1. **Leitura em lote**:
   - Etapa 2: `data.reader.read_user_histories(directory, user_ids=chunk)` —
     já existe, já lê N arquivos, deduplica por `tweet_id` e ordena por
     `[user_id, created_at]`. Não precisa de função nova.
   - Etapa 6: leitura por usuário dentro do chunk via
     `read_user_partition` em loop (mantém leitura individual — tweets
     rotulados podem ser grandes; concatenar após ler é equivalente a um
     `read_partitioned` restrito ao chunk) **ou** nova função
     `read_user_partitions(directory, user_ids)` análoga a
     `read_user_histories`, para paralelismo de leitura. Decisão de
     implementação, não de arquitetura — ambas preservam a mesma saída.
2. **Processamento em lote, sem mudança de assinatura**:
   - Etapa 2: `run_preprocessing(chunk_frame, config, allow_empty=True)` —
     já aceita múltiplos usuários (`group_by` interno).
   - Etapa 6: `build_user_features_raw(chunk_frame, config.features,
     metadata=..., psychological_scores=...)` — já aceita múltiplos
     usuários; `metadata`/`psychological_scores` precisam ser filtrados (ou
     passados inteiros, já que os `join`s internos são `left`/por
     `user_id`) para o chunk antes da chamada.
3. **Escrita em lote**: nova função `data.writer.write_batch_partition(frame,
   directory, batch_id)` — mesma escrita atômica de `write_parquet`, nome de
   arquivo determinístico (`batch_<indice_zero_padded>.parquet` — sequencial
   e reprodutível, não hash, para facilitar depuração e ordenação no
   diretório).
4. **Retomada**: nova função `data.reader.list_collected_users_batched(directory)`
   = `pl.scan_parquet(directory/"*.parquet").select("user_id").unique().collect()["user_id"].to_list()`.
   Usada no lugar de `list_collected_users` (que assume 1 arquivo = 1
   usuário) especificamente para os diretórios que passam a ser gravados em
   lote (`tweets_clean`, `user_features_raw`). Diretórios que continuam
   particionados por usuário (`tweets_labeled`, `psychological_scores`,
   cache de embeddings) continuam usando `list_collected_users` normalmente
   — mudança é local às etapas 2 e 6, não ao módulo `data.reader` como um
   todo.
5. **Progresso**: `rich.progress` não é seguro entre processos. A barra passa
   a avançar por **lote concluído** via
   `concurrent.futures.as_completed(futures)`, uma unidade a cada 50–200
   usuários em vez de uma a cada usuário — perda de granularidade aceita
   como troca pela paralelização real.
6. **Erros**: preserva o comportamento atual (uma exceção não tratada dentro
   do processamento de um chunk propaga e derruba a etapa) — não há hoje
   isolamento por usuário dentro de `run_preprocessing`/
   `build_user_features_raw`, e este trabalho não adiciona um; é uma
   mudança de escopo maior, não pedida.

**Configuração nova** (`configs/preprocessing.yaml` e `configs/features.yaml`):
`batch_size` (default a definir pela Fase 0, entre 50 e 200) e
`max_workers` (default: `os.process_cpu_count()` ou config explícita).

**Fora de escopo:** etapas 3 (`label`, depende de estatística populacional
não decomponível por lote independente — a parte por tweet já é
incremental, mas rotular por usuário exige o acumulado inteiro), 4 (`psych`,
I/O-bound via threads, não CPU-bound) e 5 (`embed`, GPU-bound, cache já
incremental por usuário — só entra no profiling da Fase 0, não ganha
multiprocessing/lote).

## Testes e validação

- Fase 1: `tests/test_preprocessing.py`, `tests/test_features.py` (e
  `tests/test_labeling*.py` se a vetorização de léxico for reusada por
  `weak_supervision.py`) devem passar sem alteração de asserções.
- Fase 2: testes novos para `write_batch_partition` e
  `list_collected_users_batched` em `tests/test_data_writer.py` /
  `tests/test_data_reader.py` (nomes exatos a confirmar na fase de
  implementação); teste de integração leve confirmando que processar N
  usuários em 1 chunk produz o mesmo resultado (linha a linha) que processar
  os mesmos N usuários um a um com o código atual — é o teste de
  equivalência que sustenta a mudança de granularidade.
- `make lint`/`make test` (ou equivalente do projeto) antes de cada commit,
  conforme padrão do repositório.

## Riscos e limitações conhecidas

- `ProcessPoolExecutor` com `spawn` reimporta o módulo em cada worker —
  custo de startup por processo (mitigado por processar lotes grandes por
  worker, não 1 usuário por worker).
- A granularidade de retomada piora de "1 usuário" para "1 lote": uma
  interrupção no meio de um lote perde o progresso desse lote inteiro (até
  `batch_size` usuários), não só o usuário em processamento. Troca aceita
  pelo ganho de throughput; não mitigada nesta fase (mitigação possível —
  checkpoint intra-lote — fica registrada aqui como ideia não implementada,
  não como TODO pendente).
- Mudar `tweets_clean` e `user_features_raw` de "1 arquivo por usuário" para
  "1 arquivo por lote" é uma mudança de formato em disco: dados já
  processados por execuções anteriores (arquivos antigos, 1 por usuário)
  continuam legíveis por `list_collected_users_batched` (ela só agrega
  `user_id`, não assume 1 usuário por arquivo), então não é necessário
  reprocessar do zero — mas o `.gitignore`/documentação de `data/interim/`
  deve mencionar que o formato de partição mudou.

## Não-objetivos (reafirmados)

- Nenhuma mudança nos schemas de dados (`RawTweetSchema`, `CleanTweetSchema`,
  schema de features) — só em como os dados chegam até eles.
- Nenhuma mudança nas etapas 3, 4 e 5 além do profiling da etapa 5 na Fase 0.
- Nenhum isolamento de erro por usuário novo dentro de um lote.
