# Performance das Etapas 2-6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Perfilar as etapas 2, 5 e 6 do pipeline, eliminar as chamadas `map_elements` (UDF Python linha-a-linha) das etapas 2 e 6 por expressões polars nativas, e paralelizar + agrupar em lotes o processamento por usuário das etapas 2 e 6 via `ProcessPoolExecutor`.

**Architecture:** Três fases sequenciais. Fase 0 mede onde o tempo/memória realmente vão (scalene + memray) sobre as etapas 2/5/6, sem mudar código de produção. Fase 1 reescreve `normalize_text`/`clean_text` (etapa 2) e o matching de léxico (etapa 6) como expressões `pl.Expr` vetorizadas, preservando comportamento (testes existentes continuam passando). Fase 2 unifica os pedidos de "multiprocessing" e "gravar em lotes": os usuários pendentes são fatiados em lotes de `batch_size`, cada lote é processado **inteiro** por uma chamada de `run_preprocessing`/`build_user_features_raw` (que já aceitam múltiplos usuários via `group_by(user_id)` interno) dentro de um worker de processo, e gravado em um único parquet — a vetorização por lote e o paralelismo entre lotes se somam.

**Tech Stack:** Python 3.10+, polars, `concurrent.futures.ProcessPoolExecutor` (spawn), scalene, memray, pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-pipeline-performance-design.md`

## Global Constraints

- Código em inglês, docstrings/comentários/logs em pt-BR (padrão NumPy), conforme `CLAUDE.md`.
- Nenhuma mudança nos schemas `RawTweetSchema`/`CleanTweetSchema`/schema de features — só em como os dados chegam até eles.
- Nenhuma mudança nas etapas 3, 4 e 5 além do profiling da etapa 5 na Fase 0.
- `ProcessPoolExecutor` deve usar `mp_context=multiprocessing.get_context("spawn")` — único método suportado no Windows, e mais seguro em qualquer SO com spaCy/torch já carregados no processo pai.
- Toda função que roda em processo worker deve ser uma função de módulo top-level (importável), nunca um closure/lambda/método de instância — exigência do `spawn`.
- Testes existentes (`tests/test_preprocessing.py`, `tests/test_features.py`, `tests/test_data.py`) não podem regredir: mudança de implementação, não de comportamento.
- `make lint` e `make test` (ou os testes relevantes) devem passar antes de cada commit.

---

## Fase 0 — Profiling

### Task 0.1: Adicionar `memray` como dependência dev

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nada.
- Produces: `memray` instalável via `uv sync --dev`, usado nas Tasks 0.2/0.3.

- [ ] **Step 1: Adicionar a dependência**

Abrir `pyproject.toml`, localizar a linha `"scalene>=1.5",` (grupo de dependências dev) e adicionar logo abaixo:

```toml
    "memray>=1.13; sys_platform != 'win32'",  # Profiler de alocação de memória (Linux/macOS apenas)
```

`memray` não suporta Windows nativamente (depende de `ptrace`/mecanismos específicos do Linux/macOS) — o marcador `sys_platform != 'win32'` evita falha de instalação no ambiente de desenvolvimento atual (Windows) e mantém a dependência disponível em CI/produção Linux, onde o profiling de memória de fato roda.

- [ ] **Step 2: Sincronizar dependências**

Run: `uv sync --dev`
Expected: sincroniza sem erro (em Windows, `memray` é pulado pelo marcador de plataforma; em Linux/CI, é instalado).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore(deps): adicionar memray para profiling de memoria (linux/macos)"
```

### Task 0.2: Substituir o alvo `profile` do Makefile por alvos por etapa

**Files:**
- Modify: `Makefile`

**Interfaces:**
- Consumes: `$(RUN)` (já definido: `$(UV) run python src/main.py`), `--stage`, `--limit-users`, `--no-tracking` (já existem em `src/main.py`).
- Produces: `make profile-preprocess`, `make profile-embed`, `make profile-features` — cada um grava em `reports/profiling/<etapa>/`.

- [ ] **Step 1: Remover o alvo placeholder e adicionar os alvos reais**

Localizar em `Makefile`:

```makefile
profile:  ## Exemplo de profiling com scalene (ajuste o alvo)
	$(UV) run scalene src/main.py
```

Substituir por:

```makefile
PROFILE_LIMIT ?= 300

profile-preprocess:  ## Perfila a etapa 2 (scalene) sobre uma amostra de usuários pendentes
	mkdir -p reports/profiling/preprocess
	$(UV) run scalene --html --outfile reports/profiling/preprocess/scalene.html \
		-- src/main.py --stage preprocess --limit-users $(PROFILE_LIMIT) --no-tracking

profile-embed:  ## Perfila a etapa 5 (scalene) sobre uma amostra de usuários pendentes
	mkdir -p reports/profiling/embed
	$(UV) run scalene --html --outfile reports/profiling/embed/scalene.html \
		-- src/main.py --stage embed --limit-users $(PROFILE_LIMIT) --no-tracking

profile-features:  ## Perfila a etapa 6 (scalene) sobre uma amostra de usuários pendentes
	mkdir -p reports/profiling/features
	$(UV) run scalene --html --outfile reports/profiling/features/scalene.html \
		-- src/main.py --stage features --limit-users $(PROFILE_LIMIT) --no-tracking

profile-preprocess-memray:  ## Perfila a etapa 2 (memray, alocação de memória; Linux/macOS)
	mkdir -p reports/profiling/preprocess
	$(UV) run python -m memray run --force -o reports/profiling/preprocess/memray.bin \
		src/main.py --stage preprocess --limit-users $(PROFILE_LIMIT) --no-tracking
	$(UV) run python -m memray flamegraph --force -o reports/profiling/preprocess/memray.html \
		reports/profiling/preprocess/memray.bin

profile-features-memray:  ## Perfila a etapa 6 (memray, alocação de memória; Linux/macOS)
	mkdir -p reports/profiling/features
	$(UV) run python -m memray run --force -o reports/profiling/features/memray.bin \
		src/main.py --stage features --limit-users $(PROFILE_LIMIT) --no-tracking
	$(UV) run python -m memray flamegraph --force -o reports/profiling/features/memray.html \
		reports/profiling/features/memray.bin

profile: profile-preprocess profile-embed profile-features  ## Roda o profiling scalene das etapas 2, 5 e 6
```

- [ ] **Step 2: Atualizar a lista `.PHONY`**

Na linha `.PHONY: ... profile clean ...`, substituir `profile` por
`profile profile-preprocess profile-embed profile-features profile-preprocess-memray profile-features-memray`.

- [ ] **Step 3: Verificar que o alvo aparece na ajuda**

Run: `make help`
Expected: lista inclui `profile-preprocess`, `profile-embed`, `profile-features` com suas descrições.

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "chore(profiling): adicionar alvos make profile-* para as etapas 2, 5 e 6"
```

### Task 0.3: Rodar o profiling e registrar os achados

**Files:**
- Create: `reports/profiling/SUMMARY.md`

**Interfaces:**
- Consumes: `make profile-preprocess`, `make profile-embed`, `make profile-features` (Task 0.2); requer que `data/interim/user_histories` (ou `tweets_clean`/`tweets_labeled`, conforme a etapa) já tenha ao menos `$(PROFILE_LIMIT)` usuários coletados/pendentes localmente.
- Produces: `reports/profiling/SUMMARY.md` — o resumo que direciona o escopo/prioridade das Fases 1 e 2 abaixo (já redigidas com o desenho mais provável, mas a serem ajustadas se o profiling revelar outro gargalo dominante).

- [ ] **Step 1: Rodar o profiling das três etapas**

```bash
make profile-preprocess PROFILE_LIMIT=300
make profile-embed PROFILE_LIMIT=300
make profile-features PROFILE_LIMIT=300
```

Se o ambiente não for Windows, rodar também:

```bash
make profile-preprocess-memray PROFILE_LIMIT=300
make profile-features-memray PROFILE_LIMIT=300
```

- [ ] **Step 2: Abrir os relatórios HTML e identificar os pontos quentes**

Abrir `reports/profiling/preprocess/scalene.html`, `reports/profiling/embed/scalene.html`,
`reports/profiling/features/scalene.html` (e os `.html` do memray, se gerados) e anotar, por
etapa: percentual de tempo em I/O vs. CPU Python vs. CPU nativo (polars/torch), as 5 linhas
de código com maior `%CPU`, e (quando houver memray) o pico de alocação e sua origem.

- [ ] **Step 3: Escrever o resumo**

Criar `reports/profiling/SUMMARY.md` com esta estrutura (preencher as reticências com os
números reais observados no Step 2 — não deixar como texto genérico):

```markdown
# Profiling das etapas 2, 5 e 6 — resumo

Data: <data da execução> · Amostra: <PROFILE_LIMIT> usuários pendentes.

## Etapa 2 (preprocess)
- Tempo total: <...>s para <N> usuários (<...>s/usuário).
- Gargalo dominante: <ex.: map_elements de normalize_text/clean_text, ou I/O de leitura
  por usuário, ou overhead de chamada polars por usuário>.
- Implicação para a Fase 1 (vetorização): <confirma / reduz escopo / não se aplica>.
- Implicação para a Fase 2 (lote + multiprocessing): <confirma / ajusta batch_size sugerido>.

## Etapa 5 (embed)
- Tempo total: <...>s para <N> usuários.
- Gargalo dominante: <esperado: inferência do encoder na GPU/CPU>.
- Ação: <nenhuma mudança planejada — confirma a decisão de excluir a etapa 5 das Fases 1/2>
  ou <achado inesperado a registrar>.

## Etapa 6 (features)
- Tempo total: <...>s para <N> usuários.
- Gargalo dominante: <...>.
- Implicação para a Fase 1: <...>.
- Implicação para a Fase 2: <...>.

## Batch size recomendado
<valor entre 50-200 escolhido com base no ponto em que o overhead fixo por chamada deixa de
dominar, observado no profiling da etapa 2/6> — usado como default em `batch_size` nas
Tasks 2.4/2.5.
```

- [ ] **Step 4: Commit**

```bash
git add reports/profiling/SUMMARY.md
git commit -m "docs(profiling): registrar achados do profiling das etapas 2, 5 e 6"
```

---

## Fase 1 — Vetorização (`map_elements` → expressões polars nativas)

### Task 1.1: Expressão polars para remoção de acentos (`strip_accents_expr`)

Pré-requisito compartilhado pelas Tasks 1.2 (stopwords) e 1.4 (léxicos): tanto a comparação de
stopword quanto o `Lexicon.contains`/`.count` comparam texto **sem acento**, independentemente
da configuração `remove_accents` (ver `preprocessing/text.py::_is_removable_stopword` e
`utils/lexicons.py::normalize_term`). Sem uma versão vetorizada disso, nenhuma das duas
vetorizações fecha.

**Files:**
- Modify: `src/preprocessing/text.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Produces: `strip_accents_expr(column: pl.Expr) -> pl.Expr` em `preprocessing/text.py`, usada pelas Tasks 1.2 e 1.4.

- [ ] **Step 1: Escrever o teste que compara a versão vetorizada com `strip_accents` escalar**

Adicionar a `tests/test_preprocessing.py`, na classe `TestFuncoesAuxiliares`:

```python
    def test_strip_accents_expr_igual_a_strip_accents_escalar(self) -> None:
        """A versão vetorizada produz o mesmo resultado, texto a texto, que a escalar."""
        textos = ["solidão", "não estou bem", "Depressão? Não.", "café com açúcar", "sem acento"]
        frame = pl.DataFrame({"texto": textos})
        vetorizado = frame.select(strip_accents_expr(pl.col("texto")).alias("out"))["out"].to_list()
        assert vetorizado == [strip_accents(texto) for texto in textos]
```

Adicionar `strip_accents_expr` ao import de `preprocessing.text` no topo do arquivo (junto com
`clean_text, contains_pii, extract_hashtags, normalize_text, strip_accents, tokenize`).

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run: `uv run pytest tests/test_preprocessing.py -k strip_accents_expr -v`
Expected: FAIL — `ImportError` ou `NameError: strip_accents_expr`.

- [ ] **Step 3: Implementar `strip_accents_expr`**

Em `src/preprocessing/text.py`, logo após `strip_accents`:

```python
def strip_accents_expr(column: pl.Expr) -> pl.Expr:
    """Versão vetorizada de :func:`strip_accents`, para uso em pipelines polars.

    Decompõe em NFD, remove os caracteres da categoria Unicode "Mn" (sinais
    diacríticos) via regex e recompõe em NFC — o mesmo algoritmo de
    :func:`strip_accents`, sem UDF Python por linha.

    Parameters
    ----------
    column : pl.Expr
        Expressão de coluna de texto.

    Returns
    -------
    pl.Expr
        Expressão sem sinais diacríticos.

    Examples
    --------
    >>> import polars as pl
    >>> pl.DataFrame({"t": ["solidão"]}).select(
    ...     strip_accents_expr(pl.col("t"))
    ... )["t"].to_list()
    ['solidao']
    """
    return column.str.normalize("NFD").str.replace_all(r"\p{Mn}", "").str.normalize("NFC")
```

Adicionar `import polars as pl` ao topo de `preprocessing/text.py` (o módulo hoje só usa
`unicodedata` — a nova função é a primeira a depender de polars neste arquivo).

- [ ] **Step 4: Rodar o teste**

Run: `uv run pytest tests/test_preprocessing.py -k strip_accents_expr -v`
Expected: PASS.

Se `str.normalize` ou `\p{Mn}` não forem suportados pela versão de polars instalada (erro de
`AttributeError` ou `ComputeError` no Step 4), a expressão acima **não** deve ser usada — trocar
`strip_accents_expr` para delegar a `map_elements(strip_accents, return_dtype=pl.Utf8)`
explicitamente, documentar no docstring que é um fallback não vetorizado por limitação da versão
de polars instalada (registrar a versão em `pyproject.toml`), e seguir para o Step 5 normalmente
— as Tasks 1.2/1.4 continuam válidas chamando `strip_accents_expr`, só sem o ganho de
performance neste ponto específico.

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing/text.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): adicionar strip_accents_expr vetorizada"
```

### Task 1.2: Vetorizar `normalize_text` em `apply_text_processing`

**Files:**
- Modify: `src/preprocessing/pipeline.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `NormalizationSection` (`config/settings.py`, já existe), padrões de
  `constants/regex.py` (já existem).
- Produces: `normalize_text_expr(column: pl.Expr, config: NormalizationSection) -> pl.Expr` em
  `preprocessing/pipeline.py`, usada por `apply_text_processing` no lugar do `map_elements`
  atual. `normalize_text` (escalar, `preprocessing/text.py`) **não é removida** — continua
  usada pelos testes de unidade existentes e por qualquer chamador que precise normalizar um
  único texto fora de um DataFrame.

- [ ] **Step 1: Escrever o teste de equivalência**

Adicionar a `tests/test_preprocessing.py` uma nova classe:

```python
class TestNormalizacaoVetorizada:
    """A versão vetorizada de normalize_text deve produzir o mesmo resultado, linha a linha."""

    def test_equivale_a_normalize_text_escalar(
        self, normalization: NormalizationSection
    ) -> None:
        textos = [
            "veja https://exemplo.com/x",
            "oi @fulano",
            "contato a.b@dominio.com",
            "dia difícil #desabafo",
            "muitooooo triste",
            "Não Estou Bem",
            "RT @alguem: texto original",
            "a\n\n  b",
            "",
        ]
        frame = pl.DataFrame({"text": textos})
        vetorizado = frame.select(
            normalize_text_expr(pl.col("text"), normalization).alias("out")
        )["out"].to_list()
        assert vetorizado == [normalize_text(texto, normalization) for texto in textos]
```

Adicionar `from preprocessing.pipeline import apply_text_processing, normalize_text_expr` (ou
ajustar o import já existente de `pipeline`, se houver) ao topo de `tests/test_preprocessing.py`.

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run: `uv run pytest tests/test_preprocessing.py -k NormalizacaoVetorizada -v`
Expected: FAIL — `ImportError` ou `NameError: normalize_text_expr`.

- [ ] **Step 3: Implementar `normalize_text_expr`**

Em `src/preprocessing/pipeline.py`, adicionar (após os imports, antes de `apply_text_processing`):

```python
from preprocessing.text import strip_accents_expr
from constants.regex import (
    CONTROL_CHARS_PATTERN,
    EMAIL_PATTERN,
    EMOJI_PATTERN,
    HASHTAG_PATTERN,
    MENTION_PATTERN,
    NUMBER_PATTERN,
    PHONE_PATTERN,
    REPEATED_CHARS_PATTERN,
    RETWEET_PATTERN,
    URL_PATTERN,
    WHITESPACE_PATTERN,
)


def normalize_text_expr(column: pl.Expr, config: NormalizationSection) -> pl.Expr:
    """Versão vetorizada de :func:`preprocessing.text.normalize_text`.

    Reproduz, como cadeia de expressões polars, a mesma ordem de operações da
    versão escalar: e-mail antes de menção, telefone antes de número (ver
    docstring de :func:`preprocessing.text.normalize_text`).

    Parameters
    ----------
    column : pl.Expr
        Expressão da coluna de texto bruto.
    config : NormalizationSection
        Seção ``normalization`` de ``configs/preprocessing.yaml``.

    Returns
    -------
    pl.Expr
        Expressão do texto normalizado.

    Examples
    --------
    >>> import polars as pl
    >>> from config.settings import NormalizationSection
    >>> pl.DataFrame({"t": ["oi @fulano"]}).select(
    ...     normalize_text_expr(pl.col("t"), NormalizationSection())
    ... )["t"].to_list()
    ['oi @user']
    """
    result = column.fill_null("").str.normalize(config.unicode_form)
    result = result.str.replace_all(RETWEET_PATTERN.pattern, "")

    if config.strip_control_chars:
        result = result.str.replace_all(CONTROL_CHARS_PATTERN.pattern, " ")

    if config.replace_emails is not None:
        result = result.str.replace_all(EMAIL_PATTERN.pattern, config.replace_emails)
    if config.replace_urls is not None:
        result = result.str.replace_all(URL_PATTERN.pattern, config.replace_urls)
    if config.replace_mentions is not None:
        result = result.str.replace_all(MENTION_PATTERN.pattern, config.replace_mentions)
    if config.replace_phone_numbers is not None:
        result = result.str.replace_all(PHONE_PATTERN.pattern, config.replace_phone_numbers)
    if config.replace_numbers is not None:
        result = result.str.replace_all(NUMBER_PATTERN.pattern, config.replace_numbers)

    if config.unpack_hashtags:
        result = result.str.replace_all(HASHTAG_PATTERN.pattern, "${1}")

    if config.collapse_repeated_chars:
        keep = config.collapse_repeated_chars
        result = result.str.replace_all(REPEATED_CHARS_PATTERN.pattern, "${1}" * keep)

    if config.demojize:
        result = result.str.replace_all(EMOJI_PATTERN.pattern, " ")

    if config.collapse_whitespace:
        result = result.str.replace_all(WHITESPACE_PATTERN.pattern, " ")

    return result.str.strip_chars()
```

Atualizar `apply_text_processing` para usar a nova expressão no lugar do `map_elements`:

```python
def apply_text_processing(frame: pl.DataFrame, config: Config) -> pl.DataFrame:
    """Cria as colunas ``text_normalized`` e ``text_clean``.
    ...  (docstring inalterada)
    """
    normalization = config.preprocessing.normalization
    cleaning = config.preprocessing.cleaning
    stopwords = load_stopwords() if cleaning.remove_stopwords else frozenset()

    return frame.with_columns(
        normalize_text_expr(pl.col(TEXT), normalization).alias(TEXT_NORMALIZED)
    ).with_columns(
        clean_text_expr(pl.col(TEXT_NORMALIZED), cleaning, stopwords).alias(TEXT_CLEAN)
    )
```

(`clean_text_expr` é implementada na Task 1.3 — este passo já deixa a chamada pronta; até a
Task 1.3 rodar, o teste desta task cobre só a coluna `text_normalized`, então rode o teste com
`clean_text_expr` temporariamente apontando para
`pl.col(TEXT_NORMALIZED).map_elements(lambda text: clean_text(text, cleaning, stopwords), return_dtype=pl.Utf8)`
— a implementação atual — para não quebrar `apply_text_processing` entre as duas tasks.)

- [ ] **Step 4: Rodar o teste**

Run: `uv run pytest tests/test_preprocessing.py -k NormalizacaoVetorizada -v`
Expected: PASS.

- [ ] **Step 5: Rodar a suíte completa de preprocessing para checar regressão**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: todos os testes passam, incluindo os de `TestNormalizacao` (que testam
`normalize_text` escalar, inalterada) e qualquer teste existente de `apply_text_processing`.

- [ ] **Step 6: Commit**

```bash
git add src/preprocessing/pipeline.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): vetorizar normalize_text em apply_text_processing"
```

### Task 1.3: Vetorizar `clean_text` em `apply_text_processing`

**Files:**
- Modify: `src/preprocessing/pipeline.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `strip_accents_expr` (Task 1.1), `CleaningSection`.
- Produces: `clean_text_expr(column: pl.Expr, config: CleaningSection, stopwords: frozenset[str]) -> pl.Expr`, substitui o `map_elements` provisório da Task 1.2.

- [ ] **Step 1: Escrever o teste de equivalência**

Adicionar a `TestNormalizacaoVetorizada` (ou uma nova `TestLimpezaVetorizada`) em
`tests/test_preprocessing.py`:

```python
class TestLimpezaVetorizada:
    """A versão vetorizada de clean_text deve produzir o mesmo resultado, linha a linha."""

    def test_equivale_a_clean_text_escalar(self, cleaning: CleaningSection) -> None:
        stopwords = frozenset({"de", "para", "estou"})
        textos = [
            "Eu não estou bem hoje!",
            "café, açúcar e solidão",
            "   ",
            "Não Vou Desistir Nunca",
            "",
        ]
        frame = pl.DataFrame({"text_normalized": textos})
        vetorizado = frame.select(
            clean_text_expr(pl.col("text_normalized"), cleaning, stopwords).alias("out")
        )["out"].to_list()
        assert vetorizado == [clean_text(texto, cleaning, stopwords) for texto in textos]
```

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run: `uv run pytest tests/test_preprocessing.py -k LimpezaVetorizada -v`
Expected: FAIL — `NameError: clean_text_expr`.

- [ ] **Step 3: Implementar `clean_text_expr`**

Em `src/preprocessing/pipeline.py`, após `normalize_text_expr`:

```python
def clean_text_expr(
    column: pl.Expr, config: CleaningSection, stopwords: frozenset[str]
) -> pl.Expr:
    """Versão vetorizada de :func:`preprocessing.text.clean_text`.

    A comparação de stopword ignora acento dos dois lados (ver docstring de
    :func:`preprocessing.text.clean_text`), por isso tanto os tokens quanto a
    lista de stopwords passam por :func:`strip_accents_expr`/
    :func:`utils.lexicons.normalize_term`.

    Parameters
    ----------
    column : pl.Expr
        Expressão da coluna de texto normalizado.
    config : CleaningSection
        Seção ``cleaning`` de ``configs/preprocessing.yaml``.
    stopwords : frozenset of str
        Stopwords a remover.

    Returns
    -------
    pl.Expr
        Expressão do texto limpo, com tokens separados por espaço.

    Examples
    --------
    >>> import polars as pl
    >>> from config.settings import CleaningSection
    >>> pl.DataFrame({"t": ["Eu não estou bem hoje!"]}).select(
    ...     clean_text_expr(pl.col("t"), CleaningSection(), frozenset({"estou"}))
    ... )["t"].to_list()
    ['eu não bem hoje']
    """
    result = column.fill_null("")
    if config.lowercase:
        result = result.str.to_lowercase()
    if config.remove_emojis:
        result = result.str.replace_all(EMOJI_PATTERN.pattern, " ")
    if config.remove_accents:
        result = strip_accents_expr(result)
    if config.remove_punctuation:
        result = result.str.replace_all(PUNCTUATION_PATTERN.pattern, " ")

    result = result.str.replace_all(WHITESPACE_PATTERN.pattern, " ").str.strip_chars()

    whitelist = {term.lower() for term in config.stopwords_whitelist}
    normalized_stopwords = {strip_accents(stopword) for stopword in stopwords}

    tokens = result.str.split(" ")
    kept = tokens.list.eval(
        pl.element().filter(
            (pl.element().str.len_chars() >= config.min_token_length)
            & (
                ~config.remove_stopwords
                | pl.element().is_in(list(whitelist))
                | ~strip_accents_expr(pl.element()).is_in(list(normalized_stopwords))
            )
        )
    )
    return kept.list.join(" ")
```

Adicionar `PUNCTUATION_PATTERN` ao import de `constants.regex` já presente no arquivo (Task 1.2
já importou os demais padrões) e `strip_accents` (função escalar, usada só para pré-normalizar
`normalized_stopwords`, uma coleção pequena — não é chamada por linha) ao import de
`preprocessing.text`.

Atualizar a chamada em `apply_text_processing` para usar `clean_text_expr` de fato, substituindo
o `map_elements` provisório deixado na Task 1.2:

```python
    return frame.with_columns(
        normalize_text_expr(pl.col(TEXT), normalization).alias(TEXT_NORMALIZED)
    ).with_columns(
        clean_text_expr(pl.col(TEXT_NORMALIZED), cleaning, stopwords).alias(TEXT_CLEAN)
    )
```

- [ ] **Step 4: Rodar o teste**

Run: `uv run pytest tests/test_preprocessing.py -k LimpezaVetorizada -v`
Expected: PASS. Se `list.eval` com `pl.element()` dentro de `filter` não aceitar a combinação de
`is_in` com uma lista Python grande (`normalized_stopwords`) por limitação de versão, trocar
`.is_in(list(...))` por um `pl.Series` construído uma vez fora do laço (`pl.lit(pl.Series(list(normalized_stopwords)))`) — mesma semântica, forma alternativa de passar a coleção ao operador.

- [ ] **Step 5: Rodar a suíte completa e o teste de equivalência ponta a ponta**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: todos passam, incluindo `TestLimpeza` (versão escalar, inalterada).

- [ ] **Step 6: Commit**

```bash
git add src/preprocessing/pipeline.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): vetorizar clean_text em apply_text_processing"
```

### Task 1.4: Vetorizar o matching de léxico em `features/linguistic.py`

**Files:**
- Modify: `src/utils/lexicons.py`
- Modify: `src/features/linguistic.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `Lexicon.pattern` (já existe, `re.Pattern` compilado por `build_term_pattern`),
  `strip_accents_expr` (Task 1.1).
- Produces: `Lexicon.contains_expr(column: pl.Expr) -> pl.Expr` e
  `Lexicon.count_expr(column: pl.Expr) -> pl.Expr` em `utils/lexicons.py`, usadas por
  `_annotate_lexicon_hits` no lugar dos dois `map_elements`.

- [ ] **Step 1: Escrever o teste de equivalência dos novos métodos de `Lexicon`**

Adicionar a `tests/test_features.py` (ou a um novo `tests/test_lexicons.py`, se o projeto
preferir isolar — usar `tests/test_features.py` por padrão de proximidade ao consumidor):

```python
def test_lexicon_contains_expr_e_count_expr_equivalem_aos_metodos_escalares() -> None:
    """As versões vetorizadas de Lexicon.contains/.count batem com as escalares, linha a linha."""
    lexicons = load_lexicons()
    lexicon = lexicons["loneliness"]
    textos = [
        "me sinto sozinho e sozinho de novo",
        "hoje foi um dia tranquilo",
        "SOLIDÃO",
        "",
    ]
    frame = pl.DataFrame({"texto": textos})
    resultado = frame.select(
        lexicon.contains_expr(pl.col("texto")).alias("has"),
        lexicon.count_expr(pl.col("texto")).alias("hits"),
    )
    assert resultado["has"].to_list() == [lexicon.contains(texto) for texto in textos]
    assert resultado["hits"].to_list() == [lexicon.count(texto) for texto in textos]
```

Adicionar `from utils.lexicons import load_lexicons` ao topo de `tests/test_features.py` (se
ainda não importado).

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run: `uv run pytest tests/test_features.py -k lexicon_contains_expr -v`
Expected: FAIL — `AttributeError: 'Lexicon' object has no attribute 'contains_expr'`.

- [ ] **Step 3: Implementar `contains_expr`/`count_expr` em `Lexicon`**

Em `src/utils/lexicons.py`, adicionar `import polars as pl` ao topo e, na classe `Lexicon`, após
`contains`:

```python
    def contains_expr(self, column: pl.Expr) -> pl.Expr:
        """Versão vetorizada de :meth:`contains`, para uso em pipelines polars.

        Parameters
        ----------
        column : pl.Expr
            Expressão de coluna de texto.

        Returns
        -------
        pl.Expr
            Expressão booleana: presença de ao menos um termo do léxico.

        Examples
        --------
        >>> import polars as pl
        >>> lex = load_lexicons()["death"]
        >>> pl.DataFrame({"t": ["quero apenas dormir"]}).select(
        ...     lex.contains_expr(pl.col("t"))
        ... )["t"].to_list()
        [False]
        """
        from preprocessing.text import strip_accents_expr

        return strip_accents_expr(column.str.to_lowercase()).str.contains(self.pattern.pattern)

    def count_expr(self, column: pl.Expr) -> pl.Expr:
        """Versão vetorizada de :meth:`count`, para uso em pipelines polars.

        Parameters
        ----------
        column : pl.Expr
            Expressão de coluna de texto.

        Returns
        -------
        pl.Expr
            Expressão inteira: número de ocorrências de termos do léxico.

        Examples
        --------
        >>> import polars as pl
        >>> lex = load_lexicons()["loneliness"]
        >>> pl.DataFrame({"t": ["sozinho e sozinho de novo"]}).select(
        ...     lex.count_expr(pl.col("t"))
        ... )["t"].to_list()
        [2]
        """
        from preprocessing.text import strip_accents_expr

        normalized = strip_accents_expr(column.str.to_lowercase())
        return normalized.str.count_matches(self.pattern.pattern)
```

`normalize_term` (usada pelos métodos escalares) faz `strip().lower()` seguido de remoção de
acento; `contains_expr`/`count_expr` replicam com `str.to_lowercase()` +
`strip_accents_expr` — sem `.strip()` de espaços nas pontas, que não afeta `contains`/`count`
(regex de borda de palavra ignora espaço nas pontas do texto).

O import de `strip_accents_expr` é feito dentro dos métodos (não no topo do módulo) para evitar
import circular: `preprocessing.text` não importa `utils.lexicons`, mas `features/linguistic.py`
e outros módulos de `preprocessing` podem vir a importar `utils.lexicons` no futuro — import
local mantém a dependência unidirecional explícita sem arriscar ciclo.

- [ ] **Step 4: Rodar o teste**

Run: `uv run pytest tests/test_features.py -k lexicon_contains_expr -v`
Expected: PASS.

- [ ] **Step 5: Atualizar `_annotate_lexicon_hits` para usar as novas expressões**

Em `src/features/linguistic.py`, substituir:

```python
def _annotate_lexicon_hits(
    frame: pl.DataFrame, lexicons: dict, available: list[str]
) -> pl.DataFrame:
    """Anota, por tweet, se cada léxico ocorre no texto e quantas vezes."""
    for name in available:
        lexicon = lexicons[name]
        frame = frame.with_columns(
            pl.col(TEXT_CLEAN)
            .map_elements(lexicon.contains, return_dtype=pl.Boolean)
            .alias(f"_has_{name}"),
            pl.col(TEXT_CLEAN)
            .map_elements(lexicon.count, return_dtype=pl.Int64)
            .alias(f"_hits_{name}"),
        )
    return frame
```

por:

```python
def _annotate_lexicon_hits(
    frame: pl.DataFrame, lexicons: dict, available: list[str]
) -> pl.DataFrame:
    """Anota, por tweet, se cada léxico ocorre no texto e quantas vezes."""
    return frame.with_columns(
        [
            expr
            for name in available
            for expr in (
                lexicons[name].contains_expr(pl.col(TEXT_CLEAN)).alias(f"_has_{name}"),
                lexicons[name].count_expr(pl.col(TEXT_CLEAN)).alias(f"_hits_{name}"),
            )
        ]
    )
```

- [ ] **Step 6: Rodar a suíte de features relevante**

Run: `uv run pytest tests/test_features.py -v`
Expected: todos passam, incluindo `test_razoes_lexicais_por_usuario`,
`test_razoes_ficam_entre_zero_e_um`, `test_lexico_inexistente_e_ignorado` (que exercitam
`_annotate_lexicon_hits` indiretamente via `compute_lexicon_ratios`/`build_linguistic_features`).

- [ ] **Step 7: Commit**

```bash
git add src/utils/lexicons.py src/features/linguistic.py tests/test_features.py
git commit -m "feat(features): vetorizar matching de lexico em linguistic.py"
```

### Task 1.5: Reusar `contains_expr`/`count_expr` em `labeling/weak_supervision.py`

Efeito colateral bem-vindo da Task 1.4 (mesma classe `Lexicon`, mesmo padrão de chamada) — não
adiciona vetorização nova, só remove duplicação de UDF equivalente na etapa 3.

**Files:**
- Modify: `src/labeling/weak_supervision.py`

**Interfaces:**
- Consumes: `Lexicon.contains_expr`/`count_expr` (Task 1.4).

- [ ] **Step 1: Localizar e substituir os três usos de `map_elements`**

Em `src/labeling/weak_supervision.py`, nas linhas identificadas na exploração (~98, ~101, ~170),
substituir cada `pl.col(...).map_elements(lexicon.contains, return_dtype=pl.Boolean)` por
`lexicon.contains_expr(pl.col(...))` e `.map_elements(lexicon.count, return_dtype=pl.Int64)` por
`lexicon.count_expr(pl.col(...))`, preservando a coluna de entrada e o `.alias(...)` de cada
chamada original (ler o arquivo atual antes de editar — os três usos não têm exatamente o mesmo
formato de `with_columns`, ao contrário de `linguistic.py`).

- [ ] **Step 2: Rodar os testes da etapa 3**

Run: `uv run pytest tests/test_labeling.py -v` (ajustar o nome exato do arquivo de teste da
etapa de rotulação/supervisão fraca, conforme encontrado em `tests/`)
Expected: todos passam sem alteração de asserção.

- [ ] **Step 3: Commit**

```bash
git add src/labeling/weak_supervision.py
git commit -m "refactor(labeling): reusar contains_expr/count_expr de Lexicon"
```

---

## Fase 2 — Processamento em lote + multiprocessing (etapas 2 e 6)

### Task 2.1: Leitor/escritor de lote com manifesto de retomada

A retomada por lote não pode assumir que todo `user_id` do lote aparece nas linhas de saída: um
usuário pode ser inteiramente filtrado (conta automatizada, abaixo de `min_tweets_per_user`) e
ainda assim precisa contar como "já processado" — hoje isso é garantido por `write_user_partition`
gravar um arquivo mesmo vazio, indexado pelo nome do arquivo (o próprio `user_id`). Em lote, o
nome do arquivo não é mais o `user_id`, então a marca de "processado" precisa vir de outro lugar:
um manifesto pequeno (`_batches/batch_NNNNN.parquet`, só a coluna `user_id`) ao lado do arquivo de
dados, listando todo `user_id` que tinha ao menos uma linha de **entrada** no lote (o mesmo
critério que hoje faz `write_user_partition` ser chamado). `list_files`/`read_partitioned` fazem
`glob("*.parquet")` não-recursivo no diretório, então o subdiretório `_batches/` nunca é lido por
engano pelas etapas seguintes.

**Files:**
- Modify: `src/data/writer.py`
- Modify: `src/data/reader.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `write_batch_partition(frame: pl.DataFrame, directory: Path, batch_index: int) -> Path` (writer.py)
  - `write_batch_manifest(user_ids: list[str] | pl.Series, directory: Path, batch_index: int) -> Path` (writer.py)
  - `list_collected_users_batched(directory: Path) -> set[str]` (reader.py)

- [ ] **Step 1: Escrever os testes**

Adicionar a `tests/test_data.py`, na classe `TestEscritaELeitura`:

```python
    def test_escrita_em_lote_e_manifesto(self, tmp_path: Path) -> None:
        """write_batch_partition e write_batch_manifest gravam arquivos distintos e nomeados."""
        dados = pl.DataFrame({"user_id": ["u_a", "u_a", "u_b"], "valor": [1, 2, 3]})
        destino = write_batch_partition(dados, tmp_path, batch_index=3)
        assert destino.name == "batch_00003.parquet"

        manifesto = write_batch_manifest(["u_a", "u_b", "u_c"], tmp_path, batch_index=3)
        assert manifesto.parent.name == "_batches"
        assert manifesto.name == "batch_00003.parquet"
        assert set(pl.read_parquet(manifesto)["user_id"].to_list()) == {"u_a", "u_b", "u_c"}

    def test_lista_usuarios_em_lote_usa_o_manifesto(self, tmp_path: Path) -> None:
        """list_collected_users_batched enxerga usuários filtrados que não sobraram nos dados."""
        dados = pl.DataFrame({"user_id": ["u_a"], "valor": [1]})
        write_batch_partition(dados, tmp_path, batch_index=0)
        # u_b existia na entrada do lote mas foi inteiramente filtrado — sem manifesto,
        # ficaria pendente para sempre.
        write_batch_manifest(["u_a", "u_b"], tmp_path, batch_index=0)

        assert list_collected_users_batched(tmp_path) == {"u_a", "u_b"}

    def test_lista_usuarios_em_lote_diretorio_vazio(self, tmp_path: Path) -> None:
        """Diretório sem lotes gravados retorna conjunto vazio, sem erro."""
        assert list_collected_users_batched(tmp_path) == set()
```

Adicionar `write_batch_manifest, write_batch_partition` ao import de `data.writer` e
`list_collected_users_batched` ao import de `data.reader` já presentes em `tests/test_data.py`.

- [ ] **Step 2: Rodar os testes e confirmar que falham**

Run: `uv run pytest tests/test_data.py -k "lote or manifesto" -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implementar em `data/writer.py`**

Após `write_user_partition`:

```python
def write_batch_partition(frame: pl.DataFrame, directory: Path, batch_index: int) -> Path:
    """Grava o resultado de um lote de usuários num único parquet.

    Contraparte, em lote, de :func:`write_user_partition`: em vez de um
    arquivo por usuário, grava um arquivo por lote de ``batch_size``
    usuários, reduzindo o número de arquivos e o overhead de chamada
    repetida da função de negócio. O nome é sequencial e determinístico, não
    baseado em hash, para facilitar depuração e ordenação no diretório.

    Parameters
    ----------
    frame : pl.DataFrame
        Resultado do lote (pode ter menos usuários distintos do que os
        pedidos, se algum foi inteiramente filtrado — ver
        :func:`write_batch_manifest` para o registro de retomada completo).
    directory : Path
        Diretório de destino.
    batch_index : int
        Índice sequencial do lote nesta execução.

    Returns
    -------
    Path
        Caminho gravado (``<directory>/batch_<index>.parquet``).

    Examples
    --------
    >>> destino = Path("data/interim/x")
    >>> write_batch_partition(pl.DataFrame({"user_id": ["u_a"]}), destino, 0)  # doctest: +SKIP
    """
    return write_parquet(frame, Path(directory) / f"batch_{batch_index:05d}.parquet", log_hash=False)


def write_batch_manifest(
    user_ids: list[str] | pl.Series, directory: Path, batch_index: int
) -> Path:
    """Grava o manifesto de retomada de um lote: todo ``user_id`` com entrada processada.

    Vive em ``<directory>/_batches/``, fora do padrão ``*.parquet`` lido por
    :func:`data.reader.read_partitioned`/:func:`data.reader.list_files` no
    diretório pai — evita que o manifesto (schema de uma coluna só) seja
    concatenado por engano com os dados reais do lote, que têm o schema
    completo da etapa. É o análogo, em lote, dos arquivos ``.owner`` do cache
    de embeddings (:mod:`pipelines.embedding`): um pequeno artefato auxiliar
    ao lado do dado real, não misturado a ele.

    Parameters
    ----------
    user_ids : list of str or pl.Series
        Todos os usuários que tinham ao menos uma linha de entrada no lote —
        inclusive os que não sobraram nas linhas de saída por terem sido
        inteiramente filtrados.
    directory : Path
        Diretório de dados do lote (o manifesto vai para ``directory/_batches``).
    batch_index : int
        Índice sequencial do lote, mesmo usado em :func:`write_batch_partition`.

    Returns
    -------
    Path
        Caminho gravado (``<directory>/_batches/batch_<index>.parquet``).

    Examples
    --------
    >>> destino = Path("data/interim/x")
    >>> write_batch_manifest(["u_a", "u_b"], destino, 0)  # doctest: +SKIP
    """
    manifest = pl.DataFrame({"user_id": list(user_ids)})
    return write_parquet(
        manifest, Path(directory) / "_batches" / f"batch_{batch_index:05d}.parquet", log_hash=False
    )
```

- [ ] **Step 4: Implementar `list_collected_users_batched` em `data/reader.py`**

Após `list_collected_users`:

```python
def list_collected_users_batched(directory: Path) -> set[str]:
    """Lista os usuários já processados num diretório gravado em lote.

    Contraparte, em lote, de :func:`list_collected_users`: em vez de inferir
    o ``user_id`` do nome do arquivo (que só funciona quando há um arquivo
    por usuário), lê a coluna ``user_id`` dos manifestos gravados por
    :func:`data.writer.write_batch_manifest` em ``directory/_batches/``.

    Parameters
    ----------
    directory : Path
        Diretório de dados gravado em lote (mesmo passado a
        :func:`data.writer.write_batch_partition`).

    Returns
    -------
    set of str
        Identificadores pseudonimizados já processados, agregados de todos
        os manifestos de lote.

    Examples
    --------
    >>> list_collected_users_batched(Path("data/interim/tweets_clean")) == set()
    True
    """
    manifests_dir = Path(directory) / "_batches"
    files = list_files(manifests_dir, "*.parquet")
    if not files:
        return set()

    lazy = pl.concat([pl.scan_parquet(file).select("user_id") for file in files])
    return set(lazy.unique().collect()["user_id"].to_list())
```

- [ ] **Step 5: Rodar os testes**

Run: `uv run pytest tests/test_data.py -k "lote or manifesto" -v`
Expected: PASS.

- [ ] **Step 6: Rodar a suíte completa de `test_data.py`**

Run: `uv run pytest tests/test_data.py -v`
Expected: todos passam, sem regressão em `TestEscritaELeitura` existente.

- [ ] **Step 7: Commit**

```bash
git add src/data/writer.py src/data/reader.py tests/test_data.py
git commit -m "feat(data): adicionar escrita/leitura em lote com manifesto de retomada"
```

### Task 2.2: Configuração de `batch_size`/`max_workers`

**Files:**
- Modify: `src/config/settings.py`
- Modify: `configs/preprocessing.yaml`
- Modify: `configs/features.yaml`
- Test: `tests/test_config.py` (ajustar o nome exato conforme encontrado em `tests/` — arquivo
  que testa `load_config`/`Config`)

**Interfaces:**
- Produces: `PreprocessingConfig.batch_processing: BatchProcessingSection` e
  `FeaturesConfig.batch_processing: BatchProcessingSection`, ambas com campos `batch_size: int`
  e `max_workers: int | None`.

- [ ] **Step 1: Escrever o teste de carregamento da nova seção**

Localizar o teste existente que carrega `configs/` via `load_config` (fixture `config` de
`tests/conftest.py`, reaproveitada) e adicionar, no arquivo de teste de configuração:

```python
def test_batch_processing_preprocessing(config: Config) -> None:
    """A seção batch_processing de preprocessing.yaml carrega com os defaults esperados."""
    assert config.preprocessing.batch_processing.batch_size >= 1
    assert config.preprocessing.batch_processing.max_workers is None or (
        config.preprocessing.batch_processing.max_workers >= 1
    )


def test_batch_processing_features(config: Config) -> None:
    """A seção batch_processing de features.yaml carrega com os defaults esperados."""
    assert config.features.batch_processing.batch_size >= 1
```

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run: `uv run pytest tests/ -k batch_processing -v`
Expected: FAIL — `AttributeError` ou `ValidationError` (campo/seção inexistente).

- [ ] **Step 3: Adicionar a seção `BatchProcessingSection` em `config/settings.py`**

Antes de `PreprocessingConfig` (que agrega as seções de `preprocessing.yaml`):

```python
class BatchProcessingSection(_Section):
    """Processamento em lote e paralelo por usuário (etapas 2 e 6)."""

    batch_size: int = Field(ge=1, le=1000, default=100)
    max_workers: int | None = Field(ge=1, default=None)
```

`max_workers=None` significa "usar `os.process_cpu_count()`" (resolvido em tempo de execução na
Task 2.4/2.5, não na configuração — mantém a seção livre de lógica de plataforma). O default de
`batch_size=100` é o ponto médio da faixa 50-200 pedida; ajustar para o valor recomendado em
`reports/profiling/SUMMARY.md` (Task 0.3) se o profiling sugerir outro número dentro da faixa.

Adicionar o campo às duas classes de configuração:

```python
class PreprocessingConfig(_Section):
    """Conteúdo validado de ``configs/preprocessing.yaml``."""

    deduplication: DeduplicationSection
    normalization: NormalizationSection
    cleaning: CleaningSection
    tokenization: TokenizationSection
    filters: PreprocessingFiltersSection
    batch_processing: BatchProcessingSection
```

```python
class FeaturesConfig(_Section):
    """Conteúdo validado de ``configs/features.yaml``."""

    groups: dict[str, bool]
    linguistic: LinguisticSection
    emotional: EmotionalSection
    semantic: SemanticSection
    temporal: TemporalSection
    behavioral: BehavioralSection
    psychological: PsychologicalSection
    aggregation: AggregationSection
    scaling: ScalingSection
    batch_processing: BatchProcessingSection
```

- [ ] **Step 4: Adicionar a seção aos arquivos YAML**

Em `configs/preprocessing.yaml`, adicionar ao final:

```yaml
batch_processing:
  batch_size: 100
  max_workers: null
```

Em `configs/features.yaml`, adicionar ao final:

```yaml
batch_processing:
  batch_size: 100
  max_workers: null
```

- [ ] **Step 5: Rodar o teste**

Run: `uv run pytest tests/ -k batch_processing -v`
Expected: PASS.

- [ ] **Step 6: Rodar a suíte de configuração completa**

Run: `uv run pytest tests/ -k config -v`
Expected: todos passam — nenhuma seção existente foi alterada, só adição.

- [ ] **Step 7: Commit**

```bash
git add src/config/settings.py configs/preprocessing.yaml configs/features.yaml tests/
git commit -m "feat(config): adicionar batch_processing (batch_size, max_workers) as etapas 2 e 6"
```

### Task 2.3: Função auxiliar de particionamento em lotes

**Files:**
- Modify: `src/data/reader.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `chunk_users(pending: list[str], batch_size: int) -> list[list[str]]`.

- [ ] **Step 1: Escrever o teste**

Adicionar a `tests/test_data.py`:

```python
    def test_chunk_users_divide_em_lotes_do_tamanho_pedido(self) -> None:
        """chunk_users fatia a lista pendente em lotes consecutivos, sem reordenar."""
        pendentes = [f"u_{i}" for i in range(7)]
        lotes = chunk_users(pendentes, batch_size=3)
        assert lotes == [["u_0", "u_1", "u_2"], ["u_3", "u_4", "u_5"], ["u_6"]]

    def test_chunk_users_lista_vazia(self) -> None:
        """Lista pendente vazia produz zero lotes."""
        assert chunk_users([], batch_size=10) == []
```

Adicionar `chunk_users` ao import de `data.reader` em `tests/test_data.py`.

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run: `uv run pytest tests/test_data.py -k chunk_users -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implementar**

Após `select_pending_users` em `src/data/reader.py`:

```python
def chunk_users(pending: list[str], batch_size: int) -> list[list[str]]:
    """Fatia a lista de usuários pendentes em lotes consecutivos.

    Parameters
    ----------
    pending : list of str
        Usuários pendentes, na ordem determinística de
        :func:`select_pending_users`.
    batch_size : int
        Tamanho máximo de cada lote (o último pode ser menor).

    Returns
    -------
    list of list of str
        Lotes, na mesma ordem da lista de entrada.

    Examples
    --------
    >>> chunk_users(["u_a", "u_b", "u_c"], batch_size=2)
    [['u_a', 'u_b'], ['u_c']]
    """
    return [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]
```

- [ ] **Step 4: Rodar o teste**

Run: `uv run pytest tests/test_data.py -k chunk_users -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/reader.py tests/test_data.py
git commit -m "feat(data): adicionar chunk_users para particionar pendentes em lotes"
```

### Task 2.4: Paralelizar `PreprocessingStage` (etapa 2) por lote

**Files:**
- Modify: `src/pipelines/preprocessing.py`
- Test: `tests/test_pipelines_preprocessing.py` (criar, se não existir teste de integração da
  stage — verificar `tests/` primeiro; se já houver um arquivo cobrindo `PreprocessingStage`,
  adicionar lá em vez de criar um novo)

**Interfaces:**
- Consumes: `read_user_histories(directory, user_ids=...)` (já existe, `data/reader.py`),
  `run_preprocessing` (já existe, aceita múltiplos usuários), `write_batch_partition`,
  `write_batch_manifest` (Task 2.1), `list_collected_users_batched`, `chunk_users` (Task 2.3),
  `config.preprocessing.batch_processing` (Task 2.2).
- Produces: `_process_preprocessing_batch(batch_index, user_ids, user_histories_dir,
  tweets_clean_dir, config) -> int` — função de módulo top-level em `pipelines/preprocessing.py`
  (importável pelo worker `spawn`), retorna o número de linhas gravadas.

- [ ] **Step 1: Escrever o teste de equivalência lote-vs-usuário-a-usuário**

Este é o teste central da fase: processar N usuários em 1 lote deve produzir o mesmo resultado
(linha a linha, ordenado) que processar os mesmos N usuários um a um com o código atual.

```python
"""Testes de integração da etapa 2 (preprocess) em lote."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from config.settings import Config
from data.reader import read_partitioned, read_user_history
from data.writer import write_partitioned
from pipelines.preprocessing import PreprocessingStage, _process_preprocessing_batch
from pipelines.base import StageContext
from preprocessing.pipeline import run_preprocessing


@pytest.fixture
def user_histories_dir(tmp_path: Path, raw_tweets: pl.DataFrame) -> Path:
    """Grava os tweets brutos sintéticos particionados por usuário, como a etapa collect faria."""
    directory = tmp_path / "user_histories"
    write_partitioned(raw_tweets, directory, "user_id")
    return directory


class TestProcessamentoEmLote:
    """O processamento em lote deve produzir o mesmo resultado que usuário a usuário."""

    def test_lote_equivale_a_usuario_a_usuario(
        self, user_histories_dir: Path, config: Config, raw_tweets: pl.DataFrame
    ) -> None:
        user_ids = sorted(raw_tweets["user_id"].unique().to_list())[:5]

        # Referência: processamento atual, um usuário por vez.
        referencia = pl.concat(
            [
                run_preprocessing(
                    read_user_history(user_histories_dir, user_id), config, allow_empty=True
                )
                for user_id in user_ids
            ],
            how="vertical_relaxed",
        ).sort(["user_id", "created_at"])

        # Em teste: um único lote com os mesmos usuários.
        tweets_clean_dir = user_histories_dir.parent / "tweets_clean"
        n_gravado = _process_preprocessing_batch(
            0, user_ids, user_histories_dir, tweets_clean_dir, config
        )
        em_lote = read_partitioned(tweets_clean_dir).sort(["user_id", "created_at"])

        assert n_gravado == em_lote.height
        assert em_lote.equals(referencia.select(em_lote.columns))
```

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run: `uv run pytest tests/test_pipelines_preprocessing.py -v`
Expected: FAIL — `ImportError: _process_preprocessing_batch`.

- [ ] **Step 3: Implementar `_process_preprocessing_batch` e reescrever `PreprocessingStage.run`**

Em `src/pipelines/preprocessing.py`, substituir o conteúdo do arquivo (mantendo a docstring do
módulo) por:

```python
"""Etapa 2 — limpeza, normalização e filtragem dos tweets coletados."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from config.logging import get_logger
from config.settings import Config
from data.reader import (
    chunk_users,
    count_partitioned_rows,
    list_collected_users,
    list_collected_users_batched,
    read_user_histories,
    select_pending_users,
)
from data.writer import write_batch_manifest, write_batch_partition
from pipelines.base import PipelineStage, StageContext
from preprocessing.pipeline import run_preprocessing
from utils.progress import build_progress

logger = get_logger(__name__)


def _process_preprocessing_batch(
    batch_index: int,
    user_ids: list[str],
    user_histories_dir: Path,
    tweets_clean_dir: Path,
    config: Config,
) -> int:
    """Processa e grava um lote de usuários do pré-processamento.

    Roda dentro de um processo worker (``spawn``) — por isso é uma função de
    módulo top-level, não um método ou closure: o `multiprocessing` precisa
    conseguir importá-la no processo filho. Lê o lote inteiro com uma só
    chamada, processa com uma só chamada de :func:`run_preprocessing` (que já
    agrega por usuário internamente) e grava um único parquet de dados mais
    o manifesto de retomada.

    Parameters
    ----------
    batch_index : int
        Índice sequencial do lote nesta execução.
    user_ids : list of str
        Usuários deste lote.
    user_histories_dir : Path
        Diretório de históricos brutos particionados por usuário.
    tweets_clean_dir : Path
        Diretório de destino dos tweets limpos, gravado em lote.
    config : Config
        Configuração completa do projeto.

    Returns
    -------
    int
        Número de linhas gravadas no lote (pode ser 0 se todos os usuários
        do lote foram inteiramente filtrados).

    Examples
    --------
    >>> _process_preprocessing_batch(
    ...     0, ["u_a"], Path("data/raw/user_histories"), Path("data/interim/tweets_clean"), config
    ... )  # doctest: +SKIP
    """
    frame = read_user_histories(user_histories_dir, user_ids=user_ids)
    processed_ids = frame["user_id"].unique().to_list() if not frame.is_empty() else []

    if frame.is_empty():
        write_batch_manifest(processed_ids, tweets_clean_dir, batch_index)
        return 0

    clean = run_preprocessing(frame, config, allow_empty=True)
    write_batch_partition(clean, tweets_clean_dir, batch_index)
    write_batch_manifest(processed_ids, tweets_clean_dir, batch_index)
    return clean.height


class PreprocessingStage(PipelineStage):
    """Consolida os históricos por usuário e produz os tweets limpos.

    Processa os usuários pendentes em lotes de
    ``config.preprocessing.batch_processing.batch_size``, distribuídos entre
    processos (``ProcessPoolExecutor``, start method ``spawn``). Cada lote é
    gravado imediatamente após ser processado (retomável por lote — ver
    :func:`data.reader.list_collected_users_batched` — não mais por usuário
    individual). ``--limit-users`` limita quantos usuários pendentes esta
    execução processa, antes do fatiamento em lotes.
    """

    name = "preprocess"
    description = "Limpa, normaliza e filtra os tweets coletados"

    def required_inputs(self, context: StageContext) -> list[Path]:
        """Exige o diretório de históricos produzido pela coleta."""
        return [context.paths.data.user_histories]

    def run(self, context: StageContext) -> dict[str, Any]:
        """Executa o pré-processamento e grava ``tweets_clean/`` em lotes.

        Parameters
        ----------
        context : StageContext
            Contexto compartilhado.

        Returns
        -------
        dict
            Contagens antes e depois, e caminho gravado.

        Examples
        --------
        >>> PreprocessingStage().run(contexto)  # doctest: +SKIP
        """
        paths = context.paths
        batch_config = context.config.preprocessing.batch_processing

        available = list_collected_users(paths.data.user_histories)
        n_raw_tweets = count_partitioned_rows(paths.data.user_histories)

        already_processed = list_collected_users_batched(paths.data.tweets_clean)
        pending = select_pending_users(available, already_processed, context.option("limit_users"))
        batches = chunk_users(pending, batch_config.batch_size)
        logger.info(
            "Pré-processamento: %d usuários já processados, %d pendentes em %d lote(s).",
            len(already_processed),
            len(pending),
            len(batches),
        )

        self._run_batches(batches, paths, context.config, batch_config.max_workers)

        processed_users = list_collected_users_batched(paths.data.tweets_clean)
        n_clean_tweets = count_partitioned_rows(paths.data.tweets_clean) if processed_users else 0

        return {
            "tweets_entrada": n_raw_tweets,
            "tweets_saida": n_clean_tweets,
            "usuarios_entrada": len(available),
            "usuarios_saida": len(processed_users),
            "usuarios_processados_nesta_execucao": len(pending),
            "taxa_retencao": round(n_clean_tweets / max(n_raw_tweets, 1), 4),
            "n_lotes": len(batches),
            "written": str(paths.data.tweets_clean),
        }

    def _run_batches(
        self, batches: list[list[str]], paths: Any, config: Config, max_workers: int | None
    ) -> None:
        """Distribui os lotes entre processos e avança o progresso por lote concluído."""
        if not batches:
            return

        workers = max_workers or os.process_cpu_count() or 1
        with build_progress() as progress:
            task = progress.add_task("Pré-processando lotes de usuários", total=len(batches))
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn")
            ) as executor:
                futures = [
                    executor.submit(
                        _process_preprocessing_batch,
                        index,
                        batch,
                        paths.data.user_histories,
                        paths.data.tweets_clean,
                        config,
                    )
                    for index, batch in enumerate(batches)
                ]
                for future in as_completed(futures):
                    future.result()
                    progress.advance(task)
```

- [ ] **Step 4: Rodar o teste de equivalência**

Run: `uv run pytest tests/test_pipelines_preprocessing.py -v`
Expected: PASS.

- [ ] **Step 5: Rodar a suíte completa de preprocessing**

Run: `uv run pytest tests/test_preprocessing.py tests/test_pipelines_preprocessing.py -v`
Expected: todos passam.

- [ ] **Step 6: Teste manual de retomada**

Rodar duas vezes seguidas sobre uma amostra pequena e confirmar que a segunda execução não
reprocessa ninguém:

```bash
uv run python src/main.py --stage preprocess --limit-users 20 --no-tracking
uv run python src/main.py --stage preprocess --limit-users 20 --no-tracking
```

Expected: a segunda execução loga `0 pendentes em 0 lote(s)` (ou equivalente, se já não houver
mais usuários disponíveis além dos 20 processados).

- [ ] **Step 7: Commit**

```bash
git add src/pipelines/preprocessing.py tests/test_pipelines_preprocessing.py
git commit -m "feat(preprocess): processar em lotes paralelos via ProcessPoolExecutor"
```

### Task 2.5: Paralelizar `FeaturesStage` (etapa 6) por lote

**Files:**
- Modify: `src/pipelines/features.py`
- Test: `tests/test_pipelines_features.py` (criar, seguindo o mesmo padrão da Task 2.4; verificar
  antes se já existe um arquivo de teste de integração desta stage)

**Interfaces:**
- Consumes: `build_user_features_raw` (já existe, aceita múltiplos usuários),
  `write_batch_partition`, `write_batch_manifest`, `list_collected_users_batched`,
  `chunk_users`, `config.features.batch_processing`.
- Produces: `_process_features_batch(batch_index, user_ids, tweets_dir, metadata,
  scores_dir, config, user_features_raw_dir) -> int` — função de módulo top-level em
  `pipelines/features.py`.

- [ ] **Step 1: Escrever o teste de equivalência**

```python
"""Testes de integração da etapa 6 (features) em lote."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from config.settings import Config
from data.reader import read_partitioned, read_user_partition
from data.writer import write_partitioned
from features.builder import build_user_features_raw
from pipelines.features import _process_features_batch


@pytest.fixture
def tweets_labeled_dir(tmp_path: Path, clean_tweets: pl.DataFrame) -> Path:
    """Grava os tweets sintéticos particionados por usuário, como a etapa label faria."""
    directory = tmp_path / "tweets_labeled"
    write_partitioned(clean_tweets, directory, "user_id")
    return directory


class TestProcessamentoDeFeaturesEmLote:
    """A construção de features em lote deve produzir o mesmo resultado que usuário a usuário."""

    def test_lote_equivale_a_usuario_a_usuario(
        self, tweets_labeled_dir: Path, config: Config, clean_tweets: pl.DataFrame
    ) -> None:
        user_ids = sorted(clean_tweets["user_id"].unique().to_list())[:5]

        referencia = pl.concat(
            [
                build_user_features_raw(
                    read_user_partition(tweets_labeled_dir, user_id), config.features
                )
                for user_id in user_ids
            ],
            how="vertical_relaxed",
        ).sort("user_id")

        user_features_raw_dir = tweets_labeled_dir.parent / "user_features_raw"
        n_gravado = _process_features_batch(
            0, user_ids, tweets_labeled_dir, None, None, config, user_features_raw_dir
        )
        em_lote = read_partitioned(user_features_raw_dir).sort("user_id")

        assert n_gravado == em_lote.height
        assert em_lote.equals(referencia.select(em_lote.columns))
```

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run: `uv run pytest tests/test_pipelines_features.py -v`
Expected: FAIL — `ImportError: _process_features_batch`.

- [ ] **Step 3: Implementar `_process_features_batch` e reescrever a construção em lote**

Em `src/pipelines/features.py`, adicionar (após os imports existentes, mantendo tudo que já
existe no arquivo — `_load_semantic`, `run`, `_load_optional_inputs`, `_finalize_features`,
`_write_feature_reports` não mudam) e substituir só `_build_pending_user_features` e
`_build_and_write_user_row`:

```python
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from data.reader import chunk_users, list_collected_users_batched, read_user_partition
from data.writer import write_batch_manifest, write_batch_partition
from utils.progress import build_progress


def _process_features_batch(
    batch_index: int,
    user_ids: list[str],
    tweets_dir: Path,
    metadata: pl.DataFrame | None,
    scores_dir: Path | None,
    config: Config,
    user_features_raw_dir: Path,
) -> int:
    """Processa e grava um lote de usuários da construção de atributos.

    Função de módulo top-level (exigência do ``spawn`` — ver docstring
    equivalente em :func:`pipelines.preprocessing._process_preprocessing_batch`).
    Lê os tweets de cada usuário do lote (mantendo leitura individual: tweets
    rotulados podem ser grandes o bastante para não valer a pena um leitor de
    lote dedicado), concatena, e roda :func:`features.builder.build_user_features_raw`
    uma única vez sobre o lote inteiro.

    Parameters
    ----------
    batch_index : int
        Índice sequencial do lote nesta execução.
    user_ids : list of str
        Usuários deste lote.
    tweets_dir : Path
        Diretório de tweets rotulados, particionado por usuário.
    metadata : pl.DataFrame, optional
        Metadados públicos de todos os usuários (filtrados pelo join interno
        de :func:`features.builder.build_user_features_raw`, não aqui).
    scores_dir : Path, optional
        Diretório de vetores psicológicos, particionado por usuário.
    config : Config
        Configuração completa do projeto.
    user_features_raw_dir : Path
        Diretório de destino das features brutas, gravado em lote.

    Returns
    -------
    int
        Número de linhas (usuários) gravadas no lote.

    Examples
    --------
    >>> _process_features_batch(
    ...     0, ["u_a"], Path("data/interim/tweets_labeled"), None, None, config,
    ...     Path("data/interim/user_features_raw"),
    ... )  # doctest: +SKIP
    """
    tweets_frames = [read_user_partition(tweets_dir, user_id) for user_id in user_ids]
    non_empty = [frame for frame in tweets_frames if not frame.is_empty()]
    processed_ids = sorted({str(frame["user_id"][0]) for frame in non_empty})

    if not non_empty:
        write_batch_manifest(processed_ids, user_features_raw_dir, batch_index)
        return 0

    batch_tweets = pl.concat(non_empty, how="vertical_relaxed")

    scores = None
    if scores_dir is not None:
        score_frames = [
            frame
            for user_id in user_ids
            if not (frame := read_user_partition(scores_dir, user_id)).is_empty()
        ]
        scores = pl.concat(score_frames, how="vertical_relaxed") if score_frames else None

    raw_rows = build_user_features_raw(
        batch_tweets, config.features, metadata=metadata, psychological_scores=scores
    )
    write_batch_partition(raw_rows, user_features_raw_dir, batch_index)
    write_batch_manifest(processed_ids, user_features_raw_dir, batch_index)
    return raw_rows.height
```

Substituir, na classe `FeaturesStage`, o método `_build_pending_user_features` (e remover
`_build_and_write_user_row`, que deixa de ser usado):

```python
    def _build_pending_user_features(
        self,
        tweets_dir: Path,
        metadata: pl.DataFrame | None,
        scores_dir: Path | None,
        pending: list[str],
        config: Any,
        paths: Any,
    ) -> None:
        """Constrói e grava, em lotes paralelos, as features brutas dos pendentes."""
        batch_size = config.features.batch_processing.batch_size
        max_workers = config.features.batch_processing.max_workers or os.process_cpu_count() or 1
        batches = chunk_users(pending, batch_size)

        with build_progress() as progress:
            task = progress.add_task("Construindo atributos em lotes", total=len(batches))
            with ProcessPoolExecutor(
                max_workers=max_workers, mp_context=get_context("spawn")
            ) as executor:
                futures = [
                    executor.submit(
                        _process_features_batch,
                        index,
                        batch,
                        tweets_dir,
                        metadata,
                        scores_dir,
                        config,
                        paths.data.user_features_raw,
                    )
                    for index, batch in enumerate(batches)
                ]
                for future in as_completed(futures):
                    future.result()
                    progress.advance(task)
```

Atualizar `run()` para usar `list_collected_users_batched` no lugar de `list_collected_users`
para `paths.data.user_features_raw` (a entrada, `tweets_labeled`, continua particionada por
usuário — `list_collected_users` normal):

```python
        available = list_collected_users(paths.data.tweets_labeled)
        already_processed = list_collected_users_batched(paths.data.user_features_raw)
        pending = select_pending_users(available, already_processed, context.option("limit_users"))
```

Remover, do topo do arquivo, o import de `track` (`utils.progress`) que não é mais usado
diretamente por esta stage (o progresso passa por `build_progress`, importado acima), mantendo
os demais imports de `data.reader`/`data.writer` já existentes e adicionando os novos citados.

- [ ] **Step 4: Rodar o teste de equivalência**

Run: `uv run pytest tests/test_pipelines_features.py -v`
Expected: PASS.

- [ ] **Step 5: Rodar a suíte completa de features**

Run: `uv run pytest tests/test_features.py tests/test_pipelines_features.py -v`
Expected: todos passam.

- [ ] **Step 6: Teste manual de retomada**

```bash
uv run python src/main.py --stage features --limit-users 20 --no-tracking
uv run python src/main.py --stage features --limit-users 20 --no-tracking
```

Expected: a segunda execução não reprocessa ninguém (mesmo critério da Task 2.4, Step 6).

- [ ] **Step 7: Commit**

```bash
git add src/pipelines/features.py tests/test_pipelines_features.py
git commit -m "feat(features): construir atributos em lotes paralelos via ProcessPoolExecutor"
```

### Task 2.6: Atualizar a documentação das etapas 2 e 6

**Files:**
- Modify: `docs/guides/pipeline_stages.md`

**Interfaces:**
- Consumes: nada (documentação).

- [ ] **Step 1: Atualizar as seções "Etapa 2" e "Etapa 6"**

Em `docs/guides/pipeline_stages.md`:

- No parágrafo introdutório ("Um padrão recorrente nas etapas 2–6..."), acrescentar uma frase
  observando que, a partir desta mudança, as etapas 2 e 6 processam e gravam em **lotes** de
  usuários (não mais um por vez), com paralelismo entre lotes via `ProcessPoolExecutor`; as
  etapas 3, 4 e 5 continuam usuário a usuário.
- Na seção "Etapa 2 — Preprocessamento", subseção "Entradas / Saídas": atualizar "um `.parquet`
  por `user_id`" para "um `.parquet` por **lote** de usuários (`batch_NNNNN.parquet`), mais um
  manifesto de retomada em `tweets_clean/_batches/`".
  No passo 2 de "Implementação (passo a passo)", substituir a descrição do laço por usuário pela
  descrição do laço por lote (referenciar `_process_preprocessing_batch`,
  `configs/preprocessing.yaml:batch_processing.batch_size`).
- Na seção "Etapa 6 — Engenharia de Features", mesma atualização para
  `user_features_raw/batch_NNNNN.parquet` + `user_features_raw/_batches/`, referenciando
  `_process_features_batch` e `configs/features.yaml:batch_processing.batch_size`.
- Adicionar, ao final da seção "Design notável" de cada uma das duas etapas, uma linha sobre a
  granularidade de retomada ter mudado de "1 usuário" para "1 lote" (ver
  "Riscos e limitações conhecidas" em
  `docs/superpowers/specs/2026-08-24-pipeline-performance-design.md`).

- [ ] **Step 2: Commit**

```bash
git add docs/guides/pipeline_stages.md
git commit -m "docs(guides): atualizar etapas 2 e 6 para processamento em lote"
```

---

## Self-Review (registrado durante a escrita do plano)

- **Cobertura da spec:** Fase 0 (Tasks 0.1-0.3), Fase 1 (Tasks 1.1-1.5), Fase 2 (Tasks 2.1-2.6)
  cobrem, respectivamente, profiling, vetorização de `map_elements` e lote+multiprocessing —
  as três fases da spec. O achado de correção durante o planejamento (manifesto de retomada por
  lote, Task 2.1) resolve uma lacuna que a spec original não havia coberto (usuários
  inteiramente filtrados dentro de um lote); a spec foi atualizada para registrar essa decisão
  antes de qualquer implementação.
- **Placeholders:** nenhum "TBD"/"implementar depois" — os únicos pontos deixados como
  "verificar no Step X" (suporte de polars a `str.normalize`/`\p{Mn}`, `list.eval` com `is_in`
  grande) vêm acompanhados do código primário **e** do código de fallback concreto, disparado
  por um critério de teste explícito, não uma decisão em aberto.
- **Consistência de tipos:** `write_batch_partition`/`write_batch_manifest` (Task 2.1) são
  usadas com a mesma assinatura em `_process_preprocessing_batch` (Task 2.4) e
  `_process_features_batch` (Task 2.5); `chunk_users` (Task 2.3) é consumida por ambas com o
  mesmo tipo de retorno (`list[list[str]]`); `contains_expr`/`count_expr` (Task 1.4) têm a
  mesma assinatura `(self, column: pl.Expr) -> pl.Expr` reusada na Task 1.5.
