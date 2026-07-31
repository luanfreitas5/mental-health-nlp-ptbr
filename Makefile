# --- Configuração ----------------------------------------------------------
PYTHON := python
UV := uv
RUN := $(UV) run python src/main.py    # 'src' vira raiz do path ao rodar o script

# PYTHONHASHSEED precisa ser exportado ANTES do interpretador iniciar: definido
# dentro do processo, não afeta a ordem de iteração de conjuntos já criada.
export PYTHONHASHSEED := 42

.DEFAULT_GOAL := help
.PHONY: help init venv install install-all install-llm install-collect install-nlp spacy-model update lock export \
	check format lint typecheck security deadcode complexity docstrings refurb quality \
	test smoke test-all coverage hooks pre-commit update-hooks release docs docs-serve docs-deploy profile clean cache jupyter notebook add remove tree \
	clean-processed clean-reports clean-outputs clean-notebooks \
	status collect collect-dry-run preprocess label psych embed features split train evaluate report \
	pipeline pipeline-full pipeline-exploratory mlflow app ollama-pull

help:  ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

init:  ## Inicializa o projeto (instala dependências + hooks)
	$(MAKE) install
	$(MAKE) hooks

venv:  ## Cria o ambiente virtual (requer: uv)
	$(UV) venv

install:  ## Instala dependências (runtime + dev)
	$(UV) sync --dev

install-all:  ## Instala tudo (todos os extras + dev)
	$(UV) sync --all-extras --dev

install-llm:  ## Instala os extras de LLM (PyTorch + Transformers + Accelerate + Ollama)
	$(UV) sync --extra llm --dev

install-collect:  ## Instala os extras de coleta (twscrape)
	$(UV) sync --extra collect --dev

install-nlp:  ## Instala os extras de PLN (spaCy) e baixa o modelo pt-BR
	$(UV) sync --extra nlp --dev
	$(MAKE) spacy-model

spacy-model:  ## Baixa o modelo do spaCy para português (habilita a lematização)
	$(UV) run python -m spacy download pt_core_news_sm

ollama-pull:  ## Baixa o modelo do Ollama usado na extração psicológica
	ollama pull llama3.2

update:  ## Atualiza todas as dependências e sincroniza
	$(UV) lock --upgrade
	$(UV) sync --all-groups

lock:
	$(UV) lock

export:
	$(UV) export --no-hashes -o requirements.txt

# --- Qualidade -------------------------------------------------------------
check:  ## Checa formatação com ruff
	$(UV) run ruff check .

format:  ## Formata o código com ruff
	$(UV) run ruff format .

lint: ## Lint com ruff
	$(UV) run ruff check --fix .

typecheck:  ## Type checking estático (basedPyright)
	$(UV) run basedpyright

security:  ## Análise de segurança (bandit + pip-audit)
	$(UV) run bandit -r src -c pyproject.toml
	$(UV) run pip-audit

deadcode:  ## Detecta código morto (vulture)
	$(UV) run vulture src

complexity:  ## Limites de complexidade (xenon)
	$(UV) run xenon --max-absolute B --max-modules A --max-average A src

docstrings:  ## Cobertura de docstrings (interrogate)
	$(UV) run interrogate -v src

refurb:  ## Detecta código redundante (refurb)
	$(UV) run refurb src

quality: format lint typecheck security deadcode complexity docstrings refurb   ## Roda toda a suíte de qualidade (espelha o CI)

# --- Testes ----------------------------------------------------------------
test:  ## Roda os testes com cobertura
	$(UV) run pytest -m "not slow"

smoke:  ## Roda apenas os smoke tests
	$(UV) run pytest -m smoke -q

test-all:  ## Roda a suíte completa, inclusive os testes lentos
	$(UV) run pytest

coverage:  ## Gera o relatório de cobertura em HTML
	$(UV) run pytest -m "not slow" --cov-report=html
	@echo "Relatório disponível em htmlcov/index.html"

hooks:  ## Instala os hooks do pre-commit
	$(UV) run pre-commit install
	$(UV) run pre-commit install --hook-type commit-msg
	$(UV) run detect-secrets scan > .secrets.baseline

pre-commit:  ## Roda todos os hooks do pre-commit em todos os arquivos
	$(UV) run pre-commit run --all-files

update-hooks:  ## Atualiza os hooks do pre-commit
	$(UV) run pre-commit autoupdate

release:  ## Cria uma nova release (versão + changelog + tag)
	$(UV) run cz changelog
	$(UV) run cz bump --changelog --yes

# --- Limpeza de saídas do pipeline ------------------------------------------
clean-processed:  ## Remove os artefatos de dados processados
	rm -rf data/processed/*.parquet

clean-reports:  ## Remove os relatórios gerados (pastas por modelo + comparação)
	find reports -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

clean-outputs: clean-processed clean-reports  ## Remove todas as saídas do pipeline

clean-notebooks:  ## Remove os notebooks com células vazias
	$(UV) run nbstripout notebooks

# --- Documentação ----------------------------------------------------------
docs:  ## Constrói a documentação (modo estrito)
	$(UV) run mkdocs build --strict

docs-serve:  ## Servidor local da documentação
	$(UV) run mkdocs serve

docs-deploy:  ## Publica a documentação no GitHub Pages
	$(UV) run mkdocs gh-deploy --force

# --- Utilitários -----------------------------------------------------------
profile:  ## Exemplo de profiling com scalene (ajuste o alvo)
	$(UV) run scalene src/main.py

clean:  ## Remove caches e artefatos temporários
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml site
	find . -type d -name __pycache__ -exec rm -rf {} +

cache:
	$(UV) cache clean

# --- Jupyter ----------------------------------------------------------------
jupyter:
	$(UV) run jupyter lab

notebook:
	$(UV) run jupyter notebook

# --- Gerenciamento de pacotes -----------------------------------------------
add:
	$(UV) add $(PKG)

remove:
	$(UV) remove $(PKG)

tree:
	$(UV) tree

# --- Pipeline ---------------------------------------------------------------
# Cada alvo executa uma etapa isolada; o acoplamento entre elas é o sistema de
# arquivos, então qualquer etapa pode ser reexecutada sem repetir as anteriores.

status:  ## Mostra a situação dos artefatos de dados de cada etapa
	$(RUN) --status

collect:  ## [1] Coleta tweets (exige ETHICS_APPROVAL_ID no .env)
	$(RUN) --stage collect

collect-dry-run:  ## [1] Constrói as consultas de coleta sem fazer requisições
	$(RUN) --stage collect --dry-run

preprocess:  ## [2] Limpa, normaliza e filtra os tweets coletados
	$(RUN) --stage preprocess

label:  ## [3] Rotula sentimento por tweet e classe por usuário
	$(RUN) --stage label

psych:  ## [4] Extrai o vetor psicológico com LLM local (exige Ollama ativo)
	$(RUN) --stage psych

embed:  ## [5] Gera os embeddings semânticos dos tweets
	$(RUN) --stage embed

features:  ## [6] Constrói a matriz de atributos por usuário
	$(RUN) --stage features

split:  ## [7] Particiona os usuários em treino/validação/teste
	$(RUN) --stage split

train:  ## [8] Treina os modelos com validação cruzada
	$(RUN) --stage train

evaluate:  ## [9] Avalia no teste, compara modelos e roda a ablação
	$(RUN) --stage evaluate

report:  ## [10] Gera figuras, model card e datasheet
	$(RUN) --stage report

pipeline:  ## Roda o pipeline completo (etapas 2 a 10; a coleta fica de fora)
	$(RUN) --stage all

pipeline-exploratory:  ## Pipeline completo incluindo a extensão exploratória de modelos
	$(RUN) --stage all --include-exploratory

pipeline-full: collect pipeline  ## Coleta + pipeline completo (exige aprovação ética)

# --- Serviços auxiliares ----------------------------------------------------
mlflow:  ## Sobe a interface do MLflow para inspecionar os experimentos
	$(UV) run mlflow ui --backend-store-uri mlruns

app:  ## Sobe o dashboard Streamlit de resultados
	$(UV) run streamlit run app/dashboard.py
