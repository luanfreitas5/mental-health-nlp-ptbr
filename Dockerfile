# =============================================================================
# Dockerfile — imagem reprodutível do pipeline.
#
# Build multi-estágio: as dependências são resolvidas num estágio próprio, de
# modo que alterar o código-fonte não invalide a camada de instalação (que é a
# cara). Usuário não-root e base pinada por digest.
#
# Construir:  docker build -t mental-health-nlp-ptbr .
# Executar:   docker run --rm -v "$PWD/data:/app/data" mental-health-nlp-ptbr --status
# =============================================================================

# --- Estágio 1: dependências -------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

# Instala o uv a partir da imagem oficial (evita curl | sh no build).
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Só os arquivos de dependência: esta camada é reaproveitada enquanto o
# uv.lock não mudar.
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# --- Estágio 2: runtime ------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# git é necessário para o SHA registrado nos metadados de reprodutibilidade.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Usuário não-root: um contêiner que roda como root e monta volumes do host
# pode gravar arquivos que o usuário não consegue remover.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42

COPY --chown=app:app src/ ./src/
COPY --chown=app:app configs/ ./configs/
COPY --chown=app:app pyproject.toml Makefile README.md ./

# Diretórios de saída, criados com a propriedade correta antes de trocar de
# usuário (volumes montados depois herdam o dono do host).
RUN mkdir -p data/raw data/external data/interim data/processed \
             models reports logs \
    && chown -R app:app data models reports logs

USER app

# Verificação de saúde: a configuração precisa carregar e validar.
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=2 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); from config.settings import load_config; load_config()" || exit 1

ENTRYPOINT ["python", "src/main.py"]
CMD ["--status"]
