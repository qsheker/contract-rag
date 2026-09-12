# The FastAPI backend: retrieval, generation and indexing.
#
# Two stages so the runtime image carries the virtualenv but not uv, the build
# metadata or the lockfile resolution step.

FROM python:3.12-slim AS builder

# Installed from PyPI rather than copied out of ghcr.io/astral-sh/uv: the
# image pull needs a second registry, and that one timed out on the first
# build here while PyPI - which the dependencies come from anyway - did not.
RUN pip install --no-cache-dir uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Dependencies first, as their own layer: they change far less often than the
# code, and resolving them is the slow half of the build.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY api ./api
RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

# curl is the healthcheck below; nothing else is added to the runtime image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # ru-en-RoSBERTa is ~1.5 GB and is downloaded on first use. Pointing the
    # cache at a directory the compose file mounts as a volume keeps it across
    # container restarts, instead of re-downloading it every time.
    HF_HOME=/cache/huggingface

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY api ./api

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /cache/huggingface \
    && chown -R app:app /cache
USER app

EXPOSE 8000

# Start-up loads the embedding model, which on a cold cache means downloading
# it; the generous start period keeps the container from being declared
# unhealthy while that happens.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/openapi.json || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
