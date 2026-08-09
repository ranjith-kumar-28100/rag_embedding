# Single image shared by api / worker / ui / prefetch (only the command differs).
# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# curl: compose healthchecks. No libgl/libglib here — unlike stage 1 this image
# has no Docling/OpenCV, because parsing is not this service's job.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Project metadata + source. Editable install, so code edits don't need a
# reinstall.
COPY pyproject.toml README.md ./
COPY rag_embedding ./rag_embedding
COPY service_embed ./service_embed
COPY ui ./ui
COPY alembic ./alembic
COPY alembic.ini ./
COPY docker ./docker

# CPU-only torch. sentence-transformers pulls torch; the default CUDA wheels are
# multiple GB of unused driver payload on a CPU/WSL2 box. `--torch-backend=cpu`
# makes uv resolve the whole torch stack from the PyTorch CPU index in one
# consistent resolution.
#
# Note we do NOT install torchvision (stage 1 has to, for Docling's OCR): with
# no torchvision in the graph there is no torch/torchvision ABI pair to keep in
# lockstep, which is what made stage 1's build fragile.
RUN uv pip install --system --torch-backend=cpu -e ".[service]"

EXPOSE 8001 8502

# Overridden per service in docker-compose.
CMD ["uvicorn", "service_embed.api:app", "--host", "0.0.0.0", "--port", "8001"]
