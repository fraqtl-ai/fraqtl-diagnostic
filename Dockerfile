# fraqtl-diagnostic container — enterprise / sandboxed-env users
#
# Usage:
#   docker build -t fraqtl-diagnostic .
#   docker run --rm --gpus all -v $PWD/reports:/work fraqtl-diagnostic \
#     analyze mistralai/Mistral-7B-v0.1 --out-dir /work
#
# Published image (after first release):
#   docker run --rm --gpus all ghcr.io/fraqtl-ai/fraqtl-diagnostic:latest \
#     analyze Qwen/Qwen2.5-0.5B --out-dir /work

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal system deps for torch + wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install fraqtl-diagnostic from PyPI (pulls torch, transformers, etc.)
RUN pip install --upgrade pip && \
    pip install fraqtl-diagnostic

WORKDIR /work
ENTRYPOINT ["fraqtl"]
CMD ["--help"]
