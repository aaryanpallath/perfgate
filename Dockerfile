# CPU image for running sweeps on Linux hosts. For NVIDIA GPUs, start from an
# nvidia/cuda or NGC PyTorch base image instead and drop the CPU index override.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev
COPY configs ./configs
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["perfgate"]
CMD ["--help"]
