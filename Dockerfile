FROM ghcr.io/praekeltfoundation/python-base-nw:3.11-bullseye AS build

COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /uvx /bin/
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/.venv

COPY pyproject.toml uv.lock README.md ./
COPY src src/
RUN uv sync --locked --no-dev --no-editable --compile-bytecode

FROM ghcr.io/praekeltfoundation/python-base-nw:3.11-bullseye
COPY --from=build .venv/ .venv/
COPY src src/

ENV PATH="/.venv/bin:${PATH}"
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 5000

WORKDIR /src

CMD ["gunicorn", "application:app", "-b", "0.0.0.0:5000", "-w", "4"]
