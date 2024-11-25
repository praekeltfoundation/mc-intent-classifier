FROM ghcr.io/praekeltfoundation/python-base-nw:3.11-bullseye as build

RUN pip install poetry==1.8.4
COPY . ./
RUN poetry config virtualenvs.in-project true \
    && poetry install --no-dev --no-interaction --no-ansi

FROM ghcr.io/praekeltfoundation/python-base-nw:3.11-bullseye
COPY --from=build .venv/ .venv/
COPY src src/

EXPOSE 5000

WORKDIR /src

CMD ["/.venv/bin/gunicorn", "application:app", "-b", "0.0.0.0:5000", "-w", "4"]
