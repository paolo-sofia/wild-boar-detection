FROM python:3.12-slim-bookworm AS builder
LABEL authors="paolo"

ENV PATH /root/.local/bin:${PATH}
ENV PYTHONPATH=/app

RUN python -m venv /opt/venv
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

COPY python-api/requirements.txt .
RUN pip install -r requirements.txt && find /opt/venv/ -follow -type f  \
                                                        -name '*.a' -name '*.txt' -name '*.md' -name '*.png' \
                                                        -name '*.jpg' -name '*.jpeg' -name '*.js.map' -name '*.pyc' \
                                                        -name '*.c' -name '*.pxc' -name '*.pyd' \
                                                        -delete \
                                                        && find /opt/venv/ -name '__pycache__' | xargs rm -r

FROM python:3.12-slim-bookworm AS runner
COPY --from=builder /opt/venv /opt/venv

COPY python-api/main.py .env /app/src/
COPY models /app/models

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH"
CMD ["python", "src/main.py"]