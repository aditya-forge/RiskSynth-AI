# RiskSynth AI - Streamlit app container (Render, Google Cloud Run, any Docker host)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8501

WORKDIR /srv/risksynth

# Install dependencies first so code changes do not invalidate this layer
COPY pyproject.toml setup.py README.md ./
COPY src ./src
# Editable install: the package reads configs/ and data/ from this directory
RUN pip install -e ".[app]"

COPY configs ./configs
COPY data/processed ./data/processed
COPY app ./app
COPY .streamlit ./.streamlit

# Writable runtime folders, owned by an unprivileged user
RUN mkdir -p data/synthetic reports logs \
    && useradd --create-home --uid 1000 risksynth \
    && chown -R risksynth:risksynth data/synthetic reports logs
USER risksynth

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8501\")}/_stcore/health', timeout=4)"

# Render and Cloud Run inject $PORT; locally it defaults to 8501
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0"]
