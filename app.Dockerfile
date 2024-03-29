# Stage 1: `python-base` sets up all our shared environment variables.
FROM python:3.9-slim as python-base

ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # PIP
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # POETRY
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.4.2 \
    # Install poetry to this location
    POETRY_HOME="/opt/poetry" \
    # make poetry create the virtual environment in the project's root
    # it gets named `.venv`
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    # poetry virtual env location
    VENV_PATH="/.venv"

# Add POETRY_HOME to path, and VENV_PATH (poetry virtual env location) to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"


# Stage 2: `builder-base` stage is used to build deps + create our virtual environment. Cuda already installed on pytorch wheel
FROM python-base as builder-base

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        # deps for installing poetry
        curl \
        # deps for building python deps
        build-essential

# Install poetry - respects $POETRY_VERSION & $POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy requirements
COPY poetry.lock pyproject.toml ./

RUN poetry install

# Stage 3: `production` image used for runtime
FROM python-base as production

RUN apt-get update && apt-get install -y ffmpeg # Lib for streamlit image comparison

COPY --from=builder-base $VENV_PATH $VENV_PATH

# Install project
COPY src/super_resolution src/super_resolution

# Copy demo
COPY api/app /app

# Copy app config
COPY .streamlit /.streamlit

# Add project root to path (not to $PATH !)
ENV PYTHONPATH="${PYTHONPATH}:/src"

CMD poetry run super_resolution
