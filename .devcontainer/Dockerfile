FROM python:3.14-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        build-essential \
        make \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Fixing the "Running pip as the 'root' user" warning.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip


WORKDIR /workspaces/ml_ws
