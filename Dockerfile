# A slim Debian-based image: small, but still has apt and glibc when a wheel needs them.
FROM python:3.11-slim

# PYTHONDONTWRITEBYTECODE: skip .pyc files, they only bloat the image.
# PYTHONUNBUFFERED: send logs straight to stdout so `docker logs` shows them live.
# HF_HOME: where model weights are cached. A named volume is mounted here by
# docker-compose so the ~4.75 GB of weights survive image rebuilds.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface

WORKDIR /code

# Install the CPU-only builds of PyTorch first. They are the largest and least
# frequently changing dependency, so they earn their own cache layer at the bottom.
# The default PyPI wheels bundle CUDA libraries (~2.5 GB) that this image never uses.
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Dependencies change rarely, application code changes constantly: copy and install the
# requirements before the source so that editing the code never re-runs pip.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Utility scripts change rarely; the application source changes on almost every build.
COPY scripts/ ./scripts/
COPY app/ ./app/

# Run as an unprivileged user; a process that is root inside the container is a needless
# risk if the application is ever compromised. The cache directory is created and owned
# here so that the mounted volume inherits the right permissions.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /opt/huggingface \
    && chown -R appuser:appuser /code /opt/huggingface
USER appuser

# Documents the port for readers and tooling; it does not publish anything by itself.
EXPOSE 8000

# 0.0.0.0 is required: binding to 127.0.0.1 would only accept connections from inside
# the container itself, making the API unreachable from the host or other containers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
