# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/pytorch:24.10-py3

# Avoid some interactive prompts + make pip quieter/reproducible-ish.
# HF_HOME points at the cache populated during the build, so runtime never
# needs to download anything.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    TEXT_ENCODERS_DIR=/opt/text-encoders \
    HF_HUB_DISABLE_TELEMETRY=1

# Where your code will live inside the container
WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      cmake build-essential \
      gosu \
    && rm -rf /var/lib/apt/lists/*

# Some base images ship a broken `/usr/local/bin/cmake` shim (from a partial pip install),
# which shadows `/usr/bin/cmake` and breaks builds that invoke `cmake` (e.g. MotionCorrection).
# Prefer the system cmake.
RUN rm -f /usr/local/bin/cmake || true

# Install from docker_requirements.txt: kimodo editable (-e .),
# but MotionCorrection non-editable (./MotionCorrection). The -e . line ensures [project.scripts]
# from pyproject.toml are installed (kimodo_gen, kimodo_demo, kimodo_textencoder).
# SKIP_MOTION_CORRECTION_IN_SETUP=1 so setup.py does not bundle motion_correction; it is
# installed separately from ./MotionCorrection in the requirements file (non-editable).
COPY docker_requirements.txt /workspace/docker_requirements.txt
COPY setup.py /workspace/setup.py
COPY pyproject.toml /workspace/pyproject.toml
COPY README.md /workspace/README.md
COPY kimodo /workspace/kimodo
COPY kimodo-viser /workspace/kimodo-viser
COPY MotionCorrection /workspace/MotionCorrection

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
 && SKIP_MOTION_CORRECTION_IN_SETUP=1 python -m pip install -r docker_requirements.txt

# Bake every runtime model into the image. Nothing is fetched when a container
# starts. All repos used here are ungated, so no Hugging Face token is needed.
#
# The cache mount keeps downloads across failed/restarted builds; only the
# finished copy is committed to the image layer.
COPY docker/download_models.py /workspace/docker/download_models.py
RUN --mount=type=cache,target=/hf-cache,id=kimodo-hf-cache \
    python /workspace/docker/download_models.py --cache-dir /hf-cache/hub && \
    rm -rf /hf-cache/hub/models--meta-llama--Meta-Llama-3-8B-Instruct \
           /hf-cache/hub/models--nvidia--Kimodo-SMPLX-RP-v1 && \
    mkdir -p /opt/hf-cache && \
    cp -a /hf-cache/hub /opt/hf-cache/hub && \
    chmod -R a+rX /opt/hf-cache /opt/text-encoders

# From here on the container is strictly offline for model loading. If a model
# is missing, fail loudly instead of silently downloading it at runtime.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    LOCAL_CACHE=1

# Sanity check: the cache must exist and be non-empty.
RUN test -d /opt/hf-cache/hub && du -sh /opt/hf-cache

# Pre-build the Viser web client. Without this, viser downloads Node.js + npm
# packages on the first demo start. The caches used for building are removed;
# only the static client build is kept.
RUN python -c "from viser._client_autobuild import ensure_client_is_built; ensure_client_is_built()" \
 && rm -rf /workspace/kimodo-viser/src/viser/client/.nodeenv \
           /workspace/kimodo-viser/src/viser/client/node_modules

# Use the docker-entrypoint script, to allow the docker to run as the actual user instead of root
COPY kimodo/scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint
RUN chmod +x /usr/local/bin/docker-entrypoint

# Default command (change to your entrypoint if you have one)
ENTRYPOINT ["docker-entrypoint"]
CMD ["bash"]
