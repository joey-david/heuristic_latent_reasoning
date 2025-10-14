# syntax=docker/dockerfile:1.6
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ARG USERNAME=coconut
ARG USER_UID=1000
ARG USER_GID=1000

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    WANDB_DIR=/workspace/.cache/wandb

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN if ! getent group "${USER_GID}" >/dev/null; then \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
    fi && \
    useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home "${USERNAME}" && \
    mkdir -p /workspace/.cache/huggingface /workspace/.cache/wandb && \
    chown -R "${USERNAME}:${USERNAME}" /workspace

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY . /opt/coconut
RUN chown -R "${USERNAME}:${USERNAME}" /opt/coconut

USER "${USERNAME}"

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

CMD ["/bin/bash"]
