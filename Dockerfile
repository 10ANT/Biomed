# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM continuumio/miniconda3:latest

# Add build argument to force rebuild
ARG CACHEBUST=1

# Avoid tzdata interactive configuration
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependenciess
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    build-essential \
    python3-dev \
    wget \
    openmpi-bin \
    libopenmpi-dev \
    libopenmpi3 \
    libhwloc15 \
    libevent-dev \
    libpmix2 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up OpenMPI environment
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    PATH=/usr/lib/x86_64-linux-gnu/openmpi/bin:$PATH \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Copy environment file
COPY colabs/environment.yml /tmp/environment.yml

# Create conda environment and install gradio and pydantic
RUN conda env create -f /tmp/environment.yml && \
    conda run -n biomedparse pip install gradio==4.44.1 && \
    conda run -n biomedparse pip install pydantic==2.10.6

# Initialize conda in bash
RUN conda init bash

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "biomedparse", "/bin/bash", "-c"]

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set up HF token for the user (HARDCODED - NOT RECOMMENDED)
#RUN echo 'export HF_TOKEN="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"' >> $HOME/.bashrc

# A docker env must work:
ENV HF_TOKEN="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"

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy all files to the app directory
COPY --chown=user . $HOME/app

# Set permissions for entrypoint script
RUN chmod 755 $HOME/app/entrypoint.sh

# Add conda environment to user's path
RUN echo "conda activate biomedparse" >> $HOME/.bashrc

# Use entrypoint script to set up environment and run application
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["exec /home/user/app/entrypoint.sh"]