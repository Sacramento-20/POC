# syntax=docker/dockerfile:1.4

# =============================================================
# ETAPA 1: Compilação do Python 3.10.13 (100% em Cache Permanente)
# =============================================================
FROM ubuntu:22.04 AS python-builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Mantém o cache do APT entre builds
RUN rm -f /etc/apt/apt.conf.d/docker-clean

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    ca-certificates \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    libffi-dev \
    uuid-dev

WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz && \
    tar -xf Python-3.10.13.tgz && \
    cd Python-3.10.13 && \
    ./configure --prefix=/opt/python-3.10.13 --enable-optimizations --with-ensurepip=install && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && \
    rm -rf Python-3.10.13 Python-3.10.13.tgz

# =============================================================
# ETAPA 2: Imagem Final
# =============================================================
FROM ubuntu:22.04 AS final

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Mantém o cache do APT entre builds
RUN rm -f /etc/apt/apt.conf.d/docker-clean

# Cache inteligente do APT: não baixa novamente os pacotes .deb mesmo se o comando for reexecutado
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc-10 \
    g++-10 \
    wget \
    curl \
    git \
    ca-certificates \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    flex \
    bison \
    libssl-dev \
    zlib1g-dev \
    libsqlite3-dev \
    libbz2-dev \
    libffi-dev \
    # Dependências do NS-3 / MPI / Hypatia / Cartopy
    openmpi-bin \
    libopenmpi-dev \
    pkg-config \
    libxml2 \
    libxml2-dev \
    libboost-all-dev \
    cmake \
    rsync \
    unzip \
    gnuplot \
    graphviz \
    imagemagick \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    lcov \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-10 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-10 100

# Copia o Python 3.10.13 compilado da etapa 1 (Instantâneo / Cached)
COPY --from=python-builder /opt/python-3.10.13 /usr/local

# Configuração de PATH e links simbólicos
ENV PATH="/usr/local/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib64:/usr/local/lib"
ENV CC="gcc"
ENV CXX="g++"

RUN ln -sf /usr/local/bin/python3.10 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3.10 /usr/local/bin/python && \
    /usr/local/bin/python3 -m ensurepip --default-pip && \
    ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip3 && \
    ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip

# Cache de pacotes do PIP (não baixa novamente da internet)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel

# Instalação do Antigravity CLI
RUN curl -fsSL https://antigravity.google/cli/install.sh | bash

# -------------------------------------------------------------
# 3. Clone do repositório Hypatia e Descompactação do NS-3
# -------------------------------------------------------------
WORKDIR /workspace
RUN git clone --recurse-submodules https://github.com/snkas/hypatia.git /workspace/hypatia

# Descompacta o ns-3.31 para dentro do diretório simulator/ (onde fica o waf)
WORKDIR /workspace/hypatia/ns3-sat-sim
RUN unzip ns-3.31.zip && \
    cp -r ns-3.31/* simulator/ && \
    rm -rf ns-3.31 && \
    cd simulator && \
    git submodule update --init --recursive

# Instalação das dependências Python do Hypatia com Cache do PIP
WORKDIR /workspace/hypatia
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
    numpy \
    astropy \
    ephem \
    networkx \
    sgp4 \
    geopy \
    matplotlib \
    statsmodels \
    cartopy \
    exputil && \
    if [ -f /workspace/hypatia/satgenpy/setup.py ]; then \
        python3 -m pip install -e /workspace/hypatia/satgenpy; \
    fi

# -------------------------------------------------------------
# 4. Substituição dos arquivos customizados da raiz
# -------------------------------------------------------------
COPY arbiter-satnet.cc /workspace/hypatia/ns3-sat-sim/simulator/contrib/satellite-network/model/arbiter-satnet.cc
COPY generate_dynamic_state.py /workspace/hypatia/satgenpy/satgen/dynamic_state/generate_dynamic_state.py
COPY satellite.cc /workspace/hypatia/ns3-sat-sim/simulator/src/satellite/model/satellite.cc

# -------------------------------------------------------------
# 5. Criação do ponto de montagem (Volume Mount Point)
# -------------------------------------------------------------
RUN mkdir -p /workspace/hypatia/my_simulation
VOLUME ["/workspace/hypatia/my_simulation"]

# -------------------------------------------------------------
# 6. Configuração e Compilação do Simulador NS-3 (Hypatia)
# -------------------------------------------------------------
WORKDIR /workspace/hypatia/ns3-sat-sim/simulator
RUN chmod +x ./waf && \
    ./waf configure --build-profile=debug --enable-mpi --enable-examples --enable-tests && \
    ./waf -j$(nproc)

# Diretório de trabalho padrão ao iniciar o container (/workspace/hypatia)
WORKDIR /workspace/hypatia

CMD ["/bin/bash"]
