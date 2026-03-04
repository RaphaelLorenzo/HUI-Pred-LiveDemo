# 1. Start with Dusty's ROS2 Humble image for L4T R35.4.1
FROM dustynv/ros:humble-desktop-l4t-r35.4.1

# Required for nvidia-container-runtime to mount GPU devices and driver libs
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Set environment
ENV DEBIAN_FRONTEND=interactive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# 2. Install Miniforge (Conda for aarch64)
RUN apt-get update && apt-get install -y wget curl && \
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O miniforge.sh && \
    /bin/bash miniforge.sh -b -p $CONDA_DIR && \
    rm miniforge.sh
    
# 3. Create Conda environment with Python 3.8 (Matches JetPack 5 wheels)
RUN conda create -n ai_env python=3.8 -y

# 4. Install PyTorch & Ultralytics inside Conda
SHELL ["conda", "run", "-n", "ai_env", "/bin/bash", "-c"]

RUN apt-get install -y libopenblas-dev

RUN pip install --upgrade pip

RUN pip install "numpy==1.24.4"

USER root
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    libpython3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install ultralytics==8.4.0 
RUN pip install einops 
RUN pip install timm

# Install all pip packages (torch will get pulled from PyPI here — we fix it below)
# RUN pip install --no-cache-dir \
#     --extra-index-url https://pypi.jetson-ai-lab.dev/jp5 \
#     torchvision==0.16.1

# Force-reinstall the JetPack CUDA torch wheel, overwriting the generic PyPI one
RUN pip install --force-reinstall \
    https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

# RUN pip install --no-cache-dir --no-deps \
#     --extra-index-url https://pypi.jetson-ai-lab.dev/jp5 \
#     torchvision==0.16.1
RUN git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    # We set this to ensure the build sees the GPU
    export FORCE_CUDA=1 && \
    python setup.py install && \
    cd .. && rm -rf torchvision

# remove past installation of torchvision via pip
RUN pip uninstall torchvision -y

# install lap for ultralytics
RUN pip install lap==0.5.13


RUN echo "export ROS_DOMAIN_ID=42" >> ~/.bashrc

# 5. Integrate ROS2 with Conda
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source activate base && conda activate ai_env" >> ~/.bashrc

# Ensure the container knows where CUDA lives (including Tegra driver path)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH

# Ensure the environment is active for any command run via 'docker run'
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ai_env", "/bin/bash", "-l", "-c"]
CMD ["/bin/bash"]
