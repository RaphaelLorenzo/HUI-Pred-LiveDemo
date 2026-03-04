# 1. Start with Dusty's ROS2 Humble image for L4T R35.4.1
FROM dustynv/ros:humble-desktop-l4t-r35.4.1

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

RUN pip install --upgrade pip

RUN pip install "numpy<2"

# This wheel now matches the Python version (cp38)
RUN pip install https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

# Torchvision must be built from source on Jetson for Python 3.8 + Torch 2.1
RUN pip install git+https://github.com/pytorch/vision.git@v0.16.1

RUN pip install ultralytics==8.4.0 einops timm

RUN apt-get install -y libopenblas-dev

RUN echo "export ROS_DOMAIN_ID=42" >> ~/.bashrc


# 5. Integrate ROS2 with Conda
# We add a helper to the bashrc to source both ROS and the Conda env automatically
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "conda init && conda activate ai_env" >> ~/.bashrc

WORKDIR /ros2_ws
CMD ["/bin/bash", "-l"]