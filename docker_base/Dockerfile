# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Builds ultralytics/ultralytics:latest image on DockerHub https://hub.docker.com/r/ultralytics/ultralytics
# Image is CUDA-optimized for YOLO single/multi-GPU training and inference

# Start FROM PyTorch image https://hub.docker.com/r/pytorch/pytorch or nvcr.io/nvidia/pytorch:25.02-py3
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Set environment variables
# Avoid DDP error "MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library"
# Suppress TensorFlow cuDNN, cuBLAS, and cuFFT Registration Warnings and PyTorch NNPACK warnings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TORCH_CPP_LOG_LEVEL=ERROR

# Downloads to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# Install linux packages
# gnupg required for Edge TPU install
# libsm6 required by libqxcb to create QT-based windows for visualization; set 'QT_DEBUG_PLUGINS=1' to test in docker
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc git zip unzip wget curl htop libgl1 libglib2.0-0 gnupg libsm6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /ultralytics

# Copy contents and configure git
COPY . .
RUN sed -i '/^\[http "https:\/\/github\.com\/"\]/,+1d' .git/config && \
    sed -i'' -e 's/"opencv-python/"opencv-python-headless/' pyproject.toml
ADD https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt .

# Install pip packages (uv already installed in base image)
RUN uv pip install --system -e "." albumentations faster-coco-eval nvidia-ml-py && \
    # Remove extra build files \
    rm -rf tmp /root/.config/Ultralytics/persistent_cache.json

# ---------------------------------------------------------------------------
# Install ROS2
# ---------------------------------------------------------------------------
ENV ROS_DISTRO=humble

RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ros-${ROS_DISTRO}-desktop-full \
    ros-dev-tools && \
    rm -rf /var/lib/apt/lists/*

RUN pip install "setuptools<81"
RUN rosdep init || true && rosdep update

# ROS2 dependencies
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp && \
    rm -rf /var/lib/apt/lists/*

# ROS2 environment for conda/system Python
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/opt/ros/${ROS_DISTRO}/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc && \
    echo "export AMENT_PREFIX_PATH=/opt/ros/${ROS_DISTRO}:\$AMENT_PREFIX_PATH" >> ~/.bashrc && \
    echo "export ROS_DOMAIN_ID=1" >> ~/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/${ROS_DISTRO}/lib/python3.10/site-packages:\$PYTHONPATH" >> ~/.bashrc && \
    echo "export FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA?max_msg_size=1MB&soets_size=1MB&non_blocking=true&tcp_negotiation_timeout=50" >> ~/.bashrc && \
    echo "[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash" >> ~/.bashrc

ENV SHELL=/bin/bash
RUN ln -sf /bin/bash /bin/sh

# Usage --------------------------------------------------------------------------------------------------------------

# Production builds: https://github.com/ultralytics/ultralytics/blob/main/.github/workflows/docker.yml
# Example (build): t=ultralytics/ultralytics:latest && docker build -f docker/Dockerfile -t $t .
# Example (push): docker push $t
# Example (pull): t=ultralytics/ultralytics:latest && docker pull $t
# Example (run-gpu): docker run -it --ipc=host --runtime=nvidia --gpus all $t
# Example (run-gpu-subset): docker run -it --ipc=host --runtime=nvidia --gpus "device=2,3" $t
# Note: device=2,3 maps to CUDA 0,1 inside the container.
# Example (run-with-volume): docker run -it --ipc=host --runtime=nvidia --gpus all -v "$PWD/shared/datasets:/datasets" $t
# Example (tag-release): t=ultralytics/ultralytics:latest tnew=ultralytics/ultralytics:vX.Y && docker tag $t $tnew && docker push $tnew