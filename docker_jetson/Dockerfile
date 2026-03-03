# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Builds ultralytics/ultralytics:latest-jetson-jetpack6 image on DockerHub https://hub.docker.com/r/ultralytics/ultralytics
# Supports JetPack6.1 for YOLO on Jetson AGX Orin, Orin NX and Orin Nano Series

# Start FROM https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Downloads to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# Install dependencies and cleanup
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb .
RUN dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    git python3-pip libopenmpi-dev libopenblas-base libomp-dev libcusparselt0 libcusparselt-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

# Create working directory
WORKDIR /ultralytics

# Copy contents and configure git
COPY . .
RUN sed -i '/^\[http "https:\/\/github\.com\/"\]/,+1d' .git/config && \
    sed -i'' -e 's/"opencv-python/"opencv-python-headless/' pyproject.toml
ADD https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt .

# Pip install onnxruntime-gpu, torch, torchvision and ultralytics, then remove build files
RUN python3 -m pip install --upgrade pip uv && \
    uv pip install --system \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl && \
    uv pip install --system -e ".[export]" && \
    # Remove extra build files
    rm -rf *.whl /root/.config/Ultralytics/persistent_cache.json

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

# ROS2 environment (Python 3.10 on Jetson JetPack 6 / Ubuntu 22.04)
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
# Example (build): t=ultralytics/ultralytics:latest-jetson-jetpack6 && docker build --platform linux/arm64 -f docker/Dockerfile-jetson-jetpack6 -t $t .
# Example (push): docker push $t
# Example (pull): t=ultralytics/ultralytics:latest-jetson-jetpack6 && docker pull $t
# Example (run): docker run -it --ipc=host --runtime=nvidia $t
# Example (run-with-volume): docker run -it --ipc=host --runtime=nvidia -v "$PWD/shared/datasets:/datasets" $t