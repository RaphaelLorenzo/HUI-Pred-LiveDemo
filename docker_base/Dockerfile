FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

SHELL ["/bin/bash", "--login", "-c"]

###### Install Python Dependencies
RUN apt-get update && apt-get upgrade -yy

RUN apt-get install -y \
    python-is-python3 \
    python3-pip

RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

RUN pip install scipy joblib ruamel.yaml pyyaml opencv-python imageio einops

ENV OPENCV_IO_ENABLE_OPENEXR=1

RUN pip install scikit-image webdataset omegaconf

RUN pip install openpyxl wandb imgaug Ninja xlsxwriter timm albumentations xatlas rtree ultralytics==8.0.120 pycocotools videoio numba

RUN pip install h5py

###### Install ROS2
ENV ROS_DISTRO=humble
RUN apt-get update && apt-get upgrade -yy
RUN apt-get install -y \
    curl \
    gnupg2 \
    lsb-release

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt-get update && apt-get upgrade -y
RUN apt install ros-humble-desktop-full -y
RUN apt install ros-dev-tools -y
RUN rosdep init && rosdep update

###### Install ROS Dependencies
RUN apt install -y \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-rviz2

###### Set up ROS2 for conda environment (Python 3.10)
# ROS2 apt packages are built for Python 3.10, which matches our conda environment
# Add ROS2 Python packages to PYTHONPATH so they can be imported in conda
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/opt/ros/humble/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc && \
    echo "export AMENT_PREFIX_PATH=/opt/ros/humble:\$AMENT_PREFIX_PATH" >> ~/.bashrc && \
    echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc && \
    echo "export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:\$PYTHONPATH" >> ~/.bashrc

###### Update Buffer
RUN sysctl net.ipv4.ipfrag_time=3
RUN sysctl net.ipv4.ipfrag_high_thresh=134217728

RUN echo "export FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA?max_msg_size=1MB&soets_size=1MB&non_blocking=true&tcp_negotiation_timeout=50" >> ~/.bashrc

###### Source ROS2 and workspace (ros2_ws only if present, e.g. mounted)
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN echo "[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash" >> ~/.bashrc

RUN apt-get update && apt-get install -y ros-humble-rmw-cyclonedds-cpp
RUN echo "export ROS_DOMAIN_ID=2" >> ~/.bashrc


# Fix versions !
RUN pip install ultralytics==8.4


# Setup shell
ENV SHELL=/bin/bash
RUN ln -sf /bin/bash /bin/sh

# Pre-download YOLO pose weights so they are cached in the image
COPY download_yolo_weights.py /tmp/download_yolo_weights.py
RUN python /tmp/download_yolo_weights.py --output_path /root/.config/Ultralytics/weights/checkpoints/yolo26x-pose.pt
