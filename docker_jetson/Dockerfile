FROM dustynv/ros:humble-desktop-l4t-r36.4.0

SHELL ["/bin/bash", "--login", "-c"]
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

###### Install Python Dependencies
RUN apt-get update && apt-get upgrade -yy

RUN apt-get install -y \
    python-is-python3 \
    python3-pip

# RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
RUN export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
RUN pip install --upgrade pip; python3 -m pip install numpy==1.26.1; python3 -m pip install --no-cache $TORCH_INSTALL





# RUN pip install scipy joblib ruamel.yaml pyyaml opencv-python imageio einops

# ENV OPENCV_IO_ENABLE_OPENEXR=1

# RUN pip install scikit-image webdataset omegaconf

# RUN pip install openpyxl wandb imgaug Ninja xlsxwriter timm albumentations xatlas rtree ultralytics==8.0.120 pycocotools videoio numba

# RUN pip install h5py

# ###### Install ROS2
# ENV ROS_DISTRO=humble


# ###### Set up ROS2 for conda environment (Python 3.10)
# # ROS2 apt packages are built for Python 3.10, which matches our conda environment
# # Add ROS2 Python packages to PYTHONPATH so they can be imported in conda
# RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
#     echo "export LD_LIBRARY_PATH=/opt/ros/humble/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc && \
#     echo "export AMENT_PREFIX_PATH=/opt/ros/humble:\$AMENT_PREFIX_PATH" >> ~/.bashrc && \
#     echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc && \
#     echo "export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:\$PYTHONPATH" >> ~/.bashrc

# ###### Update Buffer
# RUN sysctl net.ipv4.ipfrag_time=3
# RUN sysctl net.ipv4.ipfrag_high_thresh=134217728

# RUN echo "export FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA?max_msg_size=1MB&soets_size=1MB&non_blocking=true&tcp_negotiation_timeout=50" >> ~/.bashrc

# ###### Source ROS2 and workspace (ros2_ws only if present, e.g. mounted)
# RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
# RUN echo "[ -f /ros2_ws/install/setup.bash ] && source /ros2_ws/install/setup.bash" >> ~/.bashrc

# RUN apt-get update && apt-get install -y ros-humble-rmw-cyclonedds-cpp
# RUN echo "export ROS_DOMAIN_ID=1" >> ~/.bashrc


# # Fix versions !
# RUN pip install ultralytics==8.4


# # Setup shell
# ENV SHELL=/bin/bash
# RUN ln -sf /bin/bash /bin/sh
