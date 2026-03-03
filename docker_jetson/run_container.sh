#!/bin/bash

docker rm -f huipreddemo_jetson
DIR=$(pwd)/
xhost +local:docker
docker run \
    --name huipreddemo_jetson \
    --gpus all \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    -it \
    --net host \
    --ipc host \
    --pid host \
    --privileged \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --sysctl net.ipv4.ipfrag_time=3 \
    --sysctl net.ipv4.ipfrag_high_thresh=134217728 \
    -v $DIR:$DIR \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp:/tmp \
    -e DISPLAY=${DISPLAY} \
    -e GIT_INDEX_FILE \
    -e ROS_DOMAIN_ID=1 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=/xml_configs/cyclonedds.xml \
    -v $(pwd)/configs/:/xml_configs \
    huipreddemo_jetson:latest \
    bash -c "source /opt/ros/humble/setup.bash && cd $DIR && bash"
