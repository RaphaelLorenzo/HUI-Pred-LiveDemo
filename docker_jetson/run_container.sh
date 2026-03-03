#!/bin/bash

DIR=$(pwd)/
IsRunning=$(docker ps -f name=huipreddemo_jetson | grep -c "huipreddemo_jetson")

if [ "$IsRunning" -eq "0" ]; then
    xhost +local:docker
    docker run --rm \
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
else
    echo "huipreddemo_jetson container is already running. Opening new terminal..."
    docker exec -ti huipreddemo_jetson bash -c "source /opt/ros/humble/setup.bash && cd $DIR && bash"
fi
