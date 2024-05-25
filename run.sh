docker build -t trafficworld .
xhost +
sudo modprobe uinput
docker run -d  \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v .:/ws \
    -v /dev:/dev \
    --rm -it \
    --gpus all \
    --privileged \
    --device /dev/uinput \
    trafficworld \
    /bin/bash
