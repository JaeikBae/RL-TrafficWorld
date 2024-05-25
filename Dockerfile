FROM nvidia/cuda:12.1.0-base-ubuntu20.04

WORKDIR /ws

ENV TZ=Asia/Seoul

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3.8 python3-pip && \
    apt-get install -y --no-install-recommends vim git curl wget tzdata && \
    apt-get install -y --no-install-recommends console-setup && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get install -y console-setup

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .