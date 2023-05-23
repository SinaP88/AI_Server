FROM python:3.6-slim-stretch
COPY . .
WORKDIR .
RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-numpy \
    zip \
    libboost-dev \
    libboost-all-dev \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*
RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN export CPLUS_INCLUDE_PATH=/usr/local/include/python3.6m && \
    pip3.6 install --upgrade pip && \
    pip3.6 install --no-cache-dir -r /root/requirements.txt
RUN sh ./build.sh
RUN pip cache info
RUN pip cache purge
EXPOSE 3000
CMD ["flask", "run", "--host=0.0.0.0", "--port=3000"]