FROM nvcr.io/nvidia/deepstream:6.0.1-triton

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update

RUN apt install python3-gi \
                python3-dev \
                python3-gst-1.0 \
                python3-gi-cairo \
                gir1.2-gtk-3.0 \
                libgirepository1.0-dev \
                libcairo2-dev \
                pkg-config \
                python-gi-dev \
                git \
                cmake \
                g++ \
                python-dev \
                python3 \
                python3-pip \
                python3.8-dev \
                build-essential \
                libglib2.0-dev \
                libglib2.0-dev-bin \
                python-gi-dev \
                libtool \
                m4 \
                autoconf \
                apt-transport-https \
                ca-certificates \
                automake -y

RUN update-ca-certificates
RUN pip install -U pip && pip install pycairo PyGObject
ENV GI_TYPELIB_PATH /usr/lib/x86_64-linux-gnu/girepository-1.0/

RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git \
    && cd deepstream_python_apps/bindings \
    && git submodule update --init

RUN cd deepstream_python_apps/3rdparty/gst-python/ \
    && ./autogen.sh && make && make install

RUN cd deepstream_python_apps/bindings \
    && mkdir build \
    && cd build \
    && cmake .. -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=8 \
    && make

RUN cd deepstream_python_apps/bindings/build/ && pip install pyds-1.1.1-py3-none-linux_x86_64.whl

WORKDIR /app

CMD ["/bin/bash"]

# RUN cd deepstream_python_apps/apps/deepstream-test1 && python3 deepstream_test_1.py