FROM lofareosc/lofar-pipeline as builder
WORKDIR /source 
USER root
RUN apt-get update && apt install -y \
        casacore-data \
        casacore-dev \
        libboost-python-dev \
        libcfitsio-dev \
        python-dev \
        python3-numpy \
        cmake \
        build-essential \
        libhdf5-serial-dev \
        libarmadillo-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libboost-date-time-dev \
        libboost-numpy-dev \
        libboost-signals-dev \ 
        libboost-program-options-dev \
        libboost-test-dev \
        libxml2-dev \
        libpng-dev \
        pkg-config \
        aoflagger-dev \
        libgtkmm-3.0-dev \
        git \
        wget \
        libfftw3-dev \
        libgsl-dev \ 
        python-pip
        
RUN git clone https://github.com/aroffringa/modeltools.git modeltools && \
    cd modeltools && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j4 && \
    cp bbs2model cluster editmodel render /usr/local/bin/ 

FROM lofareosc/lofar-pipeline
COPY --from=builder /usr/local /usr/local
USER root

RUN apt-get update && apt install -y python-pip 
RUN python -m pip install matplotlib h5py astropy pandas pyyaml
RUN python3 -m pip install matplotlib h5py astropy pandas pyyaml

ADD ddcal.py /usr/local/bin/
ADD cluster.py /usr/local/bin/

RUN useradd -m apertif
WORKDIR /home/apertif
USER apertif


    