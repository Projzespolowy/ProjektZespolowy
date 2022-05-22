FROM intel/oneapi-basekit
RUN apt-get update && \
    apt-get -y install gcc mono-mcs
COPY . /opt/project
WORKDIR /opt/project
RUN make clean
ENTRYPOINT [ "make" ]