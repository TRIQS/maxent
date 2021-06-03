# See ../triqs/packaging for other options
FROM flatironinstitute/triqs:unstable-ubuntu-clang
ARG APPNAME=maxent

RUN apt-get install -y python3-decorator || yum install -y python3-decorator
COPY requirements.txt /src/$APPNAME/requirements.txt
RUN pip3 install -r /src/$APPNAME/requirements.txt

COPY --chown=build . $SRC/$APPNAME
WORKDIR $BUILD/$APPNAME
RUN chown build .
USER build
ARG BUILD_ID
ARG CMAKE_ARGS
ARG USE_TRIQS=1
RUN PYTHONPATH=${USE_TRIQS:+$PYTHONPATH} CMAKE_PREFIX_PATH=${USE_TRIQS:+$CMAKE_PREFIX_PATH} cmake $SRC/$APPNAME -DUSE_TRIQS=${USE_TRIQS} -DTRIQS_ROOT=${USE_TRIQS:+$INSTALL} -DCMAKE_INSTALL_PREFIX=${INSTALL} $CMAKE_ARGS && make -j4 || make -j1 VERBOSE=1
USER root
RUN make install
