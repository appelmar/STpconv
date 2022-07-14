FROM jupyter/tensorflow-notebook:latest

USER root
RUN apt update && apt install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:ubuntugis/ppa && apt update && apt install -y --no-install-recommends libgdal-dev gdal-bin g++ graphviz
USER ${NB_UID}

COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/
RUN echo "GDAL==$(gdal-config --version)" >> /tmp/requirements.txt

RUN pip install setuptools==57.5.0 && \
    pip install --quiet --no-cache-dir --requirement /tmp/requirements.txt && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
