# To build this docker image, run from the project root directory:
#
#   docker build --file ./docker/python310.Dockerfile . --tag=kadlu-test
#
# A kadlu data directory will be created in /src/kadlu_data.
# To persist data used by kadlu, mount this folder as a directory when running the container
#
#   docker run -it --volume ./kadlu_data/:/src/kadlu_data kadlu-test
#

FROM python:3.10-slim

WORKDIR /src

RUN python -m venv /env
RUN /env/bin/python -m pip install --upgrade pip wheel setuptools pytest

COPY ./setup.py /src/setup.py
COPY ./kadlu /src/kadlu/

RUN /env/bin/python -m pip install .

RUN mkdir /src/kadlu_data
RUN /env/bin/python -c 'import kadlu; kadlu.storage_cfg("/src/kadlu_data")'

CMD ["/env/bin/python", "-m", "pytest", "-s"]
