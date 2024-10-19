#!/bin/bash
docker build --tag=meridiancfi/kadlu:latest --compress --file docker/Dockerfile .
docker tag meridiancfi/kadlu:latest meridiancfi/kadlu:latest
docker push meridiancfi/kadlu:latest

# run the package tests locally with:
# docker build --file ./docker/Dockerfile . --tag=kadlu-test
# docker run -it --volume /RAID0/kadlu_data/:/src/kadlu_data kadlu-test
