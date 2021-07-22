FROM python:3.9.2-slim
RUN python -m pip install -U pip
RUN pip install numpy pandas sklearn matplotlib --no-cache-dir

WORKDIR /py3_ml
COPY dockers/py3_ml/*.py .
RUN mkdir /data 
# COPY dockers/data/py3_ml/*.csv /data/
