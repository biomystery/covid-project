FROM python:3.9.2-slim
RUN python -m pip install -U pip
RUN pip install numpy pandas sklearn matplotlib seaborn --no-cache-dir

WORKDIR /py3-ml
COPY dockers/py3-ml/*.py .
#RUN mkdir /data 
# COPY dockers/data/py3_ml/*.csv /data/
