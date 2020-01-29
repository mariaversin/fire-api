FROM jjanzic/docker-python3-opencv

FROM tensorflow/tensorflow:latest-gpu-py3

ADD . /app

WORKDIR /app

RUN pip install keras

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "app.py", "src/predict.py"]