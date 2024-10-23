ARG PYTHON_VERSION=3.12.7
FROM python:${PYTHON_VERSION}-slim as base

EXPOSE 5002

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY ./templates /app/templates
COPY ./carMobileNetSSD.py /app
COPY ./mobilenet_iter_73000.caffemodel /app
COPY ./deploy.prototxt /app
COPY ./requirements.txt /app

# Install pip requirements
#COPY requirements.txt .
#RUN python -m pip install -r requirements.txt


#RUN python -m pip install --no-cache-dir ultralytics
RUN python -m pip install eventlet opencv-python-headless flask flask-socketio paho-mqtt


# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
#CMD ["gunicorn", "--bind", "0.0.0.0:5002", "car_deteciont:app"]

CMD python ./carMobileNetSSD.py