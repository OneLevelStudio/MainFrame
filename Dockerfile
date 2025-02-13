FROM python:3.10.10-slim-bullseye
ENV OS_ENV_FASTAPI_WRAPPER=TRUE
ENV OS_ENV_DOCKER_WRAPPER=TRUE

WORKDIR /workdir
COPY . /workdir

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app/app.py"]