FROM python:3.10.10-slim-bullseye
ENV OS_ENV_DOCKER=TRUE

WORKDIR /workdir
COPY . /workdir

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]