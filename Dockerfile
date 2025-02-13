FROM python:3.10.10-slim-bullseye
ENV OS_ENV_DOCKER=TRUE

WORKDIR /workdir
COPY . /workdir

RUN pip install --no-cache-dir -r requirements.txt

# CMD ["python3", "app.py"]
# CMD ["sh", "-c", "python3 app1.py & python3 app2.py"]
# CMD ["sh", "-c", "python3 -m app.app1.app1"]
# CMD ["sh", "-c", "python3 app/app1/app1.py & python3 -m http.server 80 --directory app/root"]
CMD ["python3", "app/app.py"]