FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y libpq-dev python3-dev
RUN pip install -r requirements.txt
COPY ./* /app/
EXPOSE 3000
ENTRYPOINT [ "uvicorn, "main:app", "--host", "0.0.0.0", "--port", "3000" ]