FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY ./* /app/
EXPOSE 3000
ENTRYPOINT [ "uvicorn, "main:app", "--host", "0.0.0.0", "--port", "3000" ]