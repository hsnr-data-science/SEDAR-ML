FROM python:3.12-slim-bookworm

COPY common/. /app/common/
COPY backend/. /app/backend/
WORKDIR /app/backend

RUN apt update && apt upgrade -y
RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

ENTRYPOINT ["python3", "run.py"]
