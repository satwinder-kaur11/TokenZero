FROM python:3.11-slim

WORKDIR /app
COPY . /app

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
