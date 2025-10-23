FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./api_service /app/api_service

CMD ["uvicorn", "api_service.main:app", "--host", "0.0.0.0", "--port", "8000"]