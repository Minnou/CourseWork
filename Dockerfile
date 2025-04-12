FROM python:3.11

WORKDIR /

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN alembic upgrade head
RUN chmod 777 /database.db

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]