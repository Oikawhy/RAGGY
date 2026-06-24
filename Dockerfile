FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BGE-M3 model into image (Architecture: startup must not pull from HuggingFace)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY app ./app
COPY db ./db
COPY scripts ./scripts
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "app.main:create_production_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
