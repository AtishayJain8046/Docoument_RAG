# Slim runtime image for the Streamlit demo (Hugging Face Spaces compatible).
# Uses the lean requirements-app.txt and embedded Qdrant — single container,
# no external services needed.
FROM python:3.13-slim

WORKDIR /app

# requirements first for layer caching
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY . .

# Embedded Qdrant + reranker off → lean, fast, self-contained (eval shows
# dense + Gemini embeddings already rank answers first).
ENV QDRANT_EMBEDDED=true \
    USE_RERANKER=false \
    PORT=7860

EXPOSE 7860

# Hugging Face Spaces routes to $PORT (7860). --server.address=0.0.0.0 for
# container networking.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
