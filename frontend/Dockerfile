FROM python:3.11-slim

WORKDIR /app

# --- зависимости ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- код ---
COPY app.py .
COPY api_client.py .
COPY feature_generator.py .
COPY mappings.py .

# --- порт Streamlit ---
EXPOSE 8501

# --- Streamlit config ---
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# --- запуск ---
CMD ["streamlit", "run", "app.py"]
