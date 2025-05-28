# Use this for both Intel/AMD and Apple Silicon
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose both Streamlit and health check ports
EXPOSE 8501 8080

# Set the entrypoint to run your Streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]