# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv sync --no-dev

# Copy application code
COPY app/ ./app/

# Create .env file with default values for container
RUN echo 'CONNECTION_STRING="postgresql://pgvector:pgvector@db:5432/pgvector"' > .env && \
    echo 'SENTENCE_TRANSFORMER_MODEL="sentence-transformers/all-MiniLM-L6-v2"' >> .env

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]