FROM python:3.11-slim

LABEL maintainer="OpenEnv Contributors"
LABEL description="OpenEnv Data Cleaning — LLM Agent Evaluation Environment"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Generate datasets at build time (if not already present)
RUN python scripts/generate_datasets.py

# Expose HuggingFace Spaces default port
EXPOSE 7860

# Default command: start the FastAPI server (NOT inference.py)
# The HF Space must respond to POST /reset with HTTP 200
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
