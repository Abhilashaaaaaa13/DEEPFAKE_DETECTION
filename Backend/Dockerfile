FROM python:3.10-slim

# Prevent Python buffering issues
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy code
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
