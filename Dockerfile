# Use slim python and install libgomp for LightGBM
FROM python:3.10-slim

# Prevent interactive tzdata etc.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps: libgomp for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code and models
COPY . .

# Hugging Face exposes $PORT, so bind to it
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
