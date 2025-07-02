# Use Python 3.11 slim as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies including OCR
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    libpoppler-cpp-dev \
    pkg-config \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Find and set correct TESSDATA_PREFIX
RUN find /usr -name "tessdata" -type d 2>/dev/null | head -1 > /tmp/tessdata_path
RUN export TESSDATA_PREFIX=$(cat /tmp/tessdata_path) && echo "TESSDATA_PREFIX=$TESSDATA_PREFIX" >> /etc/environment

# Set TESSDATA_PREFIX environment variable correctly
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5.00/tessdata

# Verify tesseract installation and find actual tessdata location
RUN tesseract --version
RUN find /usr -name "*.traineddata" -type f 2>/dev/null | head -5
RUN ls -la /usr/share/tesseract-ocr/*/tessdata/ 2>/dev/null || echo "Checking tessdata location..."

# Set work directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 10000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]