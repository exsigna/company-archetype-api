# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PDF processing and OCR
RUN apt-get update && apt-get install -y \
    # PDF processing utilities
    poppler-utils \
    # OCR and Tesseract
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    # Image processing libraries
    libpng-dev \
    libjpeg-dev \
    libfreetype6-dev \
    zlib1g-dev \
    # Additional fonts for better OCR
    fonts-liberation \
    # Build tools (may be needed for some Python packages)
    gcc \
    g++ \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Tesseract environment variables
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5.00/tessdata/
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temp directory for file processing
RUN mkdir -p /app/temp

# Set permissions
RUN chmod -R 755 /app

# Expose port
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "2", "--timeout", "300", "app:app"]