# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PDF processing and OCR
RUN apt-get update && apt-get install -y \
    # PDF processing utilities
    poppler-utils \
    # OCR and Tesseract with all language data
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-osd \
    libtesseract-dev \
    # Image processing libraries
    libpng-dev \
    libjpeg-dev \
    libfreetype6-dev \
    zlib1g-dev \
    # Additional image tools
    imagemagick \
    # Additional fonts for better OCR
    fonts-liberation \
    fonts-dejavu-core \
    # Build tools (may be needed for some Python packages)
    gcc \
    g++ \
    # File utilities
    file \
    # Network utilities
    curl \
    wget \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Find and set correct Tesseract data path
RUN find /usr -name "*.traineddata" -type f 2>/dev/null | head -5 && \
    find /usr -name "tessdata" -type d 2>/dev/null | head -5

# Set Tesseract environment variables dynamically
RUN TESSDATA_PATH=$(find /usr -name "tessdata" -type d 2>/dev/null | head -1) && \
    if [ -n "$TESSDATA_PATH" ]; then \
        echo "Found tessdata at: $TESSDATA_PATH" && \
        echo "export TESSDATA_PREFIX=$TESSDATA_PATH/" >> /etc/environment; \
    else \
        echo "Creating tessdata directory" && \
        mkdir -p /usr/share/tessdata && \
        echo "export TESSDATA_PREFIX=/usr/share/tessdata/" >> /etc/environment; \
    fi

# Set environment variables with multiple fallback paths
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/
ENV OMP_THREAD_LIMIT=1
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Download language data if missing (check correct path first)
RUN EXISTING_TESSDATA=$(find /usr -name "tessdata" -type d 2>/dev/null | head -1) && \
    if [ -n "$EXISTING_TESSDATA" ] && [ -f "$EXISTING_TESSDATA/eng.traineddata" ]; then \
        echo "Using existing tessdata at: $EXISTING_TESSDATA" && \
        echo "export TESSDATA_PREFIX=$EXISTING_TESSDATA/" >> /etc/environment; \
    elif [ ! -f /usr/share/tessdata/eng.traineddata ]; then \
        echo "Downloading English language data..." && \
        mkdir -p /usr/share/tessdata && \
        wget -q https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata \
             -O /usr/share/tessdata/eng.traineddata && \
        wget -q https://github.com/tesseract-ocr/tessdata/raw/main/osd.traineddata \
             -O /usr/share/tessdata/osd.traineddata && \
        echo "export TESSDATA_PREFIX=/usr/share/tessdata/" >> /etc/environment; \
    fi

# Verify OCR setup and show final tessdata location
RUN echo "=== Tesseract Setup Verification ===" && \
    tesseract --list-langs 2>/dev/null || echo "Language check failed" && \
    echo "=== Available tessdata files ===" && \
    find /usr -name "*.traineddata" -type f 2>/dev/null | head -10 && \
    echo "=== Final TESSDATA_PREFIX ===" && \
    source /etc/environment && echo "TESSDATA_PREFIX: $TESSDATA_PREFIX" && \
    ls -la /usr/share/tessdata/ 2>/dev/null || echo "No /usr/share/tessdata/ directory" && \
    ls -la /usr/share/tesseract-ocr/*/tessdata/ 2>/dev/null || echo "No tesseract-ocr tessdata found"

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