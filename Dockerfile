FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including git-lfs
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download LFS files if any pth files are LFS pointers
RUN if [ -f ".gitattributes" ] && grep -q "\.pth.*lfs" .gitattributes; then \
        echo "Detected LFS configuration, pulling LFS files..."; \
        git lfs install && git lfs pull || echo "LFS pull failed, proceeding..."; \
    fi

# Create necessary directories
RUN mkdir -p temp results static

# Debug: List files to verify everything was copied
RUN ls -la
RUN echo "Model files:" && ls -la *.pth || echo "No .pth files found"

# Expose port (will be overridden by Cloud Run)
EXPOSE 8000

# Health check (use fixed port since Cloud Run assigns 8000)
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port $PORT 