FROM python:3.10-slim-bullseye

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

# Download LFS files directly from GitHub using public API (5 model files)
RUN if [ -f ".gitattributes" ] && grep -q "\.pth.*lfs" .gitattributes; then \
        echo "Detected LFS configuration, downloading 5 model files..."; \
        curl -L "https://github.com/juliodiaz0209/TORTUOSITY/raw/main/final_model%20(11).pth" -o "final_model (11).pth" || echo "Failed to download final_model (11).pth"; \
        curl -L "https://github.com/juliodiaz0209/TORTUOSITY/raw/main/final_model_tarsus_improved.pth" -o "final_model_tarsus_improved.pth" || echo "Failed to download final_model_tarsus_improved.pth"; \
        curl -L "https://github.com/juliodiaz0209/TORTUOSITY/raw/main/final_model_tarsus.pth" -o "final_model_tarsus.pth" || echo "Failed to download final_model_tarsus.pth"; \
        curl -L "https://github.com/juliodiaz0209/TORTUOSITY/raw/main/final_model_improved_fixed.pth" -o "final_model_improved_fixed.pth" || echo "Failed to download final_model_improved_fixed.pth"; \
        curl -L "https://github.com/juliodiaz0209/TORTUOSITY/raw/main/final_model_tarsus_improved%20(6).pth" -o "final_model_tarsus_improved (6).pth" || echo "Failed to download final_model_tarsus_improved (6).pth"; \
        ls -la *.pth; \
    fi

# Create necessary directories
RUN mkdir -p temp results static

# Debug: List files to verify everything was copied and downloaded
RUN echo "All files in app directory:" && ls -la
RUN echo "Model files downloaded:" && ls -la *.pth || echo "No .pth files found"

# Expose port (will be overridden by Cloud Run)
EXPOSE 8000

# Health check (use fixed port since Cloud Run assigns 8000)
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port $PORT 