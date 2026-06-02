FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Create necessary directories
RUN mkdir -p data models sql notebooks/charts

# Expose Streamlit port
EXPOSE 9200

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:9200/_stcore/health || exit 1

# Run the app
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=9200", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]