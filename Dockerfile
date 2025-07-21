FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variables
ENV GOOGLE_API_KEY=AIzaSyD_QkHiMP6SywCbji47EYeYEY1ysIDC00Y

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]