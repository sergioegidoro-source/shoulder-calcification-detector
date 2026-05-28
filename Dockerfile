# Use Python base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Instalación de dependencias mínimas esenciales
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and frontend
COPY Backend/ ./Backend/
COPY Frontend/ ./Frontend/
COPY run.sh .

# Expose HF Spaces port
EXPOSE 7860

# Make script executable
RUN chmod +x run.sh

# Start the application
CMD ["./run.sh"]