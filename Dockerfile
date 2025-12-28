# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for some HF models)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port HF Spaces uses
EXPOSE 7860

# Run the application (using Gunicorn for production)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]