# Use a slim Python image, showed error with 3.13-slim
FROM python:3.13

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables for Flask and Cloud Run
ENV PORT=4500
EXPOSE 4500

# Run Gunicorn with the Flask app, optimized for Cloud Run
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 webhook_flask_srvr:app