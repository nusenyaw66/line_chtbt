# Use a slim Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for Cloud Run
EXPOSE 4500

# Set environment variables
# ENV PORT=8080
ENV FLASK_ENV=production

# Run Gunicorn with the correct module and app instance
CMD ["gunicorn", "--bind", "0.0.0.0:4500", "webhook_flask_srvr:app"]