FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables
ENV PORT=8080
ENV FLASK_ENV=production

# Run with Gunicorn, using correct module name
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--log-level", "debug", "webhook_flask_srvr:app"]