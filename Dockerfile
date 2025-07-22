FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files to /app
COPY . /app

# Set environment variables
ENV PORT=8080
EXPOSE 8080

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
# Run with Gunicorn, using correct module name
# CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--log-level", "debug", "webhook_flask_srvr:app"]
ENTRYPOINT ["/app/entrypoint.sh"]