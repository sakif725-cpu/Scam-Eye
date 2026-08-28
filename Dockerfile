# Use lightweight Python 3.11 image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Enable unbuffered logs so requests show immediately in Render dashboard
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies & Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY main.py .

# Expose default port
EXPOSE 8000

# Run the app (main.py dynamically reads Render's $PORT)
CMD ["python", "main.py"]
