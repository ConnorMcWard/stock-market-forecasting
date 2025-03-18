# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Default command: run Streamlit visualization (adjust as needed)
CMD ["streamlit", "run", "scripts/visualization.py"]
