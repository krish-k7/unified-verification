# Use official Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for opencv-python and others)
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir numpy scipy torch gym matplotlib opencv-python

# Default command: start a bash shell
CMD ["bash"]
