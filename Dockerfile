# Use the official Python image as the base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 default-jre

# Build PRISM from binary
RUN cd artifact-1621 && \
    tar xfz prism-4.8.1-linux64-x86.tar.gz && \
    mv prism-4.8.1-linux64-x86 prism_ws && \
    cd prism_ws && \
    ./install.sh

# Install necessary Python packages
RUN pip install --no-cache-dir numpy scipy torch gym matplotlib opencv-python gymnasium gym[classic_control] pygame

# To observe or edit the PRISM model
RUN apt-get update && apt-get install -y nano
