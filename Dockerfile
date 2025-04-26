# Use the official Python 3.13.3 image as the base
FROM python:3.13.3-slim

# Declare build arguments
ARG PORT

# Set the working directory
WORKDIR /flask_app

# Copy the requirements file
COPY requirements.txt .

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && \
    rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies
RUN python3 -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port
EXPOSE ${PORT}

# Set the entry point
CMD ["python3", "run.py"]
