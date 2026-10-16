FROM python:3.10-slim

# Install system dependencies for PyTorch and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Set environment
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install Python dependencies
COPY --chown=user ./requirements.txt /home/user/app/requirements.txt
USER user
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt gunicorn

# Copy project files
COPY --chown=user . /home/user/app

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Launch Flask forensic server
CMD ["python", "web/app.py"]
