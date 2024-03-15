# Stage 1: Build environment
FROM python:3.9-slim as builder
WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc libpq-dev && \
    pip install poetry 

# Copy project files
COPY pyproject.toml poetry.lock* ./

# Install dependencies without creating a virtual environment
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-root 

# Add PyTorch source and install, specifying the explicit source for CUDA support
# Uncommented and included in the build stage
# RUN poetry source add -p explicit pytorch https://download.pytorch.org/whl/cu113 && \
#     poetry add --source pytorch torch==1.12.1+cu113 torchvision==0.13.1+cu113

# Copy the rest of your application's code into the build directory
COPY . .

# Stage 2: Runtime environment
FROM python:3.9-slim
WORKDIR /app

# Install Google Chrome
# RUN apt-get update && apt-get install -y wget gnupg2 \
#     && wget https://mirror.cs.uchicago.edu/google-chrome/pool/main/g/google-chrome-stable/google-chrome-stable_114.0.5735.90-1_amd64.deb \
#     && dpkg -i google-chrome-stable_114.0.5735.90-1_amd64.deb; apt-get -fy install

# Install ChromeDriver
# RUN wget https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip \
#     && apt-get install unzip -y \
#     && unzip chromedriver_linux64.zip \
#     && chmod +x chromedriver \
#     && mv chromedriver /usr/local/bin/

# Copy installed packages and binaries from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /build .

# Expose port for the application
EXPOSE 8000

# Command to run the application
CMD ["python", "pipeline.py"]