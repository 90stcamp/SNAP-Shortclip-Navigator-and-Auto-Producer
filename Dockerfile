# Stage 1: Build environment
FROM python:3.8.10-slim as builder
WORKDIR /build
# Install build dependencies
RUN apt-get -y update && \
    pip install poetry && \
    apt-get install -y sudo && \
    apt-get install -y --no-install-recommends build-essential gcc libpq-dev
# Copy project files
COPY pyproject.toml poetry.lock* ./
# Install dependencies without creating a virtual environment
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-root 
# Add PyTorch source and install, specifying the explicit source for CUDA support

# # Uncommented and included in the build stage
# RUN poetry source add -p explicit pytorch https://download.pytorch.org/whl/cu110 && \
#     poetry add --source pytorch torch==1.7.1+cu110 torchvision==0.8.2+cu110

# Copy the rest of your application's code into the build directory
COPY . .

# Stage 2: Runtime environment
FROM python:3.8.10-slim
WORKDIR /app
# Install sudo&ffmpeg
RUN apt-get -y update && \
    adduser --disabled-password --gecos "" user && \
    echo 'user:user' | chpasswd && \
    adduser user sudo && \
    echo 'user ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN apt-get install -y ffmpeg
# Copy installed packages and binaries from builder stage
COPY --from=builder /usr/local/lib/python3.8 /usr/local/lib/python3.8
COPY --from=builder /usr/local/bin /usr/local/bin
# Copy application code
COPY --from=builder /build .
# Expose port for the application
EXPOSE 8888
# Command to run the application
CMD ["python", "visual_score.py"]
# text도 args로 추가 필요(json으로)
