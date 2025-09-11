# 1) Python Image
FROM python:3.11-slim

# 2) Install OS packages some Python wheels need to compile (and git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3) Create a non-root user for safer dev 
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# 4) Set the working directory inside the container
WORKDIR /app

# 5) Copy only requirements first for better Docker layer caching
COPY requirements.txt /app/requirements.txt

# 6) Install Python deps for your app + dev tools you’ll use in the container
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pytest pytest-cov flake8 black

# 7) Copy the rest of your project files into the image
COPY . /app

# 8) Make Python logs unbuffered & avoid .pyc clutter
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# 9) Default command when the container starts (you can override it)
CMD ["/bin/bash"]
