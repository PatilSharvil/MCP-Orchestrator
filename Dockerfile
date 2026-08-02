FROM python:3.11-slim

# Install system dependencies & Node.js for stdio MCP servers (like @modelcontextprotocol/server-filesystem)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    build-essential \
    libpq-dev \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pyproject.toml and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy application source code
COPY . /app

EXPOSE 8000

CMD ["python", "-m", "orchestrator.server"]
