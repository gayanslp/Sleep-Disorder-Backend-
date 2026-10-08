FROM node:20-slim

# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package.json and requirements.txt
COPY package*.json ./
COPY requirements.txt ./

# Install Node and Python dependencies
RUN npm install
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy all backend code and model files
COPY . .

# Expose port (Railway injects PORT env variable)
ENV PORT=5000
EXPOSE 5000

CMD ["node", "server.js"]
