# Use Node.js base image for building the frontend
FROM node:18-alpine AS frontend-builder

# Install pnpm
RUN npm install -g pnpm

# Set working directory for frontend
WORKDIR /app/consultify-frontend

# Copy frontend package files
COPY consultify-frontend/package.json consultify-frontend/pnpm-lock.yaml ./

# Install frontend dependencies
RUN pnpm install

# Copy frontend source code
COPY consultify-frontend/ ./

# Build the frontend
RUN pnpm run build

# Main container with Python and Node.js
FROM node:18-alpine

# Install Python and build dependencies
RUN apk add --no-cache python3 py3-pip python3-dev build-base nginx

# Install pnpm
RUN npm install -g pnpm

# Create app directory
WORKDIR /app

# Set up Python environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy and install Python dependencies
COPY consultify-graphs/requirements.txt ./consultify-graphs/
RUN pip install --no-cache-dir -r consultify-graphs/requirements.txt

# Copy Python application
COPY consultify-graphs/ ./consultify-graphs/

# Copy frontend build from builder stage
COPY --from=frontend-builder /app/consultify-frontend/.next ./consultify-frontend/.next
COPY --from=frontend-builder /app/consultify-frontend/public ./consultify-frontend/public
COPY --from=frontend-builder /app/consultify-frontend/package.json ./consultify-frontend/
COPY --from=frontend-builder /app/consultify-frontend/next.config.mjs ./consultify-frontend/
COPY --from=frontend-builder /app/consultify-frontend/node_modules ./consultify-frontend/node_modules

# Create nginx configuration
RUN mkdir -p /etc/nginx/conf.d

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy startup script
COPY start.sh ./
RUN chmod +x start.sh

# Expose port 3000 (single port for both services)
EXPOSE 3000

# Start the application
CMD ["./start.sh"]