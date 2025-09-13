#!/bin/sh

# Start nginx in the background
nginx -g "daemon off;" &

# Start Python Flask backend in the background
cd /app/consultify-graphs
python3 api.py &

# Start Next.js frontend
cd /app/consultify-frontend
pnpm start &

# Wait for all background processes
wait