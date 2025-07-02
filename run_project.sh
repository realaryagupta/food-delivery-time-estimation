#!/bin/bash

# Define variables
VENV_DIR=".venv"
BACKEND_MODULE="backend.main:app"
BACKEND_HOST="0.0.0.0"
BACKEND_PORT=8000
FRONTEND_DIR="frontend"
FRONTEND_PORT=8001
FRONTEND_PAGE="http://localhost:${FRONTEND_PORT}/index.html"

# Step 1: Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR..."
  python3 -m venv $VENV_DIR
else
  echo "Virtual environment already exists."
fi

# Step 2: Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Step 3: Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Run backend server in background
echo "Starting backend server with uvicorn..."
uvicorn $BACKEND_MODULE --host $BACKEND_HOST --port $BACKEND_PORT &

BACKEND_PID=$!

# Step 5: Open new terminal window and run frontend server
echo "Starting frontend HTTP server in a new terminal..."

# macOS Terminal command to open new window and run commands
osascript <<EOF
tell application "Terminal"
    do script "cd $(pwd)/$FRONTEND_DIR && python3 -m http.server $FRONTEND_PORT"
    activate
end tell
EOF

# Step 6: Open default browser to frontend page
echo "Opening frontend page in default browser..."
open $FRONTEND_PAGE

# Optional: Wait for backend to finish (comment out if you want script to exit immediately)
wait $BACKEND_PID
