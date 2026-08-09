#!/usr/bin/env bash
# Double-click this file in Finder to start the dashboard and open it in
# your browser. No terminal typing required. Keep the window this opens
# alive while you use the dashboard; closing it stops the dashboard.
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Setup hasn't been run yet."
  echo "Double-click setup.command first, then try this again."
  echo
  read -n 1 -s -r -p "Press any key to close this window..."
  echo
  exit 1
fi

echo "Starting MatchPlant Dashboard..."
./.venv/bin/python app.py &
SERVER_PID=$!

for _ in $(seq 1 30); do
  if curl -sf http://127.0.0.1:5050/ >/dev/null 2>&1; then
    open http://127.0.0.1:5050
    break
  fi
  sleep 1
done

echo
echo "The dashboard is running. Keep this window open while you use it."
echo "Close this window (or press Ctrl+C) to stop the dashboard."
wait "$SERVER_PID"
