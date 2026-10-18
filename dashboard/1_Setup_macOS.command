#!/usr/bin/env bash
# Double-click this file in Finder to run first-time setup.
# If Python is missing, _internal/setup.sh offers a download choice.
cd "$(dirname "$0")" || exit 1

bash ./_internal/setup.sh
setup_status=$?

echo
if [ "$setup_status" -ne 0 ]; then
  echo "Setup is not complete. Follow the instructions above, then"
  echo "double-click 1_Setup_macOS.command again."
  echo
fi
if [ -t 0 ]; then
  read -n 1 -s -r -p "Press any key to close this window..."
  echo
fi
exit "$setup_status"
