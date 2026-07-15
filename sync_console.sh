#!/usr/bin/env bash
# Build the production console assets and restart the combined UI/API service.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
service_name="open-asr-console.service"
health_url="http://127.0.0.1:8080/api/health"

command -v node >/dev/null || {
  echo "Node.js is required but was not found in PATH." >&2
  exit 1
}
command -v npm >/dev/null || {
  echo "npm is required but was not found in PATH." >&2
  exit 1
}

echo "Installing locked frontend dependencies..."
npm --prefix "$repo_root/web" ci

echo "Building frontend..."
npm --prefix "$repo_root/web" run build

echo "Restarting $service_name..."
sudo systemctl restart "$service_name"

echo "Waiting for the console health check..."
for attempt in {1..10}; do
  if curl --fail --silent --show-error "$health_url"; then
    echo
    echo "Console synced and running at http://127.0.0.1:8080"
    exit 0
  fi
  sleep 1
done

echo "The service restarted but the health check did not succeed." >&2
echo "Inspect logs with: sudo journalctl -u $service_name -n 100 --no-pager" >&2
exit 1
