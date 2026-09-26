#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <git-sha>" >&2
    exit 1
fi

SHA="$1"

# If not running on mcmc-beast, delegate via SSH
if [[ "$(uname -n | cut -d. -f1)" != "mcmc-beast" ]] && [ ! -d "/root/BayesianFootball" ]; then
    ssh root@mcmc-beast "export PATH=\"/root/.juliaup/bin:\$PATH\"; bash -s -- '$SHA'" << 'REMOTE_EOF'
#!/usr/bin/env bash
set -euo pipefail
SHA="$1"
TARGET_DIR="/root/BF_runs/$SHA"

git -C /root/BayesianFootball fetch origin >&2

if [ ! -d "$TARGET_DIR" ]; then
    git -C /root/BayesianFootball worktree add "$TARGET_DIR" "$SHA" >&2
fi

if [ -f "/root/BayesianFootball/Manifest.toml" ]; then
    ln -sf /root/BayesianFootball/Manifest.toml "$TARGET_DIR/Manifest.toml"
fi

if [ -f "/root/BayesianFootball/.env" ]; then
    ln -sf /root/BayesianFootball/.env "$TARGET_DIR/.env"
fi

mkdir -p "$TARGET_DIR/.cache"
mkdir -p "/root/BF_runs/logs/$SHA"
if compgen -G "/root/BayesianFootball/.cache/datastore_Scottish*.jls" > /dev/null; then
    cp -n /root/BayesianFootball/.cache/datastore_Scottish*.jls "$TARGET_DIR/.cache/" 2>/dev/null || true
elif compgen -G "/root/BayesianFootball-scotland-cross-tier/.cache/datastore_Scottish*.jls" > /dev/null; then
    cp -n /root/BayesianFootball-scotland-cross-tier/.cache/datastore_Scottish*.jls "$TARGET_DIR/.cache/" 2>/dev/null || true
fi

echo "$TARGET_DIR"
REMOTE_EOF
    exit $?
fi

# Running directly on mcmc-beast:
TARGET_DIR="/root/BF_runs/$SHA"

git -C /root/BayesianFootball fetch origin >&2

if [ ! -d "$TARGET_DIR" ]; then
    git -C /root/BayesianFootball worktree add "$TARGET_DIR" "$SHA" >&2
fi

if [ -f "/root/BayesianFootball/Manifest.toml" ]; then
    ln -sf /root/BayesianFootball/Manifest.toml "$TARGET_DIR/Manifest.toml"
fi

if [ -f "/root/BayesianFootball/.env" ]; then
    ln -sf /root/BayesianFootball/.env "$TARGET_DIR/.env"
fi

mkdir -p "$TARGET_DIR/.cache"
mkdir -p "/root/BF_runs/logs/$SHA"
if compgen -G "/root/BayesianFootball/.cache/datastore_Scottish*.jls" > /dev/null; then
    cp -n /root/BayesianFootball/.cache/datastore_Scottish*.jls "$TARGET_DIR/.cache/" 2>/dev/null || true
elif compgen -G "/root/BayesianFootball-scotland-cross-tier/.cache/datastore_Scottish*.jls" > /dev/null; then
    cp -n /root/BayesianFootball-scotland-cross-tier/.cache/datastore_Scottish*.jls "$TARGET_DIR/.cache/" 2>/dev/null || true
fi

echo "$TARGET_DIR"
