#!/usr/bin/env bash
# First-time setup for a fresh Mac. Safe to re-run: it only fills in what is
# missing.  Usage:  ./setup.sh [--yes] [--download-grpc]
#
#   --yes             install/download without asking
#   --download-grpc   fetch gRPCServerCLI-macOS from the pinned release
#   --tools           (re)download flatc + protoc and regenerate gen/
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

FLATC_RELEASE="v25.12.19-2026-02-06-03fffb2"
PROTOC_VERSION="36.2"
GRPC_RELEASE="v26.1023.0"
GRPC_BIN="$HOME/.local/bin/gRPCServerCLI-macOS"
MODELS_DIR="${DT_MODELS_DIR:-$HOME/Library/Containers/com.liuliu.draw-things/Data/Documents/Models}"
VENV_PY="$ROOT/.venv/bin/python"

ASSUME_YES=0
DOWNLOAD_GRPC=0
FORCE_TOOLS=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y) ASSUME_YES=1 ;;
    --download-grpc) DOWNLOAD_GRPC=1 ;;
    --tools) FORCE_TOOLS=1 ;;
    -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  esac
done

c_reset=$'\033[0m'; c_dim=$'\033[2m'; c_red=$'\033[31m'; c_green=$'\033[32m'; c_yellow=$'\033[33m'
ok() { printf '%s✔%s %s\n' "$c_green" "$c_reset" "$*"; }
warn() { printf '%s!%s %s\n' "$c_yellow" "$c_reset" "$*"; }
err() { printf '%s✘%s %s\n' "$c_red" "$c_reset" "$*" >&2; }
step() { printf '\n%s▸ %s%s\n' "$c_dim" "$*" "$c_reset"; }

ask() {
  [ "$ASSUME_YES" = "1" ] && return 0
  [ -t 0 ] || return 1
  printf '%s [y/N] ' "$1"
  read -r answer
  case "$answer" in y|Y|yes) return 0 ;; *) return 1 ;; esac
}

ARCH="$(uname -m)"
case "$ARCH" in
  arm64) FLATC_ASSET="Mac.flatc.binary.zip"; PROTOC_ASSET="protoc-${PROTOC_VERSION}-osx-aarch_64.zip" ;;
  x86_64) FLATC_ASSET="MacIntel.flatc.binary.zip"; PROTOC_ASSET="protoc-${PROTOC_VERSION}-osx-x86_64.zip" ;;
  *) err "unsupported architecture: $ARCH"; exit 1 ;;
esac

printf 'Draw Things bridge setup · %s\n' "$ROOT"
printf '%s macOS %s · %s%s\n' "$c_dim" "$(sw_vers -productVersion 2>/dev/null)" "$ARCH" "$c_reset"

# ---------------------------------------------------------------- python + uv
step "python + dependencies"

if ! command -v uv >/dev/null 2>&1; then
  warn "uv is not installed (used to manage the virtualenv)"
  if command -v brew >/dev/null 2>&1 && ask "Install uv with Homebrew?"; then
    brew install uv
  elif ask "Install uv with the official installer?"; then
    curl -fsSL https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

if command -v uv >/dev/null 2>&1; then
  ok "uv $(uv --version | awk '{print $2}')"
else
  err "uv missing - install it (https://docs.astral.sh/uv/) and re-run"
  exit 1
fi

if [ -x "$VENV_PY" ]; then
  ok "virtualenv present (.venv)"
else
  warn "creating .venv (uv will fetch a matching CPython if needed)"
  uv venv -p 3.12 "$ROOT/.venv" || uv venv "$ROOT/.venv" || { err "could not create .venv"; exit 1; }
  ok "virtualenv created"
fi

if uv pip install --python "$VENV_PY" -q -r "$ROOT/requirements.txt"; then
  ok "python dependencies installed"
else
  err "dependency install failed"
  exit 1
fi

# ------------------------------------------------------------------- bindings
step "generated bindings (gen/)"

if [ -f "$ROOT/gen/imageService_pb2.py" ] && [ -f "$ROOT/gen/GenerationConfiguration.py" ] && [ "$FORCE_TOOLS" = "0" ]; then
  ok "gen/ already generated"
else
  warn "gen/ is missing - downloading flatc + protoc to rebuild it"
  mkdir -p "$ROOT/tools"
  if [ ! -x "$ROOT/tools/flatc" ]; then
    curl -fsSL -o "$ROOT/tools/flatc.zip" \
      "https://github.com/google/flatbuffers/releases/download/${FLATC_RELEASE}/${FLATC_ASSET}" \
      && (cd "$ROOT/tools" && unzip -oq flatc.zip && rm -f flatc.zip) || { err "flatc download failed"; exit 1; }
    chmod +x "$ROOT/tools/flatc"
  fi
  if [ ! -x "$ROOT/tools/bin/protoc" ]; then
    curl -fsSL -o "$ROOT/tools/protoc.zip" \
      "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/${PROTOC_ASSET}" \
      && (cd "$ROOT/tools" && unzip -oq protoc.zip && rm -f protoc.zip) || { err "protoc download failed"; exit 1; }
  fi
  DT_REGEN_FROM_SETUP=1 "$ROOT/regen.sh" || { err "regen.sh failed"; exit 1; }
  ok "bindings regenerated"
fi

# --------------------------------------------------------------- grpc server
step "Draw Things gRPC server"

if [ -x "$GRPC_BIN" ]; then
  ok "found $GRPC_BIN"
elif [ -x "$ROOT/gRPCServerCLI-macOS" ]; then
  ok "found $ROOT/gRPCServerCLI-macOS"
else
  warn "gRPCServerCLI-macOS not found"
  if [ "$DOWNLOAD_GRPC" = "1" ] || ask "Download gRPCServerCLI-macOS (~275 MB) from the ${GRPC_RELEASE} release?"; then
    mkdir -p "$(dirname "$GRPC_BIN")"
    curl -fL --progress-bar -o "$GRPC_BIN" \
      "https://github.com/drawthingsai/draw-things-community/releases/download/${GRPC_RELEASE}/gRPCServerCLI-macOS" \
      && chmod +x "$GRPC_BIN" && ok "downloaded $GRPC_BIN" || err "download failed"
    xattr -dr com.apple.quarantine "$GRPC_BIN" 2>/dev/null || true
  else
    warn "skipped - ./dt start will need DT_GRPC_BIN pointing at the binary"
  fi
fi

# --------------------------------------------------------------------- models
step "models + permissions"

if [ ! -d "$MODELS_DIR" ]; then
  err "models directory not found: $MODELS_DIR"
  warn "run Draw Things once, or set DT_MODELS_DIR to your Models folder"
elif ls "$MODELS_DIR" >/dev/null 2>&1; then
  ok "models readable ($(ls "$MODELS_DIR" | wc -l | tr -d ' ') files)"
else
  err "cannot read $MODELS_DIR"
  printf '  Grant Full Disk Access to Terminal (or to this editor):\n'
  printf '  System Settings → Privacy & Security → Full Disk Access\n'
  printf '  Then restart the terminal and re-run ./setup.sh\n'
fi

# ------------------------------------------------------------------ tailscale
step "tailscale (optional, for HTTPS over your tailnet)"

CLI="$(command -v tailscale || echo /Applications/Tailscale.app/Contents/MacOS/Tailscale)"
if [ -x "$CLI" ]; then
  ts_ip="$("$ROOT/.venv/bin/python" -c "import netutil;print(netutil.tailscale_ip() or '')" 2>/dev/null)"
  if [ -n "$ts_ip" ]; then
    ok "tailnet address $ts_ip"
  else
    warn "tailscale installed but not connected - open the app and log in"
  fi
else
  warn "tailscale not installed - LAN access only"
fi

# ------------------------------------------------------------------- settings
step "bridge settings"

if [ -f "$ROOT/bridge.env" ]; then
  ok "bridge.env present"
else
  cat >"$ROOT/bridge.env" <<'ENV'
# written by setup.sh - safe to edit by hand
BRIDGE_HOST=127.0.0.1
BRIDGE_PORT=9955
BRIDGE_API_KEY=
ENV
  ok "created bridge.env (loopback only - open ./dt tui, press n to expose)"
fi

if [ -f "$ROOT/config.json" ]; then
  ok "config.json present"
else
  warn "no preset loaded yet (generate once in Draw Things, then press l in ./dt tui)"
fi

printf '\n%snext steps%s\n' "$c_green" "$c_reset"
printf '  1. Tune a model & generate 1 image in the Draw Things app\n'
printf '  2. Run ./dt tui -> press s (start) -> press l (load preset from Draw Things)\n'
printf '  3. ./dt url     print endpoints once it is running\n'
