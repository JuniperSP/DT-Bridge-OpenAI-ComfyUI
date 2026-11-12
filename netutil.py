"""Network helpers: LAN address and Tailscale (tailnet) address.

No dependencies - shells out to `ifconfig` / `tailscale` like the rest of the
project does.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

TAILSCALE_CLI_CANDIDATES = (
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
    "/usr/local/bin/tailscale",
    "/opt/homebrew/bin/tailscale",
    "/usr/bin/tailscale",
)


def _run(argv: list[str], timeout: float = 3) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def lan_ip() -> str:
    for iface in ("en0", "en1"):
        out = _run(["ipconfig", "getifaddr", iface]).strip()
        if out:
            return out
    return "127.0.0.1"


def tailscale_bin() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    for candidate in TAILSCALE_CLI_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def tailscale_ip() -> str | None:
    """First 100.64.0.0/10 address on the machine (Tailscale's CGNAT range)."""
    out = _run(["ifconfig"])
    for match in re.finditer(r"inet (100\.(\d+)\.\d+\.\d+)", out):
        address, second = match.group(1), int(match.group(2))
        if 64 <= second <= 127:
            return address
    cli = tailscale_bin()
    if cli:
        out = _run([cli, "ip", "-4"]).strip()
        if out:
            return out.splitlines()[0].strip()
    return None


def tailscale_dns() -> str | None:
    cli = tailscale_bin()
    if not cli:
        return None
    out = _run([cli, "status", "--json"], timeout=5)
    if not out.strip():
        return None
    try:
        data = json.loads(out)
    except ValueError:
        return None
    name = (data.get("Self") or {}).get("DNSName") or ""
    return name.rstrip(".") or None


def https_url(https_port: int = 443) -> str | None:
    dns = tailscale_dns()
    if not dns:
        return None
    return f"https://{dns}" if https_port == 443 else f"https://{dns}:{https_port}"


def _serve_json() -> dict:
    cli = tailscale_bin()
    if not cli:
        return {}
    out = _run([cli, "serve", "status", "--json"], timeout=6)
    try:
        return json.loads(out) if out.strip() else {}
    except ValueError:
        return {}


def serve_state(bridge_port: int = 9955, https_port: int = 443) -> dict[str, object]:
    """Is `tailscale serve` exposing our bridge over HTTPS right now?"""
    want = f"http://127.0.0.1:{bridge_port}"
    web = (_serve_json().get("Web") or {})
    for key, config in web.items():
        port = key.rsplit(":", 1)[-1]
        handlers = (config or {}).get("Handlers") or {}
        proxy = ((handlers.get("/") or {}).get("Proxy") or "").rstrip("/")
        if port == str(https_port) and proxy == want:
            return {"enabled": True, "url": https_url(https_port), "target": proxy}
    return {"enabled": False, "url": https_url(https_port), "target": want}


def serve_enable(bridge_port: int = 9955, https_port: int = 443) -> tuple[bool, str]:
    cli = tailscale_bin()
    if not cli:
        return False, "tailscale CLI not found"
    try:
        proc = subprocess.run(
            [cli, "serve", "--bg", "--yes", f"--https={https_port}", f"http://127.0.0.1:{bridge_port}"],
            capture_output=True,
            text=True,
            timeout=40,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output.strip()


def serve_disable(https_port: int = 443) -> tuple[bool, str]:
    cli = tailscale_bin()
    if not cli:
        return False, "tailscale CLI not found"
    try:
        proc = subprocess.run(
            [cli, "serve", f"--https={https_port}", "off"],
            capture_output=True,
            text=True,
            timeout=40,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output.strip()


def summary(port: int = 9955) -> dict[str, object]:
    lan = lan_ip()
    ts = tailscale_ip()
    dns = tailscale_dns()
    return {
        "lan": lan,
        "tailscale": ts,
        "tailscale_dns": dns,
        "serve": serve_state(port),
        "urls": {
            "lan_openai": f"http://{lan}:{port}/v1",
            "lan_base": f"http://{lan}:{port}",
            "tailnet_openai": f"http://{ts}:{port}/v1" if ts else None,
            "tailnet_base": f"http://{ts}:{port}" if ts else None,
            "https_openai": (https_url() or "") + "/v1" if dns else None,
        },
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(summary(int(sys.argv[1]) if len(sys.argv) > 1 else 9955), indent=2))
