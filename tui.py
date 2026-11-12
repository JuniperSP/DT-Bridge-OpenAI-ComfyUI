#!/usr/bin/env python3
"""Full-screen terminal UI for the Draw Things bridge.

Keys: s start · x stop · r restart · g/b switch log · c clear log view
      p test-generate · o open output folder · q quit
Stdlib only (curses) - no browser, no extra dependencies.
"""

from __future__ import annotations

import curses
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import netutil
from dtconfig import bridge_settings, clean_prompt, recent_configs

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".run"
OUTPUT = ROOT / "output"
DT = str(ROOT / "dt")

GRPC_PORT = int(os.environ.get("DT_GRPC_PORT", "7859"))
BRIDGE_PORT = int(os.environ.get("DT_BRIDGE_PORT", "9955"))
DT_MODELS_DIR = Path(
    os.environ.get(
        "DT_MODELS_DIR",
        str(Path.home() / "Library/Containers/com.liuliu.draw-things/Data/Documents/Models"),
    )
)
LOCAL_MODELS_DIR = ROOT / "models"

S: dict[str, Any] = {
    "busy": False,
    "message": "ready",
    "tab": "bridge",
    "log": "",
    "status": None,
    "input_target": None,
    "buffer": "",
    "last_image": None,
    "started": time.time(),
    "log_offset": 0,
    "dirty": True,
    "picker": None,
    "preset_prompt": "",
    "net": {"open": False, "mode": 1, "key": "", "https": False},
    "models": {"open": False, "extra_dir": ""},
}


# ----------------------------------------------------------------------------- data


def pid_alive(name: str) -> bool:
    try:
        pid = int((STATE / f"{name}.pid").read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def pid_of(name: str) -> str:
    try:
        return (STATE / f"{name}.pid").read_text().strip()
    except OSError:
        return ""


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def health() -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"{bridge_base()}/healthz", timeout=1.5) as res:
            return json.loads(res.read())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


_base_cache: dict[str, Any] = {"base": None, "at": 0.0}


def _candidates() -> list[str]:
    hosts = ["127.0.0.1"]
    ts = netutil.tailscale_ip()
    if ts:
        hosts.append(ts)
    hosts.append(netutil.lan_ip())
    return [f"http://{host}:{BRIDGE_PORT}" for host in hosts]


def bridge_base() -> str:
    """The first base URL that actually answers /healthz (bind may be tailnet-only)."""
    now = time.time()
    if _base_cache["base"] and now - _base_cache["at"] < 10:
        return str(_base_cache["base"])
    for base in _candidates():
        try:
            with urllib.request.urlopen(f"{base}/healthz", timeout=1) as res:
                if res.status == 200:
                    _base_cache.update(base=base, at=now)
                    return base
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            continue
    return _candidates()[0]


def net_modes() -> list[tuple[str, str]]:
    ts = netutil.tailscale_ip()
    modes = [("local only", "127.0.0.1"), ("LAN + tailnet", "0.0.0.0")]
    if ts:
        modes.append((f"tailnet only ({ts})", ts))
    return modes


def net_host() -> str:
    modes = net_modes()
    index = max(0, min(S["net"]["mode"], len(modes) - 1))
    return modes[index][1]


def load_bridge_env() -> None:
    path = ROOT / "bridge.env"
    if not path.is_file():
        return
    values = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    except OSError:
        return
    S["net"]["key"] = values.get("BRIDGE_API_KEY", "")
    S["models"]["extra_dir"] = values.get("DT_EXTRA_MODELS_DIR", "")
    host = values.get("BRIDGE_HOST", "0.0.0.0")
    for i, (_, candidate) in enumerate(net_modes()):
        if candidate == host:
            S["net"]["mode"] = i
            break
    S["net"]["https"] = bool(netutil.serve_state(BRIDGE_PORT).get("enabled"))


def write_bridge_env(host: str | None = None) -> bool:
    if host is None:
        host = net_host()
    lines = [
        "# written by the dt TUI - safe to edit by hand",
        f"BRIDGE_HOST={host}",
        f"BRIDGE_PORT={BRIDGE_PORT}",
        f"BRIDGE_API_KEY={S['net']['key']}",
        f"DT_EXTRA_MODELS_DIR={S['models']['extra_dir']}",
    ]
    try:
        (ROOT / "bridge.env").write_text("\n".join(lines) + "\n")
        return True
    except OSError as exc:
        S["message"] = f"cannot write bridge.env: {exc}"
        return False


def apply_network() -> None:
    host = net_host()
    https = bool(S["net"]["https"])
    if https and host not in ("127.0.0.1", "0.0.0.0"):
        # `tailscale serve` proxies to 127.0.0.1, so a tailnet-only bind would
        # make the HTTPS endpoint unreachable.
        host = "0.0.0.0"
    if not write_bridge_env(host):
        return

    S["net"]["open"] = False
    S["busy"] = True
    S["message"] = "applying network settings…"
    S["dirty"] = True

    def work() -> None:
        ok_serve, output = (netutil.serve_enable(BRIDGE_PORT) if https else netutil.serve_disable())
        subprocess.run([DT, "restart-bridge"], capture_output=True, text=True, timeout=300, cwd=str(ROOT), check=False)
        url = netutil.https_url() if https else None
        if https and ok_serve:
            S["message"] = f"https on · {url}  (bind {host}:{BRIDGE_PORT})"
        elif https:
            S["message"] = f"tailscale serve failed: {output.splitlines()[-1][:80] if output else 'unknown'}"
        else:
            S["message"] = f"https off · bind {host}:{BRIDGE_PORT}"
        _base_cache.update(base=None, at=0.0)
        S["busy"] = False
        S["dirty"] = True

    threading.Thread(target=work, daemon=True).start()


def run_models_sync() -> None:
    if S["busy"]:
        return
    S["busy"] = True
    S["message"] = "syncing custom models into Draw Things…"
    S["dirty"] = True

    def work() -> None:
        try:
            proc = subprocess.run(
                [DT, "models", "sync"], capture_output=True, text=True, timeout=60, cwd=str(ROOT), check=False
            )
            lines = [line.strip() for line in (proc.stdout or proc.stderr).splitlines() if line.strip()]
            S["message"] = lines[0] if lines else "models synced"
        except (OSError, subprocess.SubprocessError) as exc:
            S["message"] = f"models sync failed: {exc}"
        finally:
            S["busy"] = False
            S["dirty"] = True

    threading.Thread(target=work, daemon=True).start()


def collect() -> dict[str, Any]:
    return {
        "grpc_alive": pid_alive("grpc"),
        "grpc_pid": pid_of("grpc"),
        "grpc_port": port_open(GRPC_PORT),
        "bridge_alive": pid_alive("bridge"),
        "bridge_pid": pid_of("bridge"),
        "bridge_port": port_open(BRIDGE_PORT),
        "health": health(),
        "lan": netutil.lan_ip(),
        "tailnet": netutil.tailscale_ip(),
        "dns": netutil.tailscale_dns(),
        "serve": netutil.serve_state(BRIDGE_PORT),
        "https": netutil.https_url(),
    }


def tail_lines(path: Path, limit: int = 400) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 400_000))
            data = handle.read().decode("utf-8", "replace")
    except OSError:
        return "(no log yet)"
    # Polling noise (the TUI hits /healthz every tick) is not interesting.
    lines = [
        line
        for line in data.splitlines()
        if "GET /healthz" not in line and "GET /v1/models" not in line
    ]
    return "\n".join(lines[-limit:])


# ---------------------------------------------------------------------------- actions


def run_action(name: str) -> None:
    if S["busy"]:
        return
    S["busy"] = True
    S["message"] = f"{name}…"
    S["dirty"] = True

    def work() -> None:
        try:
            proc = subprocess.run(
                [DT, name], capture_output=True, text=True, timeout=600, cwd=str(ROOT), check=False
            )
            lines = [line.strip() for line in (proc.stdout or proc.stderr).splitlines() if line.strip()]
            S["message"] = lines[-1] if lines else f"{name} done"
        except (OSError, subprocess.SubprocessError) as exc:
            S["message"] = f"{name} failed: {exc}"
        finally:
            S["busy"] = False
            S["dirty"] = True

    threading.Thread(target=work, daemon=True).start()


def run_generate(prompt: str) -> None:
    if S["busy"] or not prompt.strip():
        return
    S["busy"] = True
    S["message"] = "generating…"
    S["dirty"] = True

    def work() -> None:
        payload = json.dumps({"prompt": prompt, "response_format": "url"}).encode()
        request = urllib.request.Request(
            f"{bridge_base()}/v1/images/generations",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=900) as res:
                data = json.loads(res.read())
            name = data["data"][0]["url"].rsplit("/", 1)[-1]
            S["last_image"] = name
            S["message"] = f"saved output/{name}  (press o to reveal)"
        except urllib.error.HTTPError as exc:
            S["message"] = f"generate failed: {exc.code} {exc.read().decode('utf-8', 'replace')[:120]}"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            S["message"] = f"generate failed: {exc}"
        finally:
            S["busy"] = False
            S["dirty"] = True

    threading.Thread(target=work, daemon=True).start()


def _match(entry: dict[str, Any], query: str) -> bool:
    if not query:
        return True
    hay = " ".join(
        [
            entry.get("prompt", ""),
            entry.get("model", ""),
            entry.get("sampler", ""),
            entry.get("source", ""),
            " ".join(lora["file"] for lora in entry.get("loras") or []),
        ]
    ).lower()
    return all(term in hay for term in query.lower().split())


def refilter(picker: dict[str, Any]) -> None:
    picker["filtered"] = [entry for entry in picker["items"] if _match(entry, picker["query"])]
    picker["index"] = 0
    picker["offset"] = 0


def open_picker() -> None:
    if S["busy"] or S["picker"] is not None:
        return
    S["busy"] = True
    S["message"] = "reading Draw Things presets…"
    S["dirty"] = True

    def work() -> None:
        try:
            items = [entry for entry in recent_configs(60) if (entry.get("prompt") or "").strip()]
            if items:
                S["picker"] = {
                    "items": items,
                    "filtered": items,
                    "query": "",
                    "index": 0,
                    "offset": 0,
                }
                S["message"] = f"{len(items)} presets · newest first · type to filter"
            else:
                S["message"] = "no presets with a prompt found in Draw Things"
        except Exception as exc:  # noqa: BLE001 - reported in the status line
            S["message"] = f"cannot read presets: {exc}"
        finally:
            S["busy"] = False
            S["dirty"] = True

    threading.Thread(target=work, daemon=True).start()


def apply_preset(entry: dict[str, Any]) -> None:
    S["busy"] = True
    S["message"] = "applying preset…"
    S["dirty"] = True
    settings = bridge_settings(entry)
    # A stored seed would freeze every generation; keep it random instead.
    settings.pop("seed", None)
    S["preset_prompt"] = clean_prompt(entry.get("prompt", ""))

    def work() -> None:
        label = f"{settings.get('width')}x{settings.get('height')} · {settings.get('steps')} steps · {settings.get('sampler')}"
        try:
            body = json.dumps(settings).encode()
            request = urllib.request.Request(
                f"{bridge_base()}/admin/config",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=15) as res:
                json.loads(res.read())
            S["message"] = f"loaded {entry['source']}#{entry['rowid']} · {label}"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            # Bridge down: keep the file so the next start picks it up.
            try:
                (ROOT / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
                S["message"] = f"bridge down ({type(exc).__name__}) - saved config.json · {label}"
            except OSError as write_exc:
                S["message"] = f"apply failed: {write_exc}"
        finally:
            S["busy"] = False
            S["dirty"] = True

    threading.Thread(target=work, daemon=True).start()


def draw_picker(scr: Any) -> None:
    picker = S["picker"]
    items = picker["filtered"]
    index = picker["index"]
    height, width = scr.getmaxyx()
    box_w = max(40, min(width - 4, 120))
    box_h = max(8, min(len(items) + 5, height - 4))
    top = max(1, (height - box_h) // 2)
    left = max(1, (width - box_w) // 2)

    for i in range(box_h):
        put(scr, top + i, left, "│" + " " * (box_w - 2) + "│")
    put(scr, top, left, "┌" + "─" * (box_w - 2) + "┐")
    put(scr, top + box_h - 1, left, "└" + "─" * (box_w - 2) + "┘")
    header = f" Load preset · {len(items)}/{len(picker['items'])} · newest first "
    put(scr, top, left + 2, header, curses.A_BOLD | curses.color_pair(4))

    query = picker["query"]
    put(scr, top + 1, left + 2, "filter: ", curses.color_pair(5))
    put(scr, top + 1, left + 10, (query or "(type to filter prompt / model / lora)")[: box_w - 14],
        curses.A_REVERSE if query else curses.color_pair(5))

    visible = box_h - 4
    offset = picker["offset"]
    if index < offset:
        offset = index
    elif index >= offset + visible:
        offset = index - visible + 1
    picker["offset"] = offset

    if not items:
        put(scr, top + 3, left + 2, "no preset matches", curses.color_pair(3))
    for row in range(visible):
        i = offset + row
        if i >= len(items):
            break
        entry = items[i]
        source = (entry.get("source") or "").removesuffix(".sqlite3")[:13]
        text = (
            f"{entry.get('modified', ''):<11} {source:<13}"
            f"{entry['width']}x{entry['height']:<5} {entry['steps']:>2}st "
            f"{entry['sampler']:<13} l{len(entry['loras'])} "
            f"{entry['prompt'][:34]}"
        )
        attr = curses.A_REVERSE if i == index else 0
        put(scr, top + 2 + row, left + 2, " " * (box_w - 4), attr)
        put(scr, top + 2 + row, left + 2, text, attr)

    put(scr, top + box_h - 2, left + 2,
        "type to filter · ↑↓ select · Enter load · Esc cancel", curses.color_pair(5))


def draw_net(scr: Any) -> None:
    modes = net_modes()
    mode_label = modes[max(0, min(S["net"]["mode"], len(modes) - 1))][0]
    key = S["net"]["key"]
    lan = netutil.lan_ip()
    ts = netutil.tailscale_ip()
    dns = netutil.tailscale_dns()

    height, width = scr.getmaxyx()
    box_w = max(48, min(width - 4, 84))
    box_h = 15
    top = max(1, (height - box_h) // 2)
    left = max(1, (width - box_w) // 2)

    for i in range(box_h):
        put(scr, top + i, left, "│" + " " * (box_w - 2) + "│")
    put(scr, top, left, "┌" + "─" * (box_w - 2) + "┐")
    put(scr, top + box_h - 1, left, "└" + "─" * (box_w - 2) + "┘")
    put(scr, top, left + 2, " Network ", curses.A_BOLD | curses.color_pair(4))

    def line(offset: int, label: str, value: str, attr: int = 0) -> None:
        put(scr, top + 1 + offset, left + 3, label)
        put(scr, top + 1 + offset, left + 16, value, attr)

    line(0, "Bind", f"{mode_label}  ({net_host()})   [b] cycle")
    line(1, "Port", f"{BRIDGE_PORT}   (edit bridge.env to change)")
    line(2, "API key", ("set" if key else "not set") + "   [k] set  [d] clear",
         curses.color_pair(1) if key else curses.color_pair(3))
    https_on = bool(S["net"]["https"])
    https_val = (netutil.https_url() or "no tailscale dns") if https_on else "off   [t] enable"
    line(3, "Tailscale HTTPS", https_val, curses.color_pair(1) if https_on else curses.color_pair(3))
    line(5, "LAN", lan)
    line(6, "Tailscale", (ts or "not detected") + (f"  {dns}" if dns else ""))
    line(8, "OpenAI (https)", (netutil.https_url() or "-") + "/v1", curses.color_pair(4))
    line(9, "OpenAI (lan)", f"http://{lan}:{BRIDGE_PORT}/v1", curses.color_pair(4))

    warn_text = "no API key: anyone on the tailnet can generate" if (not key and (https_on or net_host() != "127.0.0.1")) else ""
    if warn_text:
        put(scr, top + 11, left + 3, warn_text, curses.color_pair(3))
    put(scr, top + 12, left + 3, "[a] apply & restart bridge      [Esc] close", curses.color_pair(5))


def draw_models(scr: Any) -> None:
    extra = S["models"]["extra_dir"]
    try:
        dt_files = list(DT_MODELS_DIR.glob("*.ckpt")) if DT_MODELS_DIR.is_dir() else []
        symlinks = sum(1 for p in dt_files if p.is_symlink())
        dt_summary = f"{len(dt_files)} .ckpt files ({symlinks} linked)"
    except OSError:
        dt_summary = "unreadable (check Full Disk Access)"

    LOCAL_MODELS_DIR.mkdir(exist_ok=True)
    try:
        local_count = len(list(LOCAL_MODELS_DIR.glob("*.ckpt")) + list(LOCAL_MODELS_DIR.glob("*.safetensors")))
    except OSError:
        local_count = 0

    height, width = scr.getmaxyx()
    box_w = max(56, min(width - 4, 96))
    box_h = 15
    top = max(1, (height - box_h) // 2)
    left = max(1, (width - box_w) // 2)

    for i in range(box_h):
        put(scr, top + i, left, "│" + " " * (box_w - 2) + "│")
    put(scr, top, left, "┌" + "─" * (box_w - 2) + "┐")
    put(scr, top + box_h - 1, left, "└" + "─" * (box_w - 2) + "┘")
    put(scr, top, left + 2, " Models & Custom Folder ", curses.A_BOLD | curses.color_pair(4))

    def line(offset: int, label: str, value: str, attr: int = 0) -> None:
        put(scr, top + 1 + offset, left + 3, label)
        put(scr, top + 1 + offset, left + 20, value[: box_w - 23], attr)

    line(0, "DT Models dir", str(DT_MODELS_DIR), curses.color_pair(4))
    line(1, "Status", dt_summary, curses.color_pair(1))
    line(3, "Project drop", f"{LOCAL_MODELS_DIR}  ({local_count} file(s))")
    line(4, "Extra folder", (extra or "(not set)") + "   [f] set  [d] clear",
         curses.color_pair(1) if extra else curses.color_pair(3))
    line(6, "How it works", "Drop .ckpt / .safetensors into ./models or Extra folder.")
    line(7, "", "They are symlinked into Draw Things Models automatically.")

    put(scr, top + 10, left + 3, "[o] open DT Models in Finder    [p] open ./models in Finder", curses.A_BOLD)
    put(scr, top + 11, left + 3, "[s] sync / auto-link now        [f] set extra folder   [Esc] close", curses.color_pair(5))


# ------------------------------------------------------------------------------ draw


def put(scr: Any, y: int, x: int, text: str, attr: int = 0) -> None:
    height, width = scr.getmaxyx()
    if y < 0 or y >= height or x >= width:
        return
    text = text[: max(0, width - x - 1)]
    try:
        scr.addstr(y, x, text, attr)
    except curses.error:
        pass


def draw(scr: Any) -> None:
    scr.erase()
    height, width = scr.getmaxyx()
    data = S["status"] or {}
    h = data.get("health")

    title = f" Draw Things Bridge · {time.strftime('%H:%M:%S')} "
    put(scr, 0, 1, title, curses.A_BOLD | curses.color_pair(4))
    put(scr, 0, max(1, width - len(title) - 2), "q quit ", curses.color_pair(5))

    row = 1
    if data.get("grpc_alive"):
        dot, state_text, color = "●", f"running   pid {data.get('grpc_pid')}", 1
    elif data.get("grpc_port"):
        dot, state_text, color = "●", "running (external)", 3
    else:
        dot, state_text, color = "○", "stopped", 2
    put(scr, row, 2, f"{dot} gRPC server ", curses.color_pair(color))
    put(scr, row, 17, state_text, curses.color_pair(color))
    put(scr, row, 52, f"127.0.0.1:{GRPC_PORT}")
    row += 1

    if data.get("bridge_alive"):
        dot, state_text, color = "●", f"running   pid {data.get('bridge_pid')}", 1
    elif data.get("bridge_port"):
        dot, state_text, color = "●", "running (external)", 3
    else:
        dot, state_text, color = "○", "stopped", 2
    put(scr, row, 2, f"{dot} Bridge      ", curses.color_pair(color))
    put(scr, row, 17, state_text, curses.color_pair(color))
    put(scr, row, 52, f"0.0.0.0:{BRIDGE_PORT}")
    row += 1

    if h is None:
        put(scr, row, 2, "○ Health      ", curses.color_pair(2))
        put(scr, row, 17, "no answer from /healthz", curses.color_pair(2))
    elif h.get("models_ready"):
        put(scr, row, 2, "● Health      ", curses.color_pair(1))
        put(scr, row, 17, "ok · models ready", curses.color_pair(1))
        put(scr, row, 36, str(h.get("model", ""))[: max(10, width - 60)])
    elif not h.get("model"):
        put(scr, row, 2, "○ Health      ", curses.color_pair(3))
        put(scr, row, 17, "no preset loaded · press [l] to load from Draw Things", curses.color_pair(3))
    else:
        put(scr, row, 2, "● Health      ", curses.color_pair(2))
        put(scr, row, 17, "degraded · models missing", curses.color_pair(2))
        put(scr, row, 45, ",".join(h.get("missing") or [])[: max(10, width - 50)])
    row += 2

    ip = data.get("lan", "127.0.0.1")
    ts = data.get("tailnet")
    https_url = data.get("https")
    serve_on = bool((data.get("serve") or {}).get("enabled"))
    put(scr, row, 2, "OpenAI (LAN)", curses.A_BOLD)
    put(scr, row, 22, f"http://{ip}:{BRIDGE_PORT}/v1", curses.color_pair(4))
    row += 1
    if serve_on and https_url:
        put(scr, row, 2, "OpenAI (https)", curses.A_BOLD)
        put(scr, row, 22, f"{https_url}/v1", curses.color_pair(1))
        row += 1
    elif ts:
        put(scr, row, 2, "tailnet (http)", curses.A_BOLD)
        put(scr, row, 22, f"http://{ts}:{BRIDGE_PORT}/v1", curses.color_pair(4))
        row += 1
    put(scr, row, 2, "ComfyUI base", curses.A_BOLD)
    put(scr, row, 22, f"http://{ip}:{BRIDGE_PORT}" + (f"   or {https_url}" if serve_on and https_url else ""), curses.color_pair(4))
    row += 2

    bind_host = net_host()
    keys = "[s] start  [x] stop  [r] restart  [l] preset  [m] models  [n] network  [g]/[b] logs  [p] gen  [o] out  [q] quit"
    put(scr, row, 2, keys, curses.A_BOLD)
    row += 1
    put(scr, row, 2, f"bind {bind_host}:{BRIDGE_PORT}", curses.color_pair(5))
    row += 1

    tab_label = "grpc.log" if S["tab"] == "grpc" else "bridge.log"
    put(scr, row, 2, f"── {tab_label} " + "─" * max(0, width - len(tab_label) - 8), curses.color_pair(5))
    row += 1

    # Message line sits just above the log so it is always visible.
    msg_color = 3 if S["busy"] else 0
    put(scr, height - 2, 2, ("⏳ " if S["busy"] else "» ") + S["message"], curses.color_pair(msg_color))

    log_height = max(0, height - row - 2)
    lines = (S["log"] or "(no log yet)").splitlines()[-log_height:]
    for i, line in enumerate(lines):
        put(scr, row + i, 2, line, curses.color_pair(0))

    if S["picker"] is not None:
        draw_picker(scr)

    if S["net"]["open"]:
        draw_net(scr)

    if S["models"]["open"]:
        draw_models(scr)

    if S["input_target"]:
        prefixes = {"apikey": "api key> ", "models_dir": "extra models folder> "}
        prefix = prefixes.get(S["input_target"], "prompt> ")
        put(scr, height - 3, 2, " " * max(0, width - 4))
        put(scr, height - 3, 2, prefix + S["buffer"], curses.A_REVERSE)

    scr.refresh()


# ------------------------------------------------------------------------------ main


def main(scr: Any) -> None:
    curses.curs_set(0)
    load_bridge_env()
    try:
        curses.use_default_colors()
        for i, color in ((1, curses.COLOR_GREEN), (2, curses.COLOR_RED), (3, curses.COLOR_YELLOW),
                         (4, curses.COLOR_CYAN), (5, curses.COLOR_BLUE)):
            curses.init_pair(i, color, -1)
    except curses.error:
        pass

    scr.timeout(200)
    last_status = 0.0
    last_log = 0.0

    while True:
        now = time.time()
        if now - last_status > 2.5:
            S["status"] = collect()
            last_status = now
            S["dirty"] = True
        if now - last_log > 2.0:
            S["log"] = tail_lines(STATE / ("grpc.log" if S["tab"] == "grpc" else "bridge.log"))
            last_log = now
            S["dirty"] = True

        if S["dirty"]:
            draw(scr)
            S["dirty"] = False

        key = scr.getch()
        if key == -1:
            continue
        S["dirty"] = True

        if S["input_target"]:
            target, buffer = S["input_target"], S["buffer"]
            if key in (10, 13, curses.KEY_ENTER):
                S["buffer"] = ""
                S["input_target"] = None
                if target == "apikey":
                    S["net"]["key"] = buffer.strip()
                    S["message"] = "api key set (press a to apply)"
                elif target == "models_dir":
                    expanded = str(Path(buffer.strip()).expanduser()) if buffer.strip() else ""
                    if expanded and not Path(expanded).is_dir():
                        S["message"] = f"folder not found: {expanded}"
                    else:
                        S["models"]["extra_dir"] = expanded
                        write_bridge_env()
                        run_models_sync()
                else:
                    run_generate(buffer)
            elif key == 27:
                S["input_target"] = None
                S["buffer"] = ""
                S["message"] = "cancelled"
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                S["buffer"] = S["buffer"][:-1]
            elif 32 <= key < 127:
                S["buffer"] += chr(key)
            continue

        if S["picker"] is not None:
            picker = S["picker"]
            items = picker["filtered"]
            if key in (curses.KEY_UP,):
                picker["index"] = max(0, picker["index"] - 1)
            elif key in (curses.KEY_DOWN,):
                picker["index"] = max(0, min(len(items) - 1, picker["index"] + 1))
            elif key in (curses.KEY_PPAGE,):
                picker["index"] = max(0, picker["index"] - 10)
            elif key in (curses.KEY_NPAGE,):
                picker["index"] = max(0, min(len(items) - 1, picker["index"] + 10))
            elif key in (10, 13, curses.KEY_ENTER):
                if items:
                    entry = items[picker["index"]]
                    S["picker"] = None
                    apply_preset(entry)
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                picker["query"] = picker["query"][:-1]
                refilter(picker)
            elif key == 27:
                if picker["query"]:
                    picker["query"] = ""
                    refilter(picker)
                else:
                    S["picker"] = None
                    S["message"] = "cancelled"
            elif 32 <= key < 127:
                picker["query"] += chr(key)
                refilter(picker)
            continue

        if S["net"]["open"]:
            if key == ord("b"):
                S["net"]["mode"] = (S["net"]["mode"] + 1) % len(net_modes())
            elif key == ord("k"):
                S["input_target"] = "apikey"
                S["buffer"] = S["net"]["key"]
            elif key == ord("d"):
                S["net"]["key"] = ""
                S["message"] = "api key cleared (press a to apply)"
            elif key == ord("t"):
                S["net"]["https"] = not S["net"]["https"]
                if S["net"]["https"] and net_host() == netutil.tailscale_ip():
                    S["net"]["mode"] = 1
                S["message"] = ("https on (press a to apply)" if S["net"]["https"] else "https off (press a to apply)")
            elif key == ord("a"):
                apply_network()
            elif key == 27:
                S["net"]["open"] = False
            continue

        if S["models"]["open"]:
            if key == ord("o"):
                subprocess.Popen(["open", str(DT_MODELS_DIR)])
                S["message"] = "opened Draw Things Models folder in Finder"
            elif key == ord("p"):
                LOCAL_MODELS_DIR.mkdir(exist_ok=True)
                subprocess.Popen(["open", str(LOCAL_MODELS_DIR)])
                S["message"] = "opened ./models drop-folder in Finder"
            elif key == ord("f"):
                S["input_target"] = "models_dir"
                S["buffer"] = S["models"]["extra_dir"]
            elif key == ord("d"):
                S["models"]["extra_dir"] = ""
                write_bridge_env()
                S["message"] = "extra models folder cleared"
            elif key == ord("s"):
                run_models_sync()
            elif key == 27:
                S["models"]["open"] = False
            continue

        if key in (ord("q"),):
            return
        if key == ord("s"):
            run_action("start")
        elif key == ord("x"):
            run_action("stop")
        elif key == ord("r"):
            run_action("restart")
        elif key == ord("g"):
            S["tab"] = "grpc"
        elif key == ord("b"):
            S["tab"] = "bridge"
        elif key == ord("p"):
            S["input_target"] = "prompt"
            S["buffer"] = S["preset_prompt"]
        elif key == ord("l"):
            open_picker()
        elif key == ord("m"):
            S["models"]["open"] = True
        elif key == ord("n"):
            S["net"]["open"] = True
        elif key == ord("o"):
            subprocess.Popen(["open", str(OUTPUT)])
        elif key == ord("c"):
            S["message"] = "log view cleared"
            S["log"] = ""


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        print(json.dumps(collect(), indent=2))
    else:
        curses.wrapper(main)
