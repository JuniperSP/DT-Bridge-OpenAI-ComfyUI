# /// script
# dependencies = [
#     "fastapi",
#     "uvicorn",
# ]
# ///
"""Tiny control panel for the Draw Things bridge.

Serves one page with start/stop buttons, live status, logs and a test-generate
form. All actions shell out to the `dt` script in this directory.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".run"
OUTPUT = ROOT / "output"
DT = str(ROOT / "dt")

GRPC_PORT = int(os.environ.get("DT_GRPC_PORT", "7859"))
BRIDGE_PORT = int(os.environ.get("DT_BRIDGE_PORT", "9955"))
UI_PORT = int(os.environ.get("DT_UI_PORT", "9700"))

app = FastAPI(title="DT Bridge Control")


class _QuietPolling(logging.Filter):
    """Keep the page's 2s polling out of the log file."""

    NOISY = ("/api/status", "/api/logs")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(path in message for path in self.NOISY)


logging.getLogger("uvicorn.access").addFilter(_QuietPolling())


def _pid_alive(name: str) -> bool:
    pid_file = STATE / f"{name}.pid"
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _health() -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{BRIDGE_PORT}/healthz", timeout=2) as res:
            return json.loads(res.read())
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None


def _lan_ip() -> str:
    for iface in ("en0", "en1"):
        try:
            out = subprocess.run(
                ["ipconfig", "getifaddr", iface], capture_output=True, text=True, timeout=3, check=False
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out = ""
        if out:
            return out
    return "127.0.0.1"


@app.get("/api/status")
def api_status() -> dict:
    health = _health()
    return {
        "grpc": {"running": _pid_alive("grpc") or _port_open(GRPC_PORT), "managed": _pid_alive("grpc"), "port": GRPC_PORT},
        "bridge": {"running": _pid_alive("bridge"), "port": BRIDGE_PORT},
        "ui": {"port": UI_PORT},
        "health": health,
        "lan": _lan_ip(),
    }


@app.post("/api/action/{name}")
def api_action(name: str) -> dict:
    commands = {"start": ["start"], "stop": ["stop"], "restart": ["restart"]}
    if name not in commands:
        raise HTTPException(status_code=404, detail="unknown action")
    try:
        proc = subprocess.run(
            [DT, *commands[name]], capture_output=True, text=True, timeout=240, cwd=str(ROOT), check=False
        )
    except subprocess.SubprocessError as exc:
        return {"ok": False, "output": str(exc)}
    text = (proc.stdout or "") + (proc.stderr or "")
    return {"ok": proc.returncode == 0, "output": text}


@app.get("/api/logs")
def api_logs(which: str = "bridge", lines: int = 200) -> dict:
    path = STATE / ("grpc.log" if which == "grpc" else "bridge.log")
    if not path.is_file():
        return {"text": "(no log yet)"}
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 200_000))
        data = handle.read().decode("utf-8", "replace").splitlines()[-lines:]
    return {"text": "\n".join(data)}


class GenerateBody(BaseModel):
    prompt: str
    size: str = "512x512"
    negative_prompt: str = ""


@app.post("/api/generate")
def api_generate(body: GenerateBody) -> dict:
    payload = json.dumps(
        {"prompt": body.prompt, "size": body.size, "negative_prompt": body.negative_prompt}
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{BRIDGE_PORT}/v1/images/generations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as res:
            data = json.loads(res.read())
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": exc.read().decode("utf-8", "replace")}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"ok": False, "error": str(exc)}
    image = data.get("data", [{}])[0]
    if "url" in image:
        return {"ok": True, "image": image["url"].rsplit("/", 1)[-1]}
    if "b64_json" in image:
        import base64

        name = f"ui-{os.urandom(4).hex()}.png"
        (OUTPUT / name).write_bytes(base64.b64decode(image["b64_json"]))
        return {"ok": True, "image": name}
    return {"ok": False, "error": "no image in response"}


@app.get("/api/image/{name}")
def api_image(name: str):
    path = OUTPUT / os.path.basename(name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Draw Things Bridge</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0f1115; --panel: #171a21; --line: #242a33;
    --fg: #e6e9ef; --muted: #8b93a1;
    --ok: #3ddc97; --bad: #ff5c5c; --accent: #5b9dff;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    padding: 28px; max-width: 1080px; margin-inline: auto;
  }
  h1 { font-size: 17px; font-weight: 600; margin: 0 0 2px; letter-spacing: .2px; }
  .sub { color: var(--muted); font-size: 12.5px; margin-bottom: 22px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
  .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
  .row + .row { margin-top: 10px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--bad); display: inline-block; margin-right: 8px; }
  .dot.on { background: var(--ok); }
  .name { font-weight: 500; }
  .meta { color: var(--muted); font-size: 12px; }
  button {
    font: inherit; font-size: 13px; color: var(--fg); background: #222834;
    border: 1px solid var(--line); border-radius: 8px; padding: 7px 14px; cursor: pointer;
  }
  button:hover { border-color: #3a4250; background: #262d3a; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #08111f; font-weight: 600; }
  button:disabled { opacity: .45; cursor: default; }
  .actions { display: flex; gap: 8px; flex-wrap: wrap; }
  code { font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; color: #b9c3d3; }
  .endpoint { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
  .endpoint + .endpoint { margin-top: 8px; }
  input, textarea, select {
    font: inherit; background: #11151c; color: var(--fg);
    border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; width: 100%;
  }
  textarea { resize: vertical; min-height: 62px; }
  .field { margin-bottom: 10px; }
  label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 5px; }
  pre { margin: 0; max-height: 260px; overflow: auto; font: 11.5px/1.45 ui-monospace, monospace;
        color: #9fb0c6; white-space: pre-wrap; word-break: break-all; }
  .tabs { display: flex; gap: 6px; margin-bottom: 10px; }
  .tabs button { padding: 4px 10px; font-size: 12px; }
  .tabs button.active { border-color: var(--accent); color: var(--accent); }
  #thumb { width: 100%; border-radius: 10px; margin-top: 12px; display: none; }
  .status-line { font-size: 12.5px; color: var(--muted); }
  .ok { color: var(--ok); } .bad { color: var(--bad); }
</style></head>
<body>
<h1>Draw Things Bridge</h1>
<div class="sub" id="sub">connecting…</div>

<div class="grid">
  <div class="panel">
    <div class="row"><span class="name">Services</span><span class="meta" id="managed"></span></div>
    <div class="row"><span><span class="dot" id="dot-grpc"></span>gRPC server</span><span class="meta" id="grpc-meta"></span></div>
    <div class="row"><span><span class="dot" id="dot-bridge"></span>Bridge (HTTP)</span><span class="meta" id="bridge-meta"></span></div>
    <div class="row"><span class="status-line" id="health">—</span></div>
    <div class="row actions" style="margin-top:14px">
      <button class="primary" onclick="act('start')">Start</button>
      <button onclick="act('restart')">Restart</button>
      <button onclick="act('stop')">Stop</button>
    </div>
  </div>

  <div class="panel">
    <div class="row"><span class="name">Endpoints</span></div>
    <div class="endpoint"><code id="ep-openai">—</code><button onclick="copy('ep-openai')">Copy</button></div>
    <div class="endpoint"><code id="ep-comfy">—</code><button onclick="copy('ep-comfy')">Copy</button></div>
    <div class="endpoint"><code id="ep-health">—</code><button onclick="copy('ep-health')">Copy</button></div>
  </div>

  <div class="panel">
    <div class="row"><span class="name">Test generate</span><span class="meta">uses the configured pipeline</span></div>
    <div class="field" style="margin-top:12px"><label>Prompt</label><textarea id="prompt">a red apple on a wooden table</textarea></div>
    <div class="field"><label>Negative</label><input id="negative" value=""></div>
    <div class="field"><label>Size</label>
      <select id="size">
        <option>512x512</option><option>512x768</option><option>768x512</option>
        <option selected>1024x1344</option><option>1024x1024</option>
      </select>
    </div>
    <button class="primary" id="gen" onclick="generate()">Generate</button>
    <img id="thumb" alt="">
  </div>

  <div class="panel">
    <div class="tabs">
      <button id="tab-bridge" class="active" onclick="showLog('bridge')">bridge.log</button>
      <button id="tab-grpc" onclick="showLog('grpc')">grpc.log</button>
    </div>
    <pre id="log">loading…</pre>
  </div>
</div>

<script>
let logTarget = 'bridge';
const $ = (id) => document.getElementById(id);

async function refresh() {
  try {
    const s = await (await fetch('/api/status')).json();
    $('dot-grpc').className = 'dot' + (s.grpc.running ? ' on' : '');
    $('dot-bridge').className = 'dot' + (s.bridge.running ? ' on' : '');
    $('grpc-meta').textContent = s.grpc.running ? (s.grpc.managed ? 'running' : 'external') : 'stopped';
    $('bridge-meta').textContent = s.bridge.running ? 'running' : 'stopped';
    $('managed').textContent = '127.0.0.1:' + s.grpc.port + ' → :' + s.bridge.port;
    $('sub').textContent = 'LAN ' + s.lan + ':' + s.bridge.port;

    const h = s.health;
    if (!h) {
      $('health').innerHTML = '<span class="bad">bridge not answering /healthz</span>';
    } else if (h.models_ready) {
      $('health').innerHTML = '<span class="ok">models ready</span> · ' + h.model;
    } else if (!h.model) {
      $('health').innerHTML = '<span class="bad">no preset loaded</span> · press <code>l</code> in <code>./dt tui</code> to load from Draw Things';
    } else {
      $('health').innerHTML = '<span class="bad">models missing:</span> ' + (h.missing || []).join(', ');
    }

    $('ep-openai').textContent = 'http://' + s.lan + ':' + s.bridge.port + '/v1';
    $('ep-comfy').textContent = 'http://' + s.lan + ':' + s.bridge.port;
    $('ep-health').textContent = 'curl -s http://127.0.0.1:' + s.bridge.port + '/healthz';
  } catch (e) {
    $('sub').textContent = 'control panel lost the bridge';
  }
}

async function refreshLog() {
  const r = await (await fetch('/api/logs?which=' + logTarget + '&lines=120')).json();
  const el = $('log');
  const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
  el.textContent = r.text;
  if (stick) el.scrollTop = el.scrollHeight;
}

function showLog(which) {
  logTarget = which;
  $('tab-bridge').className = which === 'bridge' ? 'active' : '';
  $('tab-grpc').className = which === 'grpc' ? 'active' : '';
  refreshLog();
}

async function act(name) {
  const buttons = document.querySelectorAll('.actions button');
  buttons.forEach(b => b.disabled = true);
  try {
    const r = await (await fetch('/api/action/' + name, {method: 'POST'})).json();
    if (!r.ok) console.warn(r.output);
  } finally {
    buttons.forEach(b => b.disabled = false);
    refresh(); refreshLog();
  }
}

async function generate() {
  const btn = $('gen');
  btn.disabled = true; btn.textContent = 'Generating…';
  try {
    const r = await (await fetch('/api/generate', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt: $('prompt').value, negative_prompt: $('negative').value, size: $('size').value})
    })).json();
    if (r.ok) {
      const img = $('thumb');
      img.src = '/api/image/' + r.image + '?t=' + Date.now();
      img.style.display = 'block';
    } else {
      alert('Generation failed:\n' + (r.error || 'unknown'));
    }
  } finally {
    btn.disabled = false; btn.textContent = 'Generate';
  }
}

function copy(id) {
  navigator.clipboard.writeText($(id).textContent);
}

refresh(); refreshLog();
setInterval(refresh, 2000);
setInterval(refreshLog, 3000);
</script>
</body></html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("DT_UI_HOST", "127.0.0.1"), port=UI_PORT)
