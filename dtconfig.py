"""Read generation settings out of the Draw Things app.

Draw Things stores every generation as a `TensorHistoryNode` flatbuffer row in
per-project SQLite workspaces inside its container. That row contains the whole
preset: model, LoRAs, steps, sampler, size, guidance, hires-fix, prompts - i.e.
everything we need to replay a look you tuned in the app.

Usage:  python dtconfig.py --list [--limit 20]
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gen"))

import TensorHistoryNode as THN  # noqa: E402

DOCS_DIR = Path(
    os.environ.get(
        "DT_DOCS_DIR",
        str(Path.home() / "Library/Containers/com.liuliu.draw-things/Data/Documents"),
    )
)

SAMPLER_NAMES = [
    "DPMPP2MKarras", "EulerA", "DDIM", "PLMS", "DPMPPSDEKarras", "UniPC", "LCM",
    "EulerASubstep", "DPMPPSDESubstep", "TCD", "EulerATrailing", "DPMPPSDETrailing",
    "DPMPP2MAYS", "EulerAAYS", "DPMPPSDEAYS", "DPMPP2MTrailing", "DDIMTrailing",
    "UniPCTrailing", "UniPCAYS", "TCDTrailing",
]
SEED_MODE_NAMES = ["Legacy", "TorchCpuCompatible", "ScaleAlike", "NvidiaGpuCompatible"]
LORA_MODE_NAMES = ["all", "base", "refiner"]
COLOR_CALIBRATION_NAMES = ["none", "lab"]


def _name(table: list[str], index: int, default: str = "") -> str:
    return table[index] if 0 <= index < len(table) else default


def _str(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def workspaces() -> list[Path]:
    try:
        return sorted(DOCS_DIR.glob("*.sqlite3"))
    except OSError:
        return []


def _rows(db: Path, limit: int) -> list[tuple[int, bytes]]:
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute(
            "select rowid, p from tensorhistorynode order by rowid desc limit ?", (limit,)
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [(int(rowid), blob) for rowid, blob in rows if blob]


def _entry(db: Path, rowid: int, blob: bytes) -> dict[str, Any] | None:
    try:
        node = THN.TensorHistoryNode.GetRootAs(blob, 0)
        model = (node.Model() or b"").decode("utf-8", "replace")
        if not model:
            return None
        loras = []
        for i in range(node.LorasLength()):
            lora = node.Loras(i)
            if lora is None:
                continue
            loras.append(
                {
                    "file": (lora.File() or b"").decode("utf-8", "replace"),
                    "weight": float(lora.Weight()),
                    "mode": _name(LORA_MODE_NAMES, lora.Mode(), "all"),
                }
            )
        return {
            "source": db.name,
            "rowid": rowid,
            "model": model,
            "prompt": _str(node.TextPrompt()),
            "negative": _str(node.NegativeTextPrompt()),
            "width": int(node.StartWidth()) * 64,
            "height": int(node.StartHeight()) * 64,
            "steps": int(node.Steps()),
            "sampler": _name(SAMPLER_NAMES, int(node.Sampler()), "LCM"),
            "guidanceScale": float(node.GuidanceScale()),
            "seed": int(node.Seed()),
            "seedMode": _name(SEED_MODE_NAMES, int(node.SeedMode()), "ScaleAlike"),
            "clipSkip": int(node.ClipSkip()),
            "strength": float(node.Strength()),
            "hiresFix": bool(node.HiresFix()),
            "hiresFixWidth": int(node.HiresFixStartWidth()) * 64,
            "hiresFixHeight": int(node.HiresFixStartHeight()) * 64,
            "hiresFixStrength": float(node.HiresFixStrength()),
            "loras": loras,
            "upscaler": _str(node.Upscaler()),
            "faceRestoration": _str(node.FaceRestoration()),
            "refinerModel": _str(node.RefinerModel()),
            "refinerStart": float(node.RefinerStart()),
            "maskBlur": float(node.MaskBlur()),
            "maskBlurOutset": int(node.MaskBlurOutset()),
            "sharpness": float(node.Sharpness()),
            "shift": float(node.Shift()),
            "tiledDecoding": bool(node.TiledDecoding()),
            "tiledDiffusion": bool(node.TiledDiffusion()),
            "preserveOriginalAfterInpaint": bool(node.PreserveOriginalAfterInpaint()),
            "colorCalibration": _name(COLOR_CALIBRATION_NAMES, int(node.ColorCalibration()), "none"),
            "batchSize": int(node.BatchSize()),
            "numFrames": int(node.NumFrames()) or 1,
            "generationTime": float(node.GenerationTime()),
            "wallClock": int(node.WallClock()),
            "modified": _stamp(int(node.WallClock())),
        }
    except Exception:  # noqa: BLE001 - rows from older/newer builds are skipped
        return None


def _stamp(wall_clock: int) -> str:
    """Draw Things stores wall_clock in microseconds since the epoch."""
    if wall_clock <= 0:
        return ""
    try:
        return time.strftime("%m-%d %H:%M", time.localtime(wall_clock / 1_000_000))
    except (OverflowError, OSError, ValueError):
        return ""


def recent_configs(limit: int = 40) -> list[dict[str, Any]]:
    """Newest-first list of generation presets across all Draw Things projects."""
    entries: list[dict[str, Any]] = []
    for db in workspaces():
        for rowid, blob in _rows(db, limit):
            entry = _entry(db, rowid, blob)
            if entry is not None:
                entries.append(entry)
    entries.sort(key=lambda entry: entry.get("wallClock", 0), reverse=True)
    return entries[:limit]


_LORA_TAG = re.compile(r"<lora:[^>]*>", re.IGNORECASE)


def clean_prompt(text: str) -> str:
    """Drop inline <lora:...> tags - the LoRA list already carries those."""
    return _LORA_TAG.sub("", text or "").strip()


def bridge_settings(entry: dict[str, Any]) -> dict[str, Any]:
    """Subset of a Draw Things preset that the bridge's config.json accepts."""
    keys = (
        "model", "steps", "guidanceScale", "strength", "sampler", "seedMode", "seed",
        "clipSkip", "shift", "sharpness", "maskBlur", "maskBlurOutset", "width", "height",
        "batchSize", "hiresFix", "hiresFixStrength", "hiresFixWidth", "hiresFixHeight",
        "upscaler", "faceRestoration", "refinerModel", "refinerStart", "colorCalibration",
        "tiledDecoding", "tiledDiffusion", "preserveOriginalAfterInpaint", "numFrames", "loras",
    )
    return {key: entry[key] for key in keys if key in entry}


if __name__ == "__main__":
    limit = 20
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    entries = recent_configs(limit)
    if "--json" in sys.argv:
        print(json.dumps(entries, indent=2))
    else:
        print(f"{len(entries)} presets from {DOCS_DIR}")
        for i, entry in enumerate(entries):
            loras = len(entry["loras"])
            print(
                f"{i:2d}  {entry['modified']:<11} {entry['source'][:22]:<22} #{entry['rowid']:<4} "
                f"{entry['width']}x{entry['height']} {entry['steps']}st "
                f"{entry['sampler']:<14} cfg{entry['guidanceScale']:.1f} "
                f"lora{loras}  {entry['model'][:30]:<30} {entry['prompt'][:40]!r}"
            )
