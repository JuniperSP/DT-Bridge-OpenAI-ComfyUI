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
COMPRESSION_ARTIFACT_NAMES = ["disabled", "h264", "h265", "jpeg"]

# Entry-level metadata that is never part of a Draw Things configuration.
META_KEYS = frozenset(
    {
        "source",
        "rowid",
        "name",
        "prompt",
        "negative",
        "modified",
        "wallClock",
        "generationTime",
        "configuration",
    }
)


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
        controls = []
        for i in range(node.ControlsLength()):
            control = node.Controls(i)
            if control is None:
                continue
            controls.append(
                {
                    "file": _str(control.File()),
                    "weight": float(control.Weight()),
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
            # The bridge renders still images, and Draw Things leaves num_frames
            # at its schema default (14) even for stills.
            "numFrames": 1,
            "imageGuidanceScale": float(node.ImageGuidanceScale()),
            "clipWeight": float(node.ClipWeight()),
            "negativePromptForImagePrior": bool(node.NegativePromptForImagePrior()),
            "imagePriorSteps": int(node.ImagePriorSteps()),
            "originalImageWidth": int(node.OriginalImageWidth()),
            "originalImageHeight": int(node.OriginalImageHeight()),
            "targetImageWidth": int(node.TargetImageWidth()),
            "targetImageHeight": int(node.TargetImageHeight()),
            "negativeOriginalImageWidth": int(node.NegativeOriginalImageWidth()),
            "negativeOriginalImageHeight": int(node.NegativeOriginalImageHeight()),
            "cropTop": int(node.CropTop()),
            "cropLeft": int(node.CropLeft()),
            "aestheticScore": float(node.AestheticScore()),
            "negativeAestheticScore": float(node.NegativeAestheticScore()),
            "zeroNegativePrompt": bool(node.ZeroNegativePrompt()),
            "guidingFrameNoise": float(node.CondAug()),
            "fps": int(node.FpsId()),
            "motionScale": int(node.MotionBucketId()),
            "startFrameGuidance": float(node.StartFrameCfg()),
            "stage2Steps": int(node.Stage2Steps()),
            "stage2Guidance": float(node.Stage2Cfg()),
            "stage2Shift": float(node.Stage2Shift()),
            "stochasticSamplingGamma": float(node.StochasticSamplingGamma()),
            "upscalerScaleFactor": int(node.UpscalerScaleFactor()),
            "t5TextEncoder": bool(node.T5TextEncoder()),
            "separateT5": bool(node.SeparateT5()),
            "t5Text": _str(node.T5Text()),
            "separateClipL": bool(node.SeparateClipL()),
            "clipLText": _str(node.ClipLText()),
            "separateOpenClipG": bool(node.SeparateOpenClipG()),
            "openClipGText": _str(node.OpenClipGText()),
            "speedUpWithGuidanceEmbed": bool(node.SpeedUpWithGuidanceEmbed()),
            "guidanceEmbed": float(node.GuidanceEmbed()),
            "resolutionDependentShift": bool(node.ResolutionDependentShift()),
            "teaCache": bool(node.TeaCache()),
            "teaCacheStart": int(node.TeaCacheStart()),
            "teaCacheEnd": int(node.TeaCacheEnd()),
            "teaCacheThreshold": float(node.TeaCacheThreshold()),
            "teaCacheMaxSkipSteps": int(node.TeaCacheMaxSkipSteps()),
            "causalInference": int(node.CausalInference()),
            "causalInferencePad": int(node.CausalInferencePad()),
            "cfgZeroStar": bool(node.CfgZeroStar()),
            "cfgZeroInitSteps": int(node.CfgZeroInitSteps()),
            "compressionArtifacts": _name(
                COMPRESSION_ARTIFACT_NAMES, int(node.CompressionArtifacts()), "disabled"
            ),
            "compressionArtifactsQuality": float(node.CompressionArtifactsQuality()),
            "expandPromptToJson": bool(node.ExpandPromptToJson()),
            "shiftForAudio": float(node.ShiftForAudio()),
            "usesSolAttention": bool(node.UsesSolAttention()),
            "solAttentionStart": int(node.SolAttentionStart()),
            "solAttentionTau": float(node.SolAttentionTau()),
            # Tile sizes are stored in 64 px blocks here, in pixels in the app's
            # saved configurations; normalise to pixels like custom_configs().
            "decodingTileWidth": int(node.DecodingTileWidth()) * 64,
            "decodingTileHeight": int(node.DecodingTileHeight()) * 64,
            "decodingTileOverlap": int(node.DecodingTileOverlap()) * 64,
            "diffusionTileWidth": int(node.DiffusionTileWidth()) * 64,
            "diffusionTileHeight": int(node.DiffusionTileHeight()) * 64,
            "diffusionTileOverlap": int(node.DiffusionTileOverlap()) * 64,
            "controls": controls,
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

CUSTOM_CONFIG_PATHS = (
    DOCS_DIR / "Models/custom_configs.json",
    Path.home() / "Downloads/custom_configs.json",
)
TEXTUAL_INVERSION_PATHS = (
    DOCS_DIR / "Models/custom_textual_inversions.json",
    Path.home() / "Downloads/custom_textual_inversions.json",
)


def _first_existing(paths: tuple[Path, ...]) -> Path | None:
    for path in paths:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def custom_source() -> Path | None:
    return _first_existing(CUSTOM_CONFIG_PATHS)


def custom_configs() -> list[dict[str, Any]]:
    """Named presets saved in the app ("Save configuration").

    These are the only place Draw Things keeps a human-readable preset name, so
    they are the ones worth surfacing in the picker.
    """
    path = custom_source()
    if path is None:
        return []
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            continue
        config = item.get("configuration")
        if not isinstance(config, dict) or not config.get("model"):
            continue
        loras = []
        for lora in config.get("loras") or []:
            if isinstance(lora, dict) and lora.get("file"):
                loras.append(
                    {
                        "file": str(lora["file"]),
                        "weight": _float(lora.get("weight"), 1.0),
                        "mode": _name(LORA_MODE_NAMES, _int(lora.get("mode"), 0), "all"),
                    }
                )
        entries.append(
            {
                "source": "custom",
                "rowid": index,
                "name": str(item.get("name") or "").strip(),
                "prompt": "",
                "negative": "",
                "modified": "",
                "wallClock": 0,
                "model": str(config.get("model")),
                "width": _int(config.get("width"), 512),
                "height": _int(config.get("height"), 512),
                "steps": _int(config.get("steps"), 20),
                "sampler": _name(SAMPLER_NAMES, _int(config.get("sampler")), "EulerA"),
                "guidanceScale": _float(config.get("guidanceScale"), 7.0),
                "seed": _int(config.get("seed"), 0),
                "seedMode": _name(SEED_MODE_NAMES, _int(config.get("seedMode")), "Legacy"),
                "clipSkip": _int(config.get("clipSkip"), 1),
                "strength": _float(config.get("strength"), 1.0),
                "hiresFix": bool(config.get("hiresFix")),
                "hiresFixWidth": _int(config.get("hiresFixWidth")),
                "hiresFixHeight": _int(config.get("hiresFixHeight")),
                "hiresFixStrength": _float(config.get("hiresFixStrength"), 0.7),
                "loras": loras,
                "upscaler": str(config.get("upscaler") or ""),
                "faceRestoration": str(config.get("faceRestoration") or ""),
                "refinerModel": str(config.get("refinerModel") or ""),
                "refinerStart": _float(config.get("refinerStart"), 0.7),
                "maskBlur": _float(config.get("maskBlur")),
                "maskBlurOutset": _int(config.get("maskBlurOutset")),
                "sharpness": _float(config.get("sharpness")),
                "shift": _float(config.get("shift"), 1.0),
                "tiledDecoding": bool(config.get("tiledDecoding")),
                "tiledDiffusion": bool(config.get("tiledDiffusion")),
                "preserveOriginalAfterInpaint": bool(config.get("preserveOriginalAfterInpaint", True)),
                "colorCalibration": _name(
                    COLOR_CALIBRATION_NAMES, _int(config.get("colorCalibration")), "none"
                ),
                "batchSize": _int(config.get("batchSize"), 1),
                "numFrames": _int(config.get("numFrames"), 1) or 1,
                "generationTime": 0.0,
                # Keep the app's own configuration so bridge_settings() can pass
                # every key through, not just the handful summarised above.
                "configuration": config,
            }
        )
    return entries


def textual_inversions() -> list[dict[str, str]]:
    """Textual inversion embeddings Draw Things knows about (keyword -> file)."""
    path = _first_existing(TEXTUAL_INVERSION_PATHS)
    if path is None:
        return []
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    items = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and item.get("file"):
            items.append(
                {
                    "name": str(item.get("name") or ""),
                    "file": str(item["file"]),
                    "keyword": str(item.get("keyword") or ""),
                }
            )
    return items


def missing_textual_inversions() -> list[str]:
    return [item["file"] for item in textual_inversions() if not (DOCS_DIR / "Models" / item["file"]).is_file()]


def all_presets(limit: int = 60) -> list[dict[str, Any]]:
    """Named app presets first, then recent history entries (newest first)."""
    history = [entry for entry in recent_configs(limit) if (entry.get("prompt") or "").strip()]
    return custom_configs() + history


def clean_prompt(text: str) -> str:
    """Drop inline <lora:...> tags - the LoRA list already carries those."""
    return _LORA_TAG.sub("", text or "").strip()


def bridge_settings(entry: dict[str, Any]) -> dict[str, Any]:
    """The whole configuration a Draw Things preset stands for.

    Keys keep the app's own spelling (pixels for tile sizes, enum names instead
    of numbers) so a preset can be written into config.json unchanged;
    bridge.build_configuration knows the FlatBuffer mapping. A stored seed is
    dropped on purpose: it would freeze every generation.
    """
    source = entry.get("configuration")
    if not isinstance(source, dict):
        source = entry
    return {
        key: value
        for key, value in source.items()
        if key not in META_KEYS and key != "seed"
    }


def preset_notes(entry: dict[str, Any]) -> list[str]:
    """What the client must know after loading this preset."""
    settings = bridge_settings(entry)
    controls = [
        control
        for control in settings.get("controls") or []
        if isinstance(control, dict) and control.get("file")
    ]
    if not controls:
        return []
    return [f"ControlNet x{len(controls)}: requests need a control image too"]


if __name__ == "__main__":
    if "--summary" in sys.argv:
        named = custom_configs()
        missing = missing_textual_inversions()
        print(
            json.dumps(
                {
                    "custom_presets": len(named),
                    "custom_names": [entry["name"] for entry in named],
                    "custom_source": str(custom_source() or ""),
                    "textual_inversions": len(textual_inversions()),
                    "textual_inversions_missing": missing,
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(0)

    limit = 20
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    if "--named" in sys.argv:
        entries = custom_configs()
        if "--json" in sys.argv:
            print(json.dumps(entries, indent=2))
            raise SystemExit(0)
        print(f"{len(entries)} named presets from {custom_source()}")
        for entry in entries:
            print(
                f"  {entry['name']:<20} {entry['width']}x{entry['height']} {entry['steps']}st "
                f"{entry['sampler']:<14} lora{len(entry['loras'])}  {entry['model'][:34]}"
            )
        raise SystemExit(0)

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
