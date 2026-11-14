# /// script
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "grpclib",
#     "flatbuffers",
#     "protobuf",
#     "numpy",
#     "pillow",
#     "python-multipart",
#     "websockets",
# ]
# ///
"""OpenAI-compatible image generation bridge for the Draw Things gRPC server.

Translation layers:
    OpenAI REST (JSON)  ->  gRPC ImageGenerationService.GenerateImage
                            (protobuf request carrying a FlatBuffer
                             GenerationConfiguration)

Generated bindings live in ./gen (see README / Makefile in this repo).

Environment variables:
    DT_ADDR          gRPC host                (default 127.0.0.1)
    DT_PORT          gRPC port                (default 7859)
    DT_TLS           1 = TLS with ALPN h2     (default 0, server --no-tls)
    BRIDGE_HOST      HTTP bind address        (default 0.0.0.0)
    BRIDGE_PORT      HTTP port                (default 9955)
    BRIDGE_API_KEY   if set, requires "Authorization: Bearer <key>"
    DT_TIMEOUT       generation timeout (s)   (default 1800)
    DT_CONFIG        path to a JSON overlay of the generation settings
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import ssl
import struct
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import flatbuffers
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from grpclib.client import Channel
from grpclib.const import Cardinality
from PIL import Image
from pydantic import BaseModel
from starlette.datastructures import UploadFile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gen"))

import Control as ControlFB
import GenerationConfiguration as GC
import imageService_pb2 as _pb
import LoRA as LoRAFB

# The protobuf module is built at runtime (descriptor pool), so static checkers
# cannot see its message classes.
pb: Any = cast(Any, _pb)

DT_ADDR = os.environ.get("DT_ADDR", "127.0.0.1")
DT_PORT = int(os.environ.get("DT_PORT", "7859"))
DT_TLS = os.environ.get("DT_TLS", "0").lower() in ("1", "true", "yes")
BRIDGE_HOST = os.environ.get("BRIDGE_HOST", "0.0.0.0")
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "9955"))
API_KEY = os.environ.get("BRIDGE_API_KEY", "")
DT_TIMEOUT = float(os.environ.get("DT_TIMEOUT", "1800"))
CONFIG_PATH = Path(os.environ.get("DT_CONFIG", str(HERE / "config.json")))

# Serialized tensor layout produced by NNC's Tensor.data(using:):
#   [uint32 identifier][ccv_nnc_tensor_param_t 64B][raw data]
# identifier 0 means uncompressed, which the server accepts for request inputs
# (the app itself only bundles fpzip for decompressing responses).
TENSOR_HEADER = 68
CCV_TENSOR_CPU_MEMORY = 0x1
CCV_TENSOR_FORMAT_NHWC = 0x02
CCV_16F = 0x20000
IMAGE_DIR = HERE / "output"
IMAGE_DIR.mkdir(exist_ok=True)
# Source images uploaded by ComfyUI clients (LoadImage) live apart from outputs.
INPUT_DIR = HERE / "input"
INPUT_DIR.mkdir(exist_ok=True)
UPLOAD_MAX_BYTES = 32 * 1024 * 1024
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
# A MacBook renders 1344 px and friends comfortably; beyond this unified memory
# pressure makes Draw Things swap or drop the request.
MAX_IMAGE_SIZE = int(os.environ.get("DT_MAX_SIZE", "2048"))

# Draw Things enums (see schema/config.fbs).
SAMPLERS = {
    "DPMPP2MKarras": 0,
    "EulerA": 1,
    "DDIM": 2,
    "PLMS": 3,
    "DPMPPSDEKarras": 4,
    "UniPC": 5,
    "LCM": 6,
    "EulerASubstep": 7,
    "DPMPPSDESubstep": 8,
    "TCD": 9,
    "EulerATrailing": 10,
    "DPMPPSDETrailing": 11,
    "DPMPP2MAYS": 12,
    "EulerAAYS": 13,
    "DPMPPSDEAYS": 14,
    "DPMPP2MTrailing": 15,
    "DDIMTrailing": 16,
    "UniPCTrailing": 17,
    "UniPCAYS": 18,
    "TCDTrailing": 19,
}
SEED_MODES = {"Legacy": 0, "TorchCpuCompatible": 1, "ScaleAlike": 2, "NvidiaGpuCompatible": 3}
LORA_MODES = {"all": 0, "base": 1, "refiner": 2}
COLOR_CALIBRATIONS = {"none": 0, "disabled": 0, "lab": 1}
COMPRESSION_ARTIFACTS = {"disabled": 0, "none": 0, "h264": 1, "h265": 2, "jpeg": 3}
CONTROL_MODES = {"balanced": 0, "prompt": 1, "control": 2}
CONTROL_INPUT_TYPES = {
    "": 0,
    "unspecified": 0,
    "custom": 1,
    "depth": 2,
    "canny": 3,
    "scribble": 4,
    "pose": 5,
    "normalbae": 6,
    "color": 7,
    "lineart": 8,
    "softedge": 9,
    "seg": 10,
    "inpaint": 11,
    "ip2p": 12,
    "shuffle": 13,
    "mlsd": 14,
    "tile": 15,
    "blur": 16,
    "lowquality": 17,
    "gray": 18,
}
# Hint types the server accepts as ControlHintType names. "scribble" wants a
# 1-byte mask tensor, which this bridge does not build yet.
HINT_TYPES = tuple(
    name for name in CONTROL_INPUT_TYPES if name not in ("", "unspecified", "scribble")
)

# Preset keys the bridge cannot honour yet. They are surfaced by /healthz and the
# TUI instead of being dropped silently.
UNSUPPORTED_SETTINGS: dict[str, str] = {}

# Built-in fallback defaults before a Draw Things preset is loaded into config.json.
# Key names and values follow the app's own preset format, so a preset can be
# dropped into config.json unchanged.
DEFAULT_SETTINGS: dict[str, Any] = {
    # Core sampling
    "model": "",
    "steps": 20,
    "guidanceScale": 7.0,
    "strength": 1.0,
    "sampler": "DPMPP2MKarras",
    "seedMode": "ScaleAlike",
    "seed": -1,
    "clipSkip": 1,
    "shift": 1.0,
    "sharpness": 0.0,
    # Size / batching
    "width": 512,
    "height": 512,
    "batchCount": 1,
    "batchSize": 1,
    "numFrames": 1,
    # Hires fix / upscaling
    "hiresFix": False,
    "hiresFixStrength": 0.7,
    "hiresFixWidth": 0,
    "hiresFixHeight": 0,
    "upscaler": "",
    "upscalerScaleFactor": 0,
    # Layers applied on top of the checkpoint
    "loras": [],
    "controls": [],
    "refinerModel": "",
    "refinerStart": 0.7,
    "faceRestoration": "",
    # Masking, tiling, colour
    "maskBlur": 2.5,
    "maskBlurOutset": 0,
    "tiledDecoding": False,
    "tiledDiffusion": False,
    "decodingTileWidth": 640,
    "decodingTileHeight": 640,
    "decodingTileOverlap": 128,
    "diffusionTileWidth": 1024,
    "diffusionTileHeight": 1024,
    "diffusionTileOverlap": 128,
    "preserveOriginalAfterInpaint": True,
    "colorCalibration": "none",
    # Guidance and prompt handling
    "clipWeight": 1.0,
    "zeroNegativePrompt": False,
    "aestheticScore": 6.0,
    "negativeAestheticScore": 2.5,
    "imageGuidanceScale": 1.5,
    "imagePriorSteps": 5,
    "negativePromptForImagePrior": True,
    "guidanceEmbed": 3.5,
    "speedUpWithGuidanceEmbed": True,
    "stochasticSamplingGamma": 0.3,
    "resolutionDependentShift": True,
    # Source-image geometry (img2img / inpaint; ignored by txt2img requests)
    "originalImageWidth": 0,
    "originalImageHeight": 0,
    "targetImageWidth": 0,
    "targetImageHeight": 0,
    "cropLeft": 0,
    "cropTop": 0,
    "negativeOriginalImageWidth": 0,
    "negativeOriginalImageHeight": 0,
    # Multi-stage samplers (SD3.5 / HiDream)
    "stage2Steps": 10,
    "stage2Guidance": 1.0,
    "stage2Shift": 1.0,
    # Video and audio knobs (stored so presets round-trip; the bridge renders stills)
    "fps": 5,
    "motionScale": 127,
    "guidingFrameNoise": 0.02,
    "startFrameGuidance": 1.0,
    "shiftForAudio": 3.0,
    "usesSolAttention": False,
    "solAttentionStart": 2,
    "solAttentionTau": 0.5,
    "causalInference": 0,
    "causalInferencePad": 0,
    "cfgZeroStar": False,
    "cfgZeroInitSteps": 0,
    # Text encoders (SDXL / FLUX)
    "t5TextEncoder": True,
    "separateT5": False,
    "t5Text": "",
    "separateClipL": False,
    "clipLText": "",
    "separateOpenClipG": False,
    "openClipGText": "",
    # Caching, compression, prompt expansion
    "teaCache": False,
    "teaCacheStart": 5,
    "teaCacheEnd": -1,
    "teaCacheThreshold": 0.06,
    "teaCacheMaxSkipSteps": 3,
    "compressionArtifacts": "disabled",
    "compressionArtifactsQuality": 43.1,
    "expandPromptToJson": False,
}


def load_settings() -> dict[str, Any]:
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    overlay = os.environ.get("DT_CONFIG", str(HERE / "config.json"))
    path = Path(overlay)
    if path.is_file():
        settings.update(json.loads(path.read_text()))
    return settings


SETTINGS = load_settings()


def _enum(value: Any, table: dict[str, int], default: int = 0) -> int:
    if isinstance(value, int):
        return value
    name = str(value)
    if name in table:
        return table[name]
    # Presets spell enums in different cases ("all" vs "All", "disabled" vs "Disabled").
    folded = name.lower()
    for key, index in table.items():
        if key.lower() == folded:
            return index
    return default


# Field tables. Draw Things preset keys map onto FlatBuffer fields that are not
# always the snake_case spelling, and a few of them use different units.
_SCALAR_FIELDS: tuple[tuple[str, str, Any], ...] = (
    ("steps", "Steps", int),
    ("guidanceScale", "GuidanceScale", float),
    ("strength", "Strength", float),
    ("clipSkip", "ClipSkip", int),
    ("shift", "Shift", float),
    ("sharpness", "Sharpness", float),
    ("maskBlur", "MaskBlur", float),
    ("maskBlurOutset", "MaskBlurOutset", int),
    ("hiresFix", "HiresFix", bool),
    ("hiresFixStrength", "HiresFixStrength", float),
    ("refinerStart", "RefinerStart", float),
    ("numFrames", "NumFrames", int),
    ("preserveOriginalAfterInpaint", "PreserveOriginalAfterInpaint", bool),
    ("tiledDecoding", "TiledDecoding", bool),
    ("tiledDiffusion", "TiledDiffusion", bool),
    ("upscalerScaleFactor", "UpscalerScaleFactor", int),
    ("clipWeight", "ClipWeight", float),
    ("zeroNegativePrompt", "ZeroNegativePrompt", bool),
    ("aestheticScore", "AestheticScore", float),
    ("negativeAestheticScore", "NegativeAestheticScore", float),
    ("imageGuidanceScale", "ImageGuidanceScale", float),
    ("imagePriorSteps", "ImagePriorSteps", int),
    ("negativePromptForImagePrior", "NegativePromptForImagePrior", bool),
    ("guidanceEmbed", "GuidanceEmbed", float),
    ("speedUpWithGuidanceEmbed", "SpeedUpWithGuidanceEmbed", bool),
    ("stochasticSamplingGamma", "StochasticSamplingGamma", float),
    ("resolutionDependentShift", "ResolutionDependentShift", bool),
    ("originalImageWidth", "OriginalImageWidth", int),
    ("originalImageHeight", "OriginalImageHeight", int),
    ("targetImageWidth", "TargetImageWidth", int),
    ("targetImageHeight", "TargetImageHeight", int),
    ("cropLeft", "CropLeft", int),
    ("cropTop", "CropTop", int),
    ("negativeOriginalImageWidth", "NegativeOriginalImageWidth", int),
    ("negativeOriginalImageHeight", "NegativeOriginalImageHeight", int),
    ("stage2Steps", "Stage2Steps", int),
    ("stage2Guidance", "Stage2Cfg", float),
    ("stage2Shift", "Stage2Shift", float),
    ("fps", "FpsId", int),
    ("motionScale", "MotionBucketId", int),
    ("guidingFrameNoise", "CondAug", float),
    ("startFrameGuidance", "StartFrameCfg", float),
    ("shiftForAudio", "ShiftForAudio", float),
    ("usesSolAttention", "UsesSolAttention", bool),
    ("solAttentionStart", "SolAttentionStart", int),
    ("solAttentionTau", "SolAttentionTau", float),
    ("causalInference", "CausalInference", int),
    ("causalInferencePad", "CausalInferencePad", int),
    ("cfgZeroStar", "CfgZeroStar", bool),
    ("cfgZeroInitSteps", "CfgZeroInitSteps", int),
    ("t5TextEncoder", "T5TextEncoder", bool),
    ("separateT5", "SeparateT5", bool),
    ("separateClipL", "SeparateClipL", bool),
    ("separateOpenClipG", "SeparateOpenClipG", bool),
    ("teaCache", "TeaCache", bool),
    ("teaCacheStart", "TeaCacheStart", int),
    ("teaCacheEnd", "TeaCacheEnd", int),
    ("teaCacheThreshold", "TeaCacheThreshold", float),
    ("teaCacheMaxSkipSteps", "TeaCacheMaxSkipSteps", int),
    ("compressionArtifactsQuality", "CompressionArtifactsQuality", float),
    ("expandPromptToJson", "ExpandPromptToJson", bool),
)

# Draw Things stores tile sizes in pixels; the FlatBuffer counts 64 px blocks.
_BLOCK_FIELDS: tuple[tuple[str, str], ...] = (
    ("decodingTileWidth", "DecodingTileWidth"),
    ("decodingTileHeight", "DecodingTileHeight"),
    ("decodingTileOverlap", "DecodingTileOverlap"),
    ("diffusionTileWidth", "DiffusionTileWidth"),
    ("diffusionTileHeight", "DiffusionTileHeight"),
    ("diffusionTileOverlap", "DiffusionTileOverlap"),
)

_ENUM_FIELDS: tuple[tuple[str, str, dict[str, int]], ...] = (
    ("sampler", "Sampler", SAMPLERS),
    ("seedMode", "SeedMode", SEED_MODES),
    ("colorCalibration", "ColorCalibration", COLOR_CALIBRATIONS),
    ("compressionArtifacts", "CompressionArtifacts", COMPRESSION_ARTIFACTS),
)

# Optional strings: absent when empty, never "" (see the note in build_configuration).
_TEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("upscaler", "Upscaler"),
    ("faceRestoration", "FaceRestoration"),
    ("refinerModel", "RefinerModel"),
    ("clipLText", "ClipLText"),
    ("openClipGText", "OpenClipGText"),
    ("t5Text", "T5Text"),
)


def _blocks(value: Any) -> int:
    """Pixels -> 64 px blocks (Draw Things presets store pixels)."""
    try:
        return max(0, int(value) // 64)
    except (TypeError, ValueError):
        return 0


def build_configuration(
    *,
    settings: dict[str, Any],
    width: int,
    height: int,
    seed: int,
    batch_count: int,
    batch_size: int = 1,
    model: str | None = None,
) -> bytes:
    """Serialize a GenerationConfiguration FlatBuffer.

    Every preset key the bridge understands is written, so a configuration
    saved in Draw Things replays here. Widths and heights are in 64 px units.
    Preset keys that are not mapped yet (`controls`, `name`) stay absent; the
    server then uses its own defaults.
    """
    b = flatbuffers.Builder(4096)
    model_off = b.CreateString(model or settings["model"])
    # Optional strings must stay absent (nil) when empty: the server treats an
    # empty upscaler/refiner/face-restoration as "use the default" and would
    # e.g. apply a 4x upscaler to every image.
    text_offsets = {
        suffix: (b.CreateString(str(settings[key])) if settings.get(key) else 0)
        for key, suffix in _TEXT_FIELDS
    }

    lora_offsets = []
    for lora in settings.get("loras") or []:
        file_off = b.CreateString(lora["file"])
        LoRAFB.LoRAStart(b)
        LoRAFB.LoRAAddFile(b, file_off)
        LoRAFB.LoRAAddWeight(b, float(lora.get("weight", 1.0)))
        LoRAFB.LoRAAddMode(b, _enum(lora.get("mode", "all"), LORA_MODES))
        lora_offsets.append(LoRAFB.LoRAEnd(b))

    if lora_offsets:
        b.StartVector(4, len(lora_offsets), 4)
        for off in reversed(lora_offsets):
            b.PrependUOffsetTRelative(off)
        loras_vec = b.EndVector()
    else:
        loras_vec = 0

    control_offsets = []
    for control in settings.get("controls") or []:
        if not isinstance(control, dict) or not control.get("file"):
            continue
        blocks = [str(block) for block in control.get("targetBlocks") or []]
        blocks_vec = 0
        if blocks:
            block_offsets = [b.CreateString(block) for block in blocks]
            b.StartVector(4, len(block_offsets), 4)
            for off in reversed(block_offsets):
                b.PrependUOffsetTRelative(off)
            blocks_vec = b.EndVector()
        control_file = b.CreateString(str(control["file"]))
        ControlFB.ControlStart(b)
        ControlFB.ControlAddFile(b, control_file)
        ControlFB.ControlAddWeight(b, float(control.get("weight", 1.0)))
        ControlFB.ControlAddGuidanceStart(b, float(control.get("guidanceStart", 0.0)))
        ControlFB.ControlAddGuidanceEnd(b, float(control.get("guidanceEnd", 1.0)))
        ControlFB.ControlAddNoPrompt(b, bool(control.get("noPrompt")))
        ControlFB.ControlAddGlobalAveragePooling(b, bool(control.get("globalAveragePooling", True)))
        ControlFB.ControlAddDownSamplingRate(b, float(control.get("downSamplingRate", 1.0)))
        ControlFB.ControlAddControlMode(
            b, _enum(control.get("controlImportance", "balanced"), CONTROL_MODES)
        )
        ControlFB.ControlAddInputOverride(
            b, _enum(control.get("inputOverride", ""), CONTROL_INPUT_TYPES)
        )
        if blocks_vec:
            ControlFB.ControlAddTargetBlocks(b, blocks_vec)
        control_offsets.append(ControlFB.ControlEnd(b))

    if control_offsets:
        b.StartVector(4, len(control_offsets), 4)
        for off in reversed(control_offsets):
            b.PrependUOffsetTRelative(off)
        controls_vec = b.EndVector()
    else:
        controls_vec = 0

    GC.GenerationConfigurationStart(b)
    GC.GenerationConfigurationAddId(b, 0)
    GC.GenerationConfigurationAddStartWidth(b, max(1, width // 64))
    GC.GenerationConfigurationAddStartHeight(b, max(1, height // 64))
    GC.GenerationConfigurationAddSeed(b, seed & 0xFFFFFFFF)
    GC.GenerationConfigurationAddModel(b, model_off)
    GC.GenerationConfigurationAddBatchCount(b, batch_count)
    GC.GenerationConfigurationAddBatchSize(b, batch_size)
    GC.GenerationConfigurationAddHiresFixStartWidth(b, _blocks(settings.get("hiresFixWidth")))
    GC.GenerationConfigurationAddHiresFixStartHeight(b, _blocks(settings.get("hiresFixHeight")))

    for suffix, offset in text_offsets.items():
        if offset:
            getattr(GC, f"GenerationConfigurationAdd{suffix}")(b, offset)
    for key, suffix, convert in _SCALAR_FIELDS:
        value = settings.get(key)
        if value is not None:
            getattr(GC, f"GenerationConfigurationAdd{suffix}")(b, convert(value))
    for key, suffix in _BLOCK_FIELDS:
        value = settings.get(key)
        if value is not None:
            getattr(GC, f"GenerationConfigurationAdd{suffix}")(b, _blocks(value))
    for key, suffix, table in _ENUM_FIELDS:
        getattr(GC, f"GenerationConfigurationAdd{suffix}")(b, _enum(settings.get(key), table))
    # Draw Things stores only the frame count for causal inference; 0 means off.
    GC.GenerationConfigurationAddCausalInferenceEnabled(
        b, bool(int(settings.get("causalInference") or 0))
    )
    if loras_vec:
        GC.GenerationConfigurationAddLoras(b, loras_vec)
    if controls_vec:
        GC.GenerationConfigurationAddControls(b, controls_vec)

    root = GC.GenerationConfigurationEnd(b)
    b.Finish(root)
    return bytes(b.Output())


def decode_tensor_png(data: bytes) -> bytes:
    """Convert an NNC fp16 tensor (HWC, [-1, 1]) into PNG bytes.

    Mirrors DrawThingsCLI.writePNG(): pixel = clamp((value + 1) * 127.5).
    """
    if len(data) <= TENSOR_HEADER:
        raise ValueError("tensor payload too small")
    identifier = struct.unpack_from("<I", data, 0)[0]
    if identifier != 0:
        raise ValueError(f"tensor is encoded with codec {identifier}; disable response compression")

    payload = len(data) - TENSOR_HEADER
    # The ccv tensor param block holds dims as int32; find the 4 that match the
    # payload size instead of hard-coding the struct layout.
    dims = None
    for offset in range(4, 4 + 32, 4):
        candidate = struct.unpack_from("<4i", data, offset)
        if all(d > 0 for d in candidate) and candidate[0] * candidate[1] * candidate[2] * candidate[3] * 2 == payload:
            dims = candidate
            break
    if dims is None:
        raise ValueError("unrecognised tensor header")

    batch, height, width, channels = dims
    if channels < 3:
        raise ValueError(f"unsupported channel count {channels}")

    values = np.frombuffer(data, dtype="<f2", count=batch * height * width * channels, offset=TENSOR_HEADER)
    image = values.reshape(batch, height, width, channels)[0].astype(np.float32)
    rgb = np.clip((image[..., :3] + 1.0) * 127.5, 0, 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def encode_image_tensor(image: bytes, width: int, height: int) -> bytes:
    """Convert PNG/JPEG bytes into a Draw Things NNC input tensor frame.

    The server never accepts PNG/JPEG for `image` (nor for `mask` or control
    hints): it wants [uint32 identifier=0][ccv_nnc_tensor_param_t 64B][fp16
    NHWC] with pixels normalised to [-1, 1]. The source is resized to the
    generation size first, which is what the app does for img2img.
    """
    with Image.open(io.BytesIO(image)) as source:
        rgb = source.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    pixels = np.asarray(rgb, dtype=np.float32) * (2.0 / 255.0) - 1.0
    dims = [1, height, width, 3, 0, 0, 0, 0, 0, 0, 0, 0]
    header = struct.pack("<I", 0) + struct.pack(
        "<16i", CCV_TENSOR_CPU_MEMORY, CCV_TENSOR_FORMAT_NHWC, CCV_16F, 0, *dims
    )
    return header + pixels.astype("<f2").tobytes()


class Backend:
    """Thin async client for the Draw Things gRPC server."""

    def __init__(self) -> None:
        self.channel: Channel | None = None
        self.gate = asyncio.Semaphore(1)

    def _make_channel(self) -> Channel:
        if not DT_TLS:
            return Channel(DT_ADDR, DT_PORT)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.set_alpn_protocols(["h2"])  # the server closes connections without ALPN
        return Channel(DT_ADDR, DT_PORT, ssl=ctx)

    async def start(self) -> None:
        self.channel = self._make_channel()

    async def close(self) -> None:
        if self.channel is not None:
            self.channel.close()
            self.channel = None

    async def echo(self, name: str = "bridge") -> Any:
        assert self.channel is not None
        async with self.channel.request(
            "/ImageGenerationService/Echo",
            Cardinality.UNARY_UNARY,
            pb.EchoRequest,
            pb.EchoReply,
            timeout=10,
        ) as stream:
            await stream.send_message(pb.EchoRequest(name=name), end=True)
            return await stream.recv_message()

    async def files_exist(self, files: list[str]) -> dict[str, bool]:
        assert self.channel is not None
        async with self.channel.request(
            "/ImageGenerationService/FilesExist",
            Cardinality.UNARY_UNARY,
            pb.FileListRequest,
            pb.FileExistenceResponse,
            timeout=20,
        ) as stream:
            await stream.send_message(pb.FileListRequest(files=files), end=True)
            reply = await stream.recv_message()
        if reply is None:
            raise RuntimeError("Draw Things returned an empty FilesExist response")
        return dict(zip(reply.files, reply.existences, strict=False))

    async def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        configuration: bytes,
        image: bytes | None = None,
        hints: list[tuple[str, bytes]] | None = None,
    ) -> list[bytes]:
        assert self.channel is not None
        images: list[bytes] = []
        buffer = bytearray()
        fields: dict[str, Any] = {
            "prompt": prompt,
            "negativePrompt": negative_prompt,
            "configuration": configuration,
            "user": "openai-bridge",
            "chunked": True,
        }
        # Large payloads are addressed by sha256 and shipped in `contents`.
        contents: list[bytes] = []
        if image is not None:
            fields["image"] = hashlib.sha256(image).digest()
            fields["scaleFactor"] = 1
            contents.append(image)
        if hints:
            hint_protos = []
            for hint_type, frame in hints:
                contents.append(frame)
                hint_protos.append(
                    pb.HintProto(
                        hintType=hint_type,
                        tensors=[pb.TensorAndWeight(tensor=hashlib.sha256(frame).digest(), weight=1.0)],
                    )
                )
            fields["hints"] = hint_protos
        if contents:
            fields["contents"] = contents
        async with self.channel.request(
            "/ImageGenerationService/GenerateImage",
            Cardinality.UNARY_STREAM,
            pb.ImageGenerationRequest,
            pb.ImageGenerationResponse,
            timeout=DT_TIMEOUT,
        ) as stream:
            await stream.send_message(pb.ImageGenerationRequest(**fields), end=True)
            async for response in stream:
                for chunk in response.generatedImages:
                    buffer.extend(chunk)
                if response.chunkState == pb.LAST_CHUNK and buffer:
                    images.append(bytes(buffer))
                    buffer.clear()
        if buffer:
            images.append(bytes(buffer))
        return images


backend = Backend()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await backend.start()
    try:
        yield
    finally:
        await backend.close()


app = FastAPI(title="Draw Things OpenAI Bridge", version="1.0.0", lifespan=lifespan)


class _QuietPolling(logging.Filter):
    """Keep /healthz polling (TUI, page refresh) out of the access log."""

    NOISY = ("/healthz", "/v1/models")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(path in message for path in self.NOISY)


logging.getLogger("uvicorn.access").addFilter(_QuietPolling())


def check_auth(request: Request) -> None:
    if not API_KEY:
        return
    header = request.headers.get("authorization", "")
    if header != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


def _hint_type(value: Any) -> str:
    """Validate a ControlNet hint type ("custom" when the client omits it)."""
    name = str(value or "custom").strip().lower()
    if name == "scribble":
        raise HTTPException(
            status_code=501, detail="scribble hints need a 1-byte mask tensor - not wired yet"
        )
    if name not in HINT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown control type {name!r}; use one of {', '.join(HINT_TYPES)}",
        )
    return name


async def render_pngs(
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    count: int,
    settings: dict[str, Any],
    model: str | None = None,
    source_image: bytes | None = None,
    strength: float | None = None,
    hint_images: list[tuple[str, bytes]] | None = None,
) -> list[bytes]:
    """Run one generation and return PNG bytes (one per image).

    `source_image` (PNG/JPEG bytes) turns this into an img2img request: the
    canvas travels as an NNC tensor and the config describes it as the original
    image, exactly as the app does.
    """
    chosen_model = model or settings.get("model") or ""
    if not chosen_model:
        raise HTTPException(
            status_code=400,
            detail="No model configured. Load a preset in './dt tui' (press l) or set 'model' in config.json.",
        )
    if source_image is not None:
        settings = {
            **settings,
            "originalImageWidth": width,
            "originalImageHeight": height,
            "targetImageWidth": width,
            "targetImageHeight": height,
            "cropLeft": 0,
            "cropTop": 0,
        }
    if strength is not None:
        settings = {**settings, "strength": float(strength)}
    tensor = encode_image_tensor(source_image, width, height) if source_image is not None else None
    # ControlNet input images travel as hints, keyed by the hint type the
    # ControlNet expects ("pose", "depth", "canny", ...).
    hints = (
        [(hint_type, encode_image_tensor(data, width, height)) for hint_type, data in hint_images]
        if hint_images
        else None
    )
    configuration = build_configuration(
        settings=settings,
        width=width,
        height=height,
        seed=seed,
        # Draw Things renders a batch in one pass; the gRPC server ignores
        # batch_count (that is client-side looping), so map n to batch_size.
        batch_count=1,
        batch_size=count,
        model=chosen_model,
    )

    async with backend.gate:
        try:
            images = await backend.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                configuration=configuration,
                image=tensor,
                hints=hints,
            )
        except Exception as exc:  # translated into an HTTP error
            raise HTTPException(status_code=502, detail=f"Draw Things error: {exc}") from exc

    if not images:
        raise HTTPException(status_code=502, detail="Draw Things returned no image")

    pngs: list[bytes] = []
    for image in images:
        try:
            pngs.append(decode_tensor_png(image))
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=f"Cannot decode Draw Things tensor: {exc}") from exc
    return pngs


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str | None = None
    n: int | None = 1
    size: str | None = None
    width: int | None = None
    height: int | None = None
    response_format: str | None = "b64_json"
    negative_prompt: str | None = None
    negativePrompt: str | None = None
    steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None
    sampler: str | int | None = None


def parse_size(payload: ImageGenerationRequest) -> tuple[int, int]:
    width, height = int(SETTINGS["width"]), int(SETTINGS["height"])
    if payload.width and payload.height:
        width, height = int(payload.width), int(payload.height)
    elif payload.size:
        try:
            w_str, h_str = payload.size.lower().split("x")
            width, height = int(w_str), int(h_str)
        except (ValueError, AttributeError):
            pass
    # start_width/start_height are in 64 px units, so snap to the grid.
    width = max(64, min((width // 64) * 64, MAX_IMAGE_SIZE))
    height = max(64, min((height // 64) * 64, MAX_IMAGE_SIZE))
    return width, height


def pick_model(requested: str | None) -> str:
    if requested and requested.strip():
        name = requested.strip()
        # OpenAI clients often send placeholder ids like "dall-e-3" or "local".
        if name.endswith((".ckpt", ".safetensors")):
            return name
        if name in {m["id"] for m in _model_list()}:
            return name
    return SETTINGS["model"]


def _model_list() -> list[dict[str, Any]]:
    return [
        {
            "id": SETTINGS["model"],
            "object": "model",
            "created": 0,
            "owned_by": "draw-things",
        }
    ]


@app.get("/admin/config")
async def admin_get_config():
    return SETTINGS


@app.post("/admin/config")
async def admin_set_config(update: dict[str, Any]):
    unknown = [key for key in update if key not in DEFAULT_SETTINGS]
    SETTINGS.update({key: value for key, value in update.items() if key in DEFAULT_SETTINGS})
    CONFIG_PATH.write_text(json.dumps(SETTINGS, indent=2) + "\n")
    return {"ok": True, "applied": sorted(set(update) - set(unknown)), "ignored": unknown}


@app.get("/v1/models")
async def list_models(request: Request):
    check_auth(request)
    return {"object": "list", "data": _model_list()}


@app.get("/healthz")
async def healthz():
    model = SETTINGS.get("model") or ""
    files = [f for f in ([model] + [lora["file"] for lora in SETTINGS.get("loras") or []]) if f]
    try:
        if files:
            existence = await backend.files_exist(files)
        else:
            await backend.echo()
            existence = {}
    except Exception as exc:  # surfaced to the caller
        raise HTTPException(status_code=503, detail=f"Draw Things unreachable: {exc}") from exc
    missing = [name for name, ok in existence.items() if not ok]
    if not model:
        missing.append("(no model configured)")
    return {
        "status": "ok" if not missing else "degraded",
        "backend": f"{DT_ADDR}:{DT_PORT}",
        "tls": DT_TLS,
        "model": model,
        "models_ready": not missing,
        "missing": missing,
        # ControlNet entries loaded from the preset; a request must also carry a
        # control image ("control_image" / a ControlNetApply LoadImage) to use them.
        "controls": [
            str(control["file"])
            for control in SETTINGS.get("controls") or []
            if isinstance(control, dict) and control.get("file")
        ],
        # Preset features that are loaded but not sent to the server yet.
        "unsupported": sorted(key for key in UNSUPPORTED_SETTINGS if SETTINGS.get(key)),
    }


def _images_payload(pngs: list[bytes], response_format: str, request: Request) -> dict[str, Any]:
    """OpenAI response envelope: b64_json payloads or URLs served by /images."""
    data: list[dict[str, str]] = []
    for png in pngs:
        if response_format == "url":
            name = f"{uuid.uuid4().hex}.png"
            (IMAGE_DIR / name).write_bytes(png)
            data.append({"url": str(request.base_url).rstrip("/") + f"/images/{name}"})
        else:
            data.append({"b64_json": base64.b64encode(png).decode("ascii")})
    return {"created": int(time.time()), "data": data}


@app.post("/v1/images/generations")
async def create_image(payload: ImageGenerationRequest, request: Request):
    check_auth(request)

    settings = dict(SETTINGS)
    if payload.steps is not None:
        settings["steps"] = payload.steps
    if payload.guidance_scale is not None:
        settings["guidanceScale"] = payload.guidance_scale
    if payload.sampler is not None:
        settings["sampler"] = payload.sampler

    width, height = parse_size(payload)
    seed = payload.seed if payload.seed is not None else int(settings.get("seed", -1))
    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
    count = max(1, min(int(payload.n or 1), 4))

    pngs = await render_pngs(
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt or payload.negativePrompt or "",
        width=width,
        height=height,
        seed=seed,
        count=count,
        settings=settings,
        model=pick_model(payload.model),
    )
    return _images_payload(pngs, payload.response_format or "b64_json", request)


@app.post("/v1/images/edits")
async def create_image_edit(request: Request):
    """OpenAI-compatible img2img: multipart `image` + `prompt`.

    `strength` (0-1) is a bridge extension and maps to the Draw Things strength;
    without it the configured value is used. Masks (inpainting) are not wired.
    """
    check_auth(request)
    form = await request.form()
    upload = form.get("image")
    if not isinstance(upload, UploadFile):
        raise HTTPException(status_code=400, detail="expected a multipart 'image' file field")
    source = await upload.read()
    if not source:
        raise HTTPException(status_code=400, detail="uploaded image is empty")
    if len(source) > UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413, detail=f"image is larger than {UPLOAD_MAX_BYTES // (1024 * 1024)} MB"
        )
    prompt = str(form.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="'prompt' is required")
    if form.get("mask"):
        raise HTTPException(status_code=501, detail="inpainting masks are not supported yet")

    settings = dict(SETTINGS)
    with Image.open(io.BytesIO(source)) as probe:
        source_width, source_height = probe.size
    width, height = source_width, source_height
    size = form.get("size")
    if isinstance(size, str) and "x" in size.lower():
        try:
            width = int(size.lower().split("x")[0])
            height = int(size.lower().split("x")[1])
        except (ValueError, IndexError):
            width, height = source_width, source_height
    width = max(64, min((width // 64) * 64, MAX_IMAGE_SIZE))
    height = max(64, min((height // 64) * 64, MAX_IMAGE_SIZE))

    strength = None
    raw_strength = form.get("strength")
    if isinstance(raw_strength, str) and raw_strength.strip():
        try:
            strength = max(0.0, min(float(raw_strength), 1.0))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="'strength' must be a number 0-1") from exc

    count = 1
    raw_n = form.get("n")
    if isinstance(raw_n, str) and raw_n.strip().isdigit():
        count = max(1, min(int(raw_n), 4))

    hint_images: list[tuple[str, bytes]] = []
    control = form.get("control_image")
    if isinstance(control, UploadFile):
        control_type = _hint_type(form.get("control_type"))
        control_bytes = await control.read()
        if control_bytes:
            hint_images.append((control_type, control_bytes))

    seed = int(settings.get("seed", -1))
    raw_seed = form.get("seed")
    if isinstance(raw_seed, str) and raw_seed.strip():
        try:
            seed = int(raw_seed)
        except ValueError:
            seed = -1
    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF

    pngs = await render_pngs(
        prompt=prompt,
        negative_prompt=str(form.get("negative_prompt") or ""),
        width=width,
        height=height,
        seed=seed,
        count=count,
        settings=settings,
        model=pick_model(str(form.get("model") or "")),
        source_image=source,
        strength=strength,
        hint_images=hint_images or None,
    )
    response_format = form.get("response_format")
    return _images_payload(pngs, response_format if isinstance(response_format, str) else "b64_json", request)


@app.get("/images/{name}")
async def get_image(name: str):
    path = IMAGE_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="image/png")


def _uploaded_name(filename: str | None, data: bytes) -> str:
    """Flat, predictable name for an uploaded source image."""
    base = os.path.basename((filename or "").replace("\\", "/")).strip()
    stem, suffix = os.path.splitext(base)
    if suffix.lower() not in IMAGE_SUFFIXES:
        suffix = ".png"
    return f"{stem or 'upload-' + uuid.uuid4().hex}{suffix}"


@app.post("/upload/image")
async def upload_image(request: Request):
    """ComfyUI's /upload/image: park a source image and return its name.

    ComfyUI clients (SillyTavern's img2img workflows, RisuAI) POST the picture
    here, then reference the returned name from a LoadImage node.
    """
    check_auth(request)
    form = await request.form()
    upload = form.get("image")
    if not isinstance(upload, UploadFile):
        raise HTTPException(status_code=400, detail="expected a multipart 'image' file field")
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="uploaded image is empty")
    if len(data) > UPLOAD_MAX_BYTES:
        raise HTTPException(
            status_code=413, detail=f"image is larger than {UPLOAD_MAX_BYTES // (1024 * 1024)} MB"
        )
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
    except Exception as exc:  # noqa: BLE001 - reported to the client
        raise HTTPException(status_code=400, detail=f"not a readable image: {exc}") from exc
    name = _uploaded_name(upload.filename, data)
    (INPUT_DIR / name).write_bytes(data)
    return {"name": name, "subfolder": "", "type": "input"}


def _uploaded_images(limit: int = 200) -> list[str]:
    """Source images a LoadImage node can pick from (newest first)."""
    try:
        entries = [path for path in INPUT_DIR.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    except OSError:
        return []
    entries.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [path.name for path in entries[:limit]]


# ---------------------------------------------------------------------------
# ComfyUI-compatible surface (subset)
#
# Clients such as SillyTavern and RisuAI talk to ComfyUI by POSTing a workflow
# graph and then polling /history + fetching /view. We interpret the prompt,
# negative, size, seed, batch, filename and - for img2img - a LoadImage node plus
# the sampler's denoise (which becomes the Draw Things strength); steps, cfg,
# sampler, checkpoint and LoRAs stay under our control via config.json.
# ---------------------------------------------------------------------------

SAMPLER_NODES = {"KSampler", "KSamplerAdvanced", "SamplerCustom"}
LATENT_NODES = {"EmptyLatentImage", "EmptySD3LatentImage", "EmptySDXLLatentImage"}
LOAD_NODES = {"LoadImage"}
CONTROL_NODES = {"ControlNetApply", "ControlNetApplyAdvanced", "ControlNetApplySD3"}
SAVE_NODES = {"SaveImage", "PreviewImage"}
TEXT_NODES = {"CLIPTextEncode", "CLIPTextEncodeSDXL", "BNK_CLIPTextEncodeAdvanced"}
LITERAL_INPUTS = ("value", "string", "text", "int", "float", "number", "seed")

COMFY_JOBS: dict[str, dict[str, Any]] = {}
COMFY_CLIENTS: list[Any] = []
COMFY_COUNTER = 0


class ComfyGraphError(Exception):
    def __init__(self, message: str, node_id: str | None = None) -> None:
        super().__init__(message)
        self.node_id = node_id


def _link(value: Any) -> tuple[str, int] | None:
    """A node link looks like ["12", 0]; widgets are plain values."""
    if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], (str, int)):
        return str(value[0]), int(value[1])
    return None


def _node(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    return graph.get(str(node_id)) or {}


def _literal(graph: dict[str, Any], value: Any, depth: int = 0) -> Any:
    """Resolve a widget value that may be linked to a primitive node."""
    if depth > 8:
        return None
    link = _link(value)
    if link is None:
        return value
    node = _node(graph, link[0])
    inputs = node.get("inputs") or {}
    for key in LITERAL_INPUTS:
        if key in inputs:
            resolved = _literal(graph, inputs[key], depth + 1)
            if resolved is not None:
                return resolved
    for candidate in inputs.values():
        resolved = _literal(graph, candidate, depth + 1)
        if resolved is not None and not isinstance(resolved, (dict, list)):
            return resolved
    return None


def _walk_up(graph: dict[str, Any], node_id: str, classes: set[str]) -> tuple[str, dict[str, Any]] | None:
    seen: set[str] = set()
    stack = [str(node_id)]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        node = _node(graph, current)
        if node.get("class_type") in classes:
            return current, node
        for value in (node.get("inputs") or {}).values():
            link = _link(value)
            if link is not None:
                stack.append(link[0])
    return None


def _images_under(graph: dict[str, Any], node_id: str) -> list[str]:
    """LoadImage file names reachable upstream from one node link."""
    names: list[str] = []
    seen: set[str] = set()
    stack = [str(node_id)]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        node = _node(graph, current)
        if node.get("class_type") in LOAD_NODES:
            name = _literal(graph, (node.get("inputs") or {}).get("image"))
            if isinstance(name, str) and name.strip() and name.strip() not in names:
                names.append(name.strip())
        for value in (node.get("inputs") or {}).values():
            link = _link(value)
            if link is not None:
                stack.append(link[0])
    return names


def _conditioning_text(graph: dict[str, Any], value: Any, key: str | None = None) -> str:
    link = _link(value)
    if link is None:
        return ""
    # ControlNetApply* nodes sit between the sampler and the text encoders; follow
    # their own positive/negative input instead of guessing among all inputs.
    node = _node(graph, link[0])
    if key is not None and node.get("class_type") in CONTROL_NODES:
        inner = (node.get("inputs") or {}).get(key)
        return _conditioning_text(graph, inner, key)
    found = _walk_up(graph, link[0], TEXT_NODES)
    if found is None:
        return ""
    _, text_node = found
    text = _literal(graph, (text_node.get("inputs") or {}).get("text"))
    return text if isinstance(text, str) else ""


def parse_comfy_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Extract prompt / negative / size / seed / batch / filename from a workflow."""
    if not isinstance(graph, dict) or not graph:
        raise ComfyGraphError("graph is empty")

    save_id, save_node = None, None
    for node_id, node in graph.items():
        if isinstance(node, dict) and node.get("class_type") in SAVE_NODES:
            save_id, save_node = str(node_id), node

    sampler = None
    sampler_id = "workflow"
    if save_node is not None:
        images_link = _link((save_node.get("inputs") or {}).get("images"))
        if images_link is not None:
            found = _walk_up(graph, images_link[0], SAMPLER_NODES)
            if found is not None:
                sampler_id, sampler = found
    if sampler is None:
        for node_id, node in graph.items():
            if isinstance(node, dict) and node.get("class_type") in SAMPLER_NODES:
                sampler_id, sampler = str(node_id), node
                break
    if sampler is None:
        raise ComfyGraphError("no KSampler node found in workflow")

    inputs = sampler.get("inputs") or {}
    prompt = _conditioning_text(graph, inputs.get("positive"), "positive")
    negative = _conditioning_text(graph, inputs.get("negative"), "negative")
    if not prompt:
        # Some custom nodes put the text straight on the sampler.
        fallback = _literal(graph, inputs.get("text") or inputs.get("prompt"))
        prompt = fallback if isinstance(fallback, str) else ""
    if not prompt:
        raise ComfyGraphError("no prompt text found (CLIPTextEncode -> positive)", sampler_id)

    width = height = None
    batch = 1
    latent_link = _link(inputs.get("latent_image"))
    if latent_link is not None:
        found = _walk_up(graph, latent_link[0], LATENT_NODES)
        if found is not None:
            _, latent = found
            latent_inputs = latent.get("inputs") or {}
            width = _literal(graph, latent_inputs.get("width"))
            height = _literal(graph, latent_inputs.get("height"))
            batch = _literal(graph, latent_inputs.get("batch_size")) or 1

    seed = _literal(graph, inputs.get("seed"))
    if seed is None:
        seed = _literal(graph, inputs.get("noise_seed"))

    prefix = None
    if save_node is not None:
        prefix = _literal(graph, (save_node.get("inputs") or {}).get("filename_prefix"))

    # img2img: a LoadImage node names a file uploaded through /upload/image, and
    # the sampler's denoise becomes the Draw Things strength. LoadImage nodes
    # wired into a ControlNetApply node are control hints, not the canvas.
    hint_names: list[str] = []
    for node in graph.values():
        if isinstance(node, dict) and node.get("class_type") in CONTROL_NODES:
            for value in (node.get("inputs") or {}).values():
                link = _link(value)
                if link is None:
                    continue
                for name in _images_under(graph, link[0]):
                    if name not in hint_names:
                        hint_names.append(name)

    canvas_names = _images_under(graph, latent_link[0]) if latent_link is not None else []
    source = canvas_names[0] if canvas_names else None
    if source is None:
        for node in graph.values():
            if isinstance(node, dict) and node.get("class_type") in LOAD_NODES:
                candidate = _literal(graph, (node.get("inputs") or {}).get("image"))
                if isinstance(candidate, str) and candidate.strip():
                    source = candidate.strip()
                    break
    hints = [name for name in hint_names if name != source]

    def _int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _float(value: Any) -> float | None:
        try:
            return max(0.0, min(float(value), 1.0))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    return {
        "prompt": prompt,
        "negative": negative,
        "width": _int(width, int(SETTINGS["width"])),
        "height": _int(height, int(SETTINGS["height"])),
        "size_from_latent": width is not None and height is not None,
        "batch": max(1, min(_int(batch, 1), 4)),
        "seed": _int(seed, -1),
        "save_node": save_id or "0",
        "prefix": str(prefix) if prefix else "DrawThings",
        "source": source,
        "hints": hints,
        "strength": _float(_literal(graph, inputs.get("denoise"))) if source else None,
    }


async def _comfy_broadcast(message: dict[str, Any]) -> None:
    for client in list(COMFY_CLIENTS):
        try:
            await client.send_json(message)
        except Exception:  # dropped clients are pruned
            if client in COMFY_CLIENTS:
                COMFY_CLIENTS.remove(client)


async def _run_comfy_job(prompt_id: str, spec: dict[str, Any]) -> None:
    global COMFY_COUNTER
    job = COMFY_JOBS[prompt_id]
    await _comfy_broadcast({"type": "executing", "data": {"node": spec["save_node"], "prompt_id": prompt_id}})
    try:
        seed = spec["seed"] if spec["seed"] >= 0 else int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
        width = max(64, min((spec["width"] // 64) * 64, MAX_IMAGE_SIZE))
        height = max(64, min((spec["height"] // 64) * 64, MAX_IMAGE_SIZE))
        source_image = None
        if spec.get("source"):
            source_path = INPUT_DIR / os.path.basename(str(spec["source"]))
            if not source_path.is_file():
                raise HTTPException(
                    status_code=400,
                    detail=f"source image {spec['source']!r} was not uploaded; POST it to /upload/image first",
                )
            source_image = source_path.read_bytes()
            if not spec.get("size_from_latent"):
                # img2img workflows that never size a latent keep the source size.
                with Image.open(io.BytesIO(source_image)) as probe:
                    source_width, source_height = probe.size
                width = max(64, min((source_width // 64) * 64, MAX_IMAGE_SIZE))
                height = max(64, min((source_height // 64) * 64, MAX_IMAGE_SIZE))
        hint_images: list[tuple[str, bytes]] = []
        for hint_name in spec.get("hints") or []:
            hint_path = INPUT_DIR / os.path.basename(str(hint_name))
            if hint_path.is_file():
                hint_images.append(("custom", hint_path.read_bytes()))
        pngs = await render_pngs(
            prompt=spec["prompt"],
            negative_prompt=spec["negative"],
            width=width,
            height=height,
            seed=seed,
            count=spec["batch"],
            settings=SETTINGS,
            source_image=source_image,
            strength=spec.get("strength"),
            hint_images=hint_images or None,
        )
        filenames = []
        for png in pngs:
            COMFY_COUNTER += 1
            name = f"{spec['prefix']}_{COMFY_COUNTER:05d}_.png"
            (IMAGE_DIR / name).write_bytes(png)
            filenames.append(name)
        job.update(status="success", images=filenames, error=None)
    except HTTPException as exc:
        job.update(status="error", images=[], error=str(exc.detail))
        await _comfy_broadcast(
            {"type": "execution_error", "data": {"prompt_id": prompt_id, "message": str(exc.detail)}}
        )
    except Exception as exc:  # reported through /history
        job.update(status="error", images=[], error=str(exc))
        await _comfy_broadcast(
            {"type": "execution_error", "data": {"prompt_id": prompt_id, "message": str(exc)}}
        )
    finally:
        await _comfy_broadcast({"type": "executing", "data": {"node": None, "prompt_id": prompt_id}})


def _comfy_entry(prompt_id: str) -> dict[str, Any]:
    job = COMFY_JOBS[prompt_id]
    images = [
        {"filename": name, "subfolder": "", "type": "output"} for name in job.get("images") or []
    ]
    status = (
        {"status_str": "success", "completed": True, "messages": []}
        if job["status"] == "success"
        else {"status_str": "error", "completed": True, "messages": [job.get("error") or "error"]}
    )
    return {
        "prompt": [0, prompt_id, job["graph"], {}, []],
        "outputs": {job["save_node"]: {"images": images}} if images else {},
        "status": status,
    }


@app.post("/prompt")
async def comfy_submit_prompt(request: Request):
    body = await request.json()
    graph = body.get("prompt") if isinstance(body, dict) else None
    try:
        spec = parse_comfy_graph(graph or {})
    except ComfyGraphError as exc:
        node_errors = {
            (exc.node_id or "workflow"): {
                "errors": [{"message": str(exc), "type": "unsupported_workflow"}]
            }
        }
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": "prompt_outputs_failed_validation",
                    "message": str(exc),
                    "details": "",
                    "extra_info": {},
                },
                "node_errors": node_errors,
            },
        )

    prompt_id = uuid.uuid4().hex
    COMFY_JOBS[prompt_id] = {
        "graph": graph,
        "save_node": spec["save_node"],
        "status": "pending",
        "images": [],
        "error": None,
    }
    # Keep memory bounded; clients only poll the most recent ids.
    for stale in list(COMFY_JOBS)[:-200]:
        del COMFY_JOBS[stale]
    asyncio.create_task(_run_comfy_job(prompt_id, spec))
    return {"prompt_id": prompt_id, "number": len(COMFY_JOBS), "node_errors": {}}


@app.get("/history/{prompt_id}")
async def comfy_history(prompt_id: str):
    job = COMFY_JOBS.get(prompt_id)
    if job is None or job["status"] == "pending":
        return {}
    return {prompt_id: _comfy_entry(prompt_id)}


@app.get("/history")
async def comfy_history_all(max_items: int = 100):
    done = [pid for pid, job in COMFY_JOBS.items() if job["status"] != "pending"]
    return {pid: _comfy_entry(pid) for pid in done[-max_items:]}


@app.get("/view")
async def comfy_view(filename: str, subfolder: str = "", type: str = "output"):  # noqa: A002
    # subfolder/type are part of the ComfyUI contract but everything is flat here.
    _ = (subfolder, type)
    name = os.path.basename(filename)
    for directory in (IMAGE_DIR, INPUT_DIR):
        path = directory / name
        if path.is_file():
            return FileResponse(path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Not found")


@app.get("/object_info")
async def comfy_object_info():
    samplers = list(SAMPLERS)
    return {
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [[SETTINGS["model"]]]}},
            "output": ["MODEL", "CLIP", "VAE"],
            "output_name": ["MODEL", "CLIP", "VAE"],
            "name": "CheckpointLoaderSimple",
            "category": "loaders",
            "output_node": False,
        },
        "CLIPTextEncode": {
            "input": {"required": {"text": ["STRING", {"multiline": True}]}},
            "output": ["CONDITIONING"],
            "output_name": ["CONDITIONING"],
            "name": "CLIPTextEncode",
            "category": "conditioning",
            "output_node": False,
        },
        "EmptyLatentImage": {
            "input": {
                "required": {
                    "width": ["INT", {"default": 512, "min": 64, "max": 2048, "step": 64}],
                    "height": ["INT", {"default": 512, "min": 64, "max": 2048, "step": 64}],
                    "batch_size": ["INT", {"default": 1, "min": 1, "max": 4}],
                }
            },
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "name": "EmptyLatentImage",
            "category": "latent",
            "output_node": False,
        },
        "KSampler": {
            "input": {
                "required": {
                    "model": ["MODEL"],
                    "positive": ["CONDITIONING"],
                    "negative": ["CONDITIONING"],
                    "latent_image": ["LATENT"],
                    "seed": ["INT", {"default": 0, "min": 0, "max": 2147483647}],
                    "steps": ["INT", {"default": 5, "min": 1, "max": 200}],
                    "cfg": ["FLOAT", {"default": 1.7}],
                    "sampler_name": [samplers],
                    "scheduler": [["normal", "karras", "simple"]],
                    "denoise": ["FLOAT", {"default": 1.0}],
                }
            },
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "name": "KSampler",
            "category": "sampling",
            "output_node": False,
        },
        "LoadImage": {
            "input": {"required": {"image": [_uploaded_images()]}},
            "output": ["IMAGE", "MASK"],
            "output_name": ["IMAGE", "MASK"],
            "name": "LoadImage",
            "category": "image",
            "output_node": False,
        },
        "VAEDecode": {
            "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "name": "VAEDecode",
            "category": "latent",
            "output_node": False,
        },
        "SaveImage": {
            "input": {
                "required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {"default": "DrawThings"}]}
            },
            "output": [],
            "output_name": [],
            "name": "SaveImage",
            "category": "image",
            "output_node": True,
        },
    }


@app.get("/queue")
async def comfy_queue():
    pending = [[i, pid] for i, pid in enumerate(COMFY_JOBS) if COMFY_JOBS[pid]["status"] == "pending"]
    return {"queue_running": pending[:1], "queue_pending": pending[1:]}


@app.post("/interrupt")
async def comfy_interrupt():
    return {}


@app.post("/queue")
async def comfy_queue_manage():
    return {}


@app.get("/system_stats")
async def comfy_system_stats():
    return {
        "system": {"os": sys.platform, "comfyui_version": "dt-bridge", "python_version": sys.version.split()[0]},
        "devices": [{"name": f"Draw Things ({DT_ADDR}:{DT_PORT})", "type": "gpu", "vram_total": 0, "vram_free": 0}],
    }


@app.websocket("/ws")
async def comfy_ws(websocket: WebSocket):
    await websocket.accept()
    COMFY_CLIENTS.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:  # client disconnect
        logging.getLogger("uvicorn.error").debug("comfy ws client disconnected")
    finally:
        if websocket in COMFY_CLIENTS:
            COMFY_CLIENTS.remove(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=BRIDGE_HOST, port=BRIDGE_PORT)

