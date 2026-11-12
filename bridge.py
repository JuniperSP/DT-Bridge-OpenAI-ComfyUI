# /// script
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "grpclib",
#     "flatbuffers",
#     "protobuf",
#     "numpy",
#     "pillow",
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gen"))

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
TENSOR_HEADER = 68
IMAGE_DIR = HERE / "output"
IMAGE_DIR.mkdir(exist_ok=True)

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
}
SEED_MODES = {"Legacy": 0, "TorchCpuCompatible": 1, "ScaleAlike": 2, "NvidiaGpuCompatible": 3}
LORA_MODES = {"all": 0, "base": 1, "refiner": 2}
COLOR_CALIBRATIONS = {"none": 0, "disabled": 0, "lab": 1}

# Built-in fallback defaults before a Draw Things preset is loaded into config.json.
DEFAULT_SETTINGS: dict[str, Any] = {
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
    "maskBlur": 2.5,
    "maskBlurOutset": 0,
    "width": 512,
    "height": 512,
    "batchCount": 1,
    "batchSize": 1,
    "hiresFix": False,
    "hiresFixStrength": 0.7,
    "hiresFixWidth": 0,
    "hiresFixHeight": 0,
    "upscaler": "",
    "faceRestoration": "",
    "refinerModel": "",
    "refinerStart": 0.7,
    "colorCalibration": "none",
    "tiledDecoding": False,
    "tiledDiffusion": False,
    "preserveOriginalAfterInpaint": True,
    "numFrames": 1,
    "loras": [],
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
    return table.get(str(value), default)


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

    Note: start_width / start_height are expressed in 64 px units.
    """
    b = flatbuffers.Builder(2048)
    model_off = b.CreateString(model or settings["model"])
    # Optional strings must stay absent (nil) when empty: the server treats an
    # empty upscaler/refiner/face-restoration as "use the default" and would
    # e.g. apply a 4x upscaler to every image.
    upscaler_off = b.CreateString(settings["upscaler"]) if settings.get("upscaler") else 0
    face_off = b.CreateString(settings["faceRestoration"]) if settings.get("faceRestoration") else 0
    refiner_off = b.CreateString(settings["refinerModel"]) if settings.get("refinerModel") else 0

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

    GC.GenerationConfigurationStart(b)
    GC.GenerationConfigurationAddId(b, 0)
    GC.GenerationConfigurationAddStartWidth(b, max(1, width // 64))
    GC.GenerationConfigurationAddStartHeight(b, max(1, height // 64))
    GC.GenerationConfigurationAddSeed(b, seed & 0xFFFFFFFF)
    GC.GenerationConfigurationAddSteps(b, int(settings["steps"]))
    GC.GenerationConfigurationAddGuidanceScale(b, float(settings["guidanceScale"]))
    GC.GenerationConfigurationAddStrength(b, float(settings.get("strength", 1.0)))
    GC.GenerationConfigurationAddModel(b, model_off)
    GC.GenerationConfigurationAddSampler(b, _enum(settings.get("sampler"), SAMPLERS))
    GC.GenerationConfigurationAddBatchCount(b, batch_count)
    GC.GenerationConfigurationAddBatchSize(b, batch_size)
    GC.GenerationConfigurationAddHiresFix(b, bool(settings.get("hiresFix")))
    GC.GenerationConfigurationAddHiresFixStartWidth(b, int(settings.get("hiresFixWidth") or 0) // 64)
    GC.GenerationConfigurationAddHiresFixStartHeight(b, int(settings.get("hiresFixHeight") or 0) // 64)
    GC.GenerationConfigurationAddHiresFixStrength(b, float(settings.get("hiresFixStrength", 0.7)))
    if upscaler_off:
        GC.GenerationConfigurationAddUpscaler(b, upscaler_off)
    GC.GenerationConfigurationAddSeedMode(b, _enum(settings.get("seedMode"), SEED_MODES))
    GC.GenerationConfigurationAddClipSkip(b, int(settings.get("clipSkip", 1)))
    if loras_vec:
        GC.GenerationConfigurationAddLoras(b, loras_vec)
    GC.GenerationConfigurationAddMaskBlur(b, float(settings.get("maskBlur", 0)))
    GC.GenerationConfigurationAddMaskBlurOutset(b, int(settings.get("maskBlurOutset", 0)))
    if face_off:
        GC.GenerationConfigurationAddFaceRestoration(b, face_off)
    if refiner_off:
        GC.GenerationConfigurationAddRefinerModel(b, refiner_off)
    GC.GenerationConfigurationAddRefinerStart(b, float(settings.get("refinerStart", 0.7)))
    GC.GenerationConfigurationAddNumFrames(b, int(settings.get("numFrames", 1)))
    GC.GenerationConfigurationAddSharpness(b, float(settings.get("sharpness", 0)))
    GC.GenerationConfigurationAddShift(b, float(settings.get("shift", 1.0)))
    GC.GenerationConfigurationAddPreserveOriginalAfterInpaint(
        b, bool(settings.get("preserveOriginalAfterInpaint", True))
    )
    GC.GenerationConfigurationAddTiledDecoding(b, bool(settings.get("tiledDecoding")))
    GC.GenerationConfigurationAddTiledDiffusion(b, bool(settings.get("tiledDiffusion")))
    GC.GenerationConfigurationAddColorCalibration(
        b, _enum(settings.get("colorCalibration", "none"), COLOR_CALIBRATIONS)
    )
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
        self, *, prompt: str, negative_prompt: str, configuration: bytes
    ) -> list[bytes]:
        assert self.channel is not None
        images: list[bytes] = []
        buffer = bytearray()
        async with self.channel.request(
            "/ImageGenerationService/GenerateImage",
            Cardinality.UNARY_STREAM,
            pb.ImageGenerationRequest,
            pb.ImageGenerationResponse,
            timeout=DT_TIMEOUT,
        ) as stream:
            await stream.send_message(
                pb.ImageGenerationRequest(
                    prompt=prompt,
                    negativePrompt=negative_prompt,
                    configuration=configuration,
                    user="openai-bridge",
                    chunked=True,
                ),
                end=True,
            )
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
) -> list[bytes]:
    """Run one generation and return PNG bytes (one per image)."""
    chosen_model = model or settings.get("model") or ""
    if not chosen_model:
        raise HTTPException(
            status_code=400,
            detail="No model configured. Load a preset in './dt tui' (press l) or set 'model' in config.json.",
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
    width = max(64, (width // 64) * 64)
    height = max(64, (height // 64) * 64)
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
    }


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

    created = int(time.time())
    data: list[dict[str, str]] = []
    for png in pngs:
        if (payload.response_format or "b64_json") == "url":
            name = f"{uuid.uuid4().hex}.png"
            (IMAGE_DIR / name).write_bytes(png)
            data.append({"url": str(request.base_url).rstrip("/") + f"/images/{name}"})
        else:
            data.append({"b64_json": base64.b64encode(png).decode("ascii")})
    return {"created": created, "data": data}


@app.get("/images/{name}")
async def get_image(name: str):
    path = IMAGE_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# ComfyUI-compatible surface (subset)
#
# Clients such as SillyTavern and RisuAI talk to ComfyUI by POSTing a workflow
# graph and then polling /history + fetching /view. We only interpret the nodes
# that describe *what* to draw (prompt, negative, size, seed, filename); steps,
# cfg, sampler, checkpoint and LoRAs stay under our control via config.json.
# ---------------------------------------------------------------------------

SAMPLER_NODES = {"KSampler", "KSamplerAdvanced", "SamplerCustom"}
LATENT_NODES = {"EmptyLatentImage", "EmptySD3LatentImage", "EmptySDXLLatentImage"}
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


def _conditioning_text(graph: dict[str, Any], value: Any) -> str:
    link = _link(value)
    if link is None:
        return ""
    found = _walk_up(graph, link[0], TEXT_NODES)
    if found is None:
        return ""
    _, node = found
    text = _literal(graph, (node.get("inputs") or {}).get("text"))
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
    prompt = _conditioning_text(graph, inputs.get("positive"))
    negative = _conditioning_text(graph, inputs.get("negative"))
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

    def _int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    return {
        "prompt": prompt,
        "negative": negative,
        "width": _int(width, int(SETTINGS["width"])),
        "height": _int(height, int(SETTINGS["height"])),
        "batch": max(1, min(_int(batch, 1), 4)),
        "seed": _int(seed, -1),
        "save_node": save_id or "0",
        "prefix": str(prefix) if prefix else "DrawThings",
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
        width = max(64, (spec["width"] // 64) * 64)
        height = max(64, (spec["height"] // 64) * 64)
        pngs = await render_pngs(
            prompt=spec["prompt"],
            negative_prompt=spec["negative"],
            width=width,
            height=height,
            seed=seed,
            count=spec["batch"],
            settings=SETTINGS,
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
    path = IMAGE_DIR / os.path.basename(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="image/png")


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

