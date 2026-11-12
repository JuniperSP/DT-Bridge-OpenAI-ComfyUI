# DT-Bridge-OpenAI-ComfyUI

Lightweight companion bridge for **Draw Things** on macOS — exposes your own
Draw Things setup as OpenAI & ComfyUI compatible HTTP APIs over LAN or
Tailscale.

Instead of configuring models and parameters from scratch, you **set up and tune
everything inside the Draw Things app first**, then load your preset straight
into this bridge with one keystroke. Any client that speaks the OpenAI images
API or ComfyUI HTTP API (SillyTavern, RisuAI, LangChain, AutoGen, the `openai`
SDK, plain `curl`) can then generate images using your exact Draw Things
pipeline.

```
OpenAI REST (JSON)  ->  gRPC ImageGenerationService/GenerateImage
                        -> flatbuffer GenerationConfiguration + prompt
                        <- NNC fp16 tensor  ->  PNG
```

## Run

Fresh Mac? Start here (idempotent, safe to re-run):

```sh
./dt setup          # uv/venv/deps, gen/ bindings, gRPC binary, permissions
./dt doctor         # verify everything, exits non-zero on problems
```

Then one command opens a full-screen terminal UI (stdlib `curses`, no browser):

```sh
./dt tui
```

> **First-time required step:** No model is pre-configured out of the box.
> Open the **Draw Things** app, pick your model/LoRAs/sampler, and generate at
> least one image you like. Then in `./dt tui`, press `s` to start and press `l`
> to load that preset from Draw Things (saves to `config.json`).

| Key | Action |
|---|---|
| `s` | start gRPC server + bridge |
| `x` | stop both |
| `r` | restart |
| `l` | **load a preset from Draw Things** (↑↓ select, Enter apply) |
| `n` | **network panel** (bind mode, API key, Tailscale HTTPS) |
| `g` / `b` | show `grpc.log` / `bridge.log` |
| `p` | test-generate (prompt is prefilled from the loaded preset) |
| `o` | open the `output/` folder in Finder |
| `q` | quit the UI (services keep running) |

### Use it over Tailscale (HTTPS)

Press `n` in the TUI, then:

| Key | Action |
|---|---|
| `b` | cycle bind: local only → LAN + tailnet → tailnet only |
| `t` | toggle **Tailscale HTTPS** (runs `tailscale serve --https=443`) |
| `k` / `d` | set / clear an API key |
| `a` | apply: writes `bridge.env` and restarts the bridge |

With HTTPS on, Tailscale terminates TLS and proxies to `127.0.0.1:9955`:

```
https://<machine>.<tailnet>.ts.net/v1     # OpenAI compatible
https://<machine>.<tailnet>.ts.net        # ComfyUI compatible
```

So the Mac only listens on loopback and the tailnet gets a real certificate -
no port forwarding, no self-signed certs. Turning HTTPS off removes only the
`:443` handler; other `tailscale serve` entries on the machine are left alone.
`./dt url` prints whatever is currently reachable.

`bridge.env` is plain shell syntax and is sourced by `dt` on every run:

```sh
BRIDGE_HOST=127.0.0.1     # 0.0.0.0 to expose on the LAN too
BRIDGE_PORT=9955
BRIDGE_API_KEY=            # optional bearer token for /v1/*
```

### Presets straight from Draw Things

The app stores every generation as a `TensorHistoryNode` row in its SQLite
workspace. `dtconfig.py` reads those rows and `l` in the TUI lists them,
newest first (by the node's `wall_clock`), with a live filter:

```
┌ Load preset · 9/42 · newest first ───────────────────────────────────────────┐
│ filter: sdxl                                                                 │
│ 11-02 02:34  768x576    18st DPMPP2MKarras l2   sd_xl_base_1.0…  a cozy…     │
└ type to filter · ↑↓ select · Enter load · Esc cancel ────────────────────────┘
```

Typing filters on prompt, model name and LoRA filenames (space = AND). `Esc`
clears the filter, a second `Esc` closes the picker.

Loading a preset pushes model, LoRAs, steps, sampler, cfg, clip-skip, size,
hires-fix and upscaler into the running bridge (`POST /admin/config`, which also
rewrites `config.json`). The stored seed is dropped so generations stay random,
and inline `<lora:...>` tags are stripped from the prompt because the LoRA list
already carries them. Tune in the Draw Things app, generate one image you like,
then load that preset here - no manual copying of values.

Inspect the presets without the UI:

```sh
.venv/bin/python dtconfig.py --limit 15        # table
.venv/bin/python dtconfig.py --json --limit 5  # raw entries
```

Prefer commands? Same thing without the UI:

```sh
./dt setup      # first-time bootstrap on a fresh Mac
./dt start      # gRPC server + bridge (waits until both answer)
./dt status     # what is up, plus model/health summary
./dt stop       # stop gRPC + bridge
./dt restart
./dt restart-bridge  # restart only the HTTP bridge
./dt logs grpc  # or: ./dt logs bridge -f
./dt doctor     # venv/deps/bindings/models/tailscale, exits non-zero on issues
./dt url        # prints LAN / tailnet / https endpoints
./dt ui         # optional browser dashboard on :9700
./dt shutdown   # stop everything, including the web panel
```

`dt` only stops processes it started (pid files live in `.run/`); if you already
have a gRPC server running by hand it reuses it and says so.

### Manual

1. Start the Draw Things gRPC server (keep it on loopback, plaintext, uncompressed):

   ```sh
   gRPCServerCLI-macOS "$HOME/Library/Containers/com.liuliu.draw-things/Data/Documents/Models" \
     --address 127.0.0.1 --port 7859 --no-tls --no-response-compression
   ```

2. Start the bridge:

   ```sh
   uv run bridge.py          # or: .venv/bin/python bridge.py
   ```

3. Check it:

   ```sh
   curl -s localhost:9955/healthz
   # {"status":"ok", ..., "models_ready":true, "missing":[]}
   ```

## Endpoints

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/images/generations` | `prompt`, `n`, `size`, `response_format`, plus extras below |
| `GET`  | `/v1/models` | lists the configured model |
|`GET` | `/healthz` | pings the server and reports missing model/lora files |
| `GET`/`POST` | `/admin/config` | read or live-update the generation settings (also rewrites `config.json`) |
| `GET`  | `/images/{name}` | files served by `response_format: "url"` |

Request fields accepted beyond the OpenAI spec: `negative_prompt`, `steps`,
`guidance_scale`, `sampler`, `width`/`height`.

```sh
curl -s -X POST localhost:9955/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red apple on a wooden table","size":"512x512","n":1,"response_format":"url"}'
```

## Configuration

No model is configured by default (`"model": ""` in `DEFAULT_SETTINGS` in
`bridge.py`); requests will return HTTP 400 until you configure one. Load a
preset from Draw Things in `./dt tui` (press `l`) or create `config.json` next
to `bridge.py` manually:

```json
{
  "model": "sd_xl_base_1.0_f16.ckpt",
  "steps": 6,
  "hiresFix": false,
  "loras": [{ "file": "lcm_sd_xl_base_1.0_lora_f16.ckpt", "weight": 1.0, "mode": "all" }]
}
```

Environment variables: `DT_ADDR`, `DT_PORT`, `DT_TLS` (1 = TLS + ALPN `h2`),
`BRIDGE_HOST`, `BRIDGE_PORT`, `BRIDGE_API_KEY`, `DT_TIMEOUT`, `DT_CONFIG`.

## Client setup

Point the client at `http://<lan-ip>:9955/v1`, any API key (unless
`BRIDGE_API_KEY` is set), model name from `/v1/models`.

- **SillyTavern** — Image Generation → source "OpenAI" → endpoint
  `http://<lan-ip>:9955/v1` (it appends `/images/generations`).
- **RisuAI** — image generation → OpenAI-compatible, base URL
  `http://<lan-ip>:9955/v1`.
- **openai SDK**:

  ```python
  from openai import OpenAI
  client = OpenAI(base_url="http://<lan-ip>:9955/v1", api_key="local")
  img = client.images.generate(prompt="a red apple", size="512x512", response_format="b64_json")
  ```

## ComfyUI-compatible surface

The same generation pipeline is also reachable through the ComfyUI HTTP API,
for clients (SillyTavern / RisuAI "ComfyUI" sources) that POST a workflow graph:

| Method | Path |
|---|---|
| `POST` | `/prompt` |
| `GET`  | `/history/{prompt_id}`, `/history` |
| `GET`  | `/view?filename=&subfolder=&type=output` |
| `GET`  | `/object_info` (only the nodes listed below) |
| `GET`  | `/queue`, `POST /queue`, `POST /interrupt`, `GET /system_stats` |
| `WS`   | `/ws` (execution progress; optional for most clients) |

The graph is not executed - it is inspected for the values that matter:

| Value | Read from |
|---|---|
| prompt / negative | `CLIPTextEncode` reached from `KSampler.positive` / `.negative` (linked primitives resolved) |
| width / height / batch | `EmptyLatentImage` (or `EmptySD3LatentImage`) reached from `latent_image` |
| seed | `KSampler.seed` / `KSamplerAdvanced.noise_seed` (random when absent) |
| filename | `SaveImage.filename_prefix` → `ComfyUI_00001_.png` |

Everything else (`steps`, `cfg`, `sampler_name`, `scheduler`, `ckpt_name`,
`LoraLoader`, `VAEDecode`) is **ignored on purpose** - steps/sampler/LoRAs come
from `config.json` so every client renders with the same tuned pipeline.
Workflows with no usable prompt return HTTP 400 with `node_errors` instead of
silently producing the wrong picture. img2img / `LoadImage` graphs are not
supported yet.

Point the client at `http://<lan-ip>:9955` (no `/v1` suffix for ComfyUI).

## Implementation notes

- `size` is snapped to multiples of 64 px; Draw Things takes
  `start_width` in 64 px units.
- `n` maps to `batchSize` (one GPU pass, several images). `batchCount` is
  client-side in Draw Things and is ignored by the server.
- Empty `upscaler` / `refinerModel` / `faceRestoration` are written as *absent*
  flatbuffer fields. Writing `""` makes the server fall back to a default 4x
  upscaler, which returns out-of-range tensors.
- The server returns raw NNC tensors, not PNG:
  `[uint32 identifier][ccv_nnc_tensor_param_t 64B][fp16 data]`, layout
  `[batch, H, W, C]`, values in `[-1, 1]`. `decode_tensor_png()` mirrors
  `DrawThingsCLI.writePNG()`: `pixel = clamp((value + 1) * 127.5)`.
- Generations are serialized with a semaphore, matching Draw Things' single
  generation queue.

## Bindings

`gen/` holds generated code (`regen.sh` rebuilds it):

- `proto/imageService.proto` → `imageService_pb2.py`
- `schema/config.fbs` → `GenerationConfiguration.py`, `LoRA.py`, enums
- `schema/tensor_history.fbs` → `TensorHistoryNode.py` (used by `dtconfig.py` to
  read Draw Things' own history DB)
