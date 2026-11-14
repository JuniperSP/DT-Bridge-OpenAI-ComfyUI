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

Fresh Mac? Clone and run setup (idempotent, safe to re-run):

```sh
git clone https://github.com/JuniperSP/DT-Bridge-OpenAI-ComfyUI.git
cd DT-Bridge-OpenAI-ComfyUI
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
| `m` | **models panel** (open DT `Models` folder in Finder, auto-link custom models) |
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

The picker merges two sources from the app:

1. **Named presets** (`★`) — read from `Documents/Models/custom_configs.json`,
   i.e. configurations you saved in Draw Things with **Save configuration**.
   These are listed first because they are the only place a preset keeps its name.
2. **Recent generations** — read from the per-project SQLite workspaces
   (`TensorHistoryNode`), newest first by `wall_clock`.

```
┌ Load preset · 44/44 · newest first ──────────────────────────────────────────┐
│ filter:                                                                      │
│            ★AAA           1024x1344   5st LCM           l5                   │
│            ★SDLCM8bits     768x512   26st DPMPP2MKarras l1                   │
│ 11-02 02:48 SD1.5LCM      1024x1344   5st LCM           l5  selfie face…     │
└ type to filter · ↑↓ select · Enter load · Esc cancel ────────────────────────┘
```

Typing filters on preset name, prompt, model, sampler, project and LoRA
filenames (space = AND). `Esc` clears the filter, a second `Esc` closes.

Loading a preset pushes the whole configuration into the running bridge
(`POST /admin/config`, which also rewrites `config.json`): model, LoRAs
(file/weight/mode), steps, sampler, seed mode, cfg, clip-skip, size, hires-fix,
upscaler + scale factor, refiner, face restoration, mask blur, tiling **and its
tile sizes**, multi-stage sampler knobs, text-encoder switches, tea-cache, CFG
zero, prompt expansion, colour calibration, compression and the video/audio
fields - i.e. every key the app stores except the ones listed below. The stored
seed is dropped so generations stay random, and inline `<lora:...>` tags are
stripped from the prompt because the LoRA list already carries them. Tune in the
Draw Things app, save or generate something you like, then load it here - no
manual copying of values.

Two things do not travel yet:

- `controls` (ControlNet) needs a source image, so it is skipped and reported
  instead of being dropped silently: `/healthz` then lists `"unsupported":
  ["controls"]` and the TUI shows `not sent: controls` after loading.
- `name` is preset metadata, not a generation setting.

The bridge writes pixel sizes; the FlatBuffer wants 64 px blocks, so widths,
heights and tile sizes are converted on the way out.

Textual inversions (`custom_textual_inversions.json`) are checked by
`./dt doctor`; the bridge itself does not need them because Draw Things loads an
embedding whenever its keyword appears in the prompt.

Inspect the presets without the UI:

```sh
.venv/bin/python dtconfig.py --named          # named presets only
.venv/bin/python dtconfig.py --limit 15       # recent history
.venv/bin/python dtconfig.py --summary        # counts + missing TI files
```

### Custom models & external folders

Draw Things keeps its `Models` folder deep inside its macOS container
(`~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models`), which
Finder hides by default. Instead of hunting for that folder by hand, press `m`
in `./dt tui` (or use `./dt models`):

- **Open in Finder (`o` in the Models panel, or `./dt models open`)** — opens
  Draw Things' internal `Models` folder directly in Finder.
- **Drop into `./models` (`p` in the Models panel, or `./dt models local`)** —
  drop `.ckpt` / `.safetensors` files into the local `./models` folder inside
  this project; `./dt start` (or `s` in the Models panel) automatically
  symlinks (`ln -s`, 0 bytes extra disk space) them into Draw Things' `Models`
  folder.
- **Link an external folder (`f` in the Models panel, or `./dt models dir <path>`)** —
  point `DT_EXTRA_MODELS_DIR` at `~/Downloads` or an external SSD and every
  model file in that folder is auto-linked into Draw Things' `Models` folder.

Prefer commands? Same thing without the UI:

```sh
./dt setup      # first-time bootstrap on a fresh Mac
./dt start      # gRPC server + bridge (waits until both answer)
./dt status     # what is up, plus model/health summary
./dt stop       # stop gRPC + bridge
./dt restart
./dt restart-bridge  # restart only the HTTP bridge
./dt models          # open Draw Things' hidden Models folder in Finder
./dt models local    # open ./models drop-folder in Finder
./dt models sync     # symlink custom models into Draw Things Models
./dt models dir ~/Downloads  # auto-link models from a custom folder
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
| `POST` | `/v1/images/edits` | multipart img2img: `image` + `prompt` (+ `strength`, optional `control_image`/`control_type`) |
| `POST` | `/upload/image` | ComfyUI's upload endpoint; stores a source image for `LoadImage` |
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

`config.json` accepts every key Draw Things writes into a preset, spelled the
same way (`resolutionDependentShift`, `stage2Steps`, `teaCacheThreshold`, ...),
so a saved configuration can be pasted in as-is. Units follow the app: sizes and
tile widths in pixels, sampler/seed-mode/colour/compression enums as either
numbers or names. Anything missing falls back to `DEFAULT_SETTINGS`.

`controls` is accepted, stored and sent: the ControlNet models come from the
preset and each request supplies its own control image (see ControlNet above).
`/healthz` lists the loaded control models under `controls`.

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
| source image | `LoadImage.image` → a file uploaded through `/upload/image` |
| strength | `KSampler.denoise` (0–1), only when a `LoadImage` node is present |
| control image | `LoadImage` wired into `ControlNetApply*` (sent as a `custom` hint) |

Everything else (`steps`, `cfg`, `sampler_name`, `scheduler`, `ckpt_name`,
`LoraLoader`, `VAEDecode`) is **ignored on purpose** - steps/sampler/LoRAs come
from `config.json` so every client renders with the same tuned pipeline.
Workflows with no usable prompt return HTTP 400 with `node_errors` instead of
silently producing the wrong picture.

img2img works the way ComfyUI clients expect: POST the picture to
`/upload/image`, then reference the returned name from a `LoadImage` node and
set `KSampler.denoise` to the strength you want. If the workflow never sizes a
latent, the source image's own size (snapped to 64 px) is used. Inpainting
(`LoadImageMask` / `mask`) is not wired yet.

### ControlNet

Draw Things receives *pre-processed* hints, not photos, so a control needs two
pieces: the control entries from the preset (`controls` in `config.json`) and a
control image per request.

- **OpenAI surface** — POST the control picture as `control_image` next to the
  normal fields, with `control_type` naming the map you are sending (`pose`,
  `depth`, `canny`, `lineart`, `tile`, ...; default `custom`).
- **ComfyUI surface** — wire a `LoadImage` into `ControlNetApplyAdvanced`; it
  travels as a `custom` hint. `ControlNetLoader.control_net_name` is ignored:
  the control model comes from the preset.

Without a control image the control entries have no effect at all (same picture
as with no control). `scribble` hints need a 1-byte tensor and answer 501.

Point the client at `http://<lan-ip>:9955` (no `/v1` suffix for ComfyUI).

## Implementation notes

- Source images (img2img, `LoadImage`) are converted to Draw Things' own NNC
  tensor frame - `[u32 identifier=0][64B param][fp16 NHWC]`, pixels in [-1, 1] -
  and referenced by `sha256` in `contents`. The server never accepts PNG/JPEG for
  `image`, and a PNG payload hangs it without a log line.
- Sizes are capped at `DT_MAX_SIZE` (default 2048 px) because a MacBook swaps or
  drops requests below what a cloud GPU would allow.

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

## Publishing (public mirror)

This repository is the development copy. A sanitised snapshot is published to a
separate public mirror by `./publish.sh`:

```sh
./publish.sh --dry-run     # show the diff that would be published
./publish.sh               # publish
```

How it works: the script takes the **tree** of the current commit, scans it for
personal strings (model/LoRA names, home paths, tokens) and aborts if anything
matches, then creates one commit whose parent is the mirror's current tip and
pushes only that. Private history is never published, so an early commit that
carried personal defaults cannot leak.

Configure the target with `DT_PUBLIC_REPO` / `DT_PUBLIC_ACCOUNT` (defaults are
set inside the script).

## Troubleshooting

**`git push` fails with `Repository not found` on a machine with several GitHub
accounts** - git resolved credentials for the wrong account; a token without
access to a private repo is answered with 404. Check which account git uses:

```sh
printf 'protocol=https\nhost=github.com\n\n' | git credential fill | grep username
```

Then pin this repository to `gh`'s active account:

```sh
gh auth switch --user <your-account>
git config --local --unset-all credential.https://github.com.helper
git config --local --add credential.https://github.com.helper ''
git config --local --add credential.https://github.com.helper '!gh auth git-credential'
```

**`/healthz` reports `models_ready: false`** - the gRPC server cannot read the
Models folder. Grant Full Disk Access to the terminal you launch `./dt` from
(System Settings → Privacy & Security → Full Disk Access), then restart it.
