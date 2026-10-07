# AGENTS.md

Agent-facing guide to **ffmpegcv** — a pure-Python, OpenCV-compatible video
reader/writer backed by the `ffmpeg` CLI.

This file is the single source of truth for coding agents working **in** this
repository or building code **with** this package. Human-facing docs live in
[README.md](./README.md) / [README_CN.md](./README_CN.md).

> Deeper references: [docs/agents/quickstart.md](docs/agents/quickstart.md),
> [api-reference.md](docs/agents/api-reference.md),
> [architecture.md](docs/agents/architecture.md),
> [troubleshooting.md](docs/agents/troubleshooting.md),
> [testing.md](docs/agents/testing.md). Machine index: [llms.txt](./llms.txt).

---

## 1. TL;DR

```python
import ffmpegcv
import numpy as np

# Read a file (zero-copy iterator; always close).
with ffmpegcv.VideoCapture("in.mp4", pix_fmt="bgr24") as cap:
    print(cap.fps, cap.count, cap.size)          # float, int, (w, h)
    for frame in cap:                             # frame: np.uint8, (h, w, 3)
        pass

# Write a file. Size is inferred from the first written frame.
with ffmpegcv.VideoWriter("out.mp4", None, 30) as out:
    out.write(np.zeros((240, 320, 3), np.uint8))  # codec defaults to h264
```

Hard rules for this package:

1. `ffmpeg` **and** `ffprobe` must be on `PATH`; importing `ffmpegcv` fails otherwise.
2. **Always** `release()` or use `with` — every reader/writer owns a child ffmpeg process.
3. Frames are **read-only** `numpy` views (`np.frombuffer`). Copy before mutating.
4. Resize sizes must be **even**. Use `>=2` even width/height everywhere.
5. `len(cap)` works for file readers and panels; camera readers have no `__len__`.
6. GPU constructors (`*NV`, `*QSV`) raise on machines without that hardware — expected.

---

## 2. What this project is

- **Package**: `ffmpegcv` (PyPI), version in [`ffmpegcv/version.py`](ffmpegcv/version.py).
- **Runtime deps**: `numpy` only. Optional `pycuda` (`pip install ffmpegcv[cuda]`) for `toCUDA`.
- **External deps**: `ffmpeg` + `ffprobe` binaries (any recent version; 6.0+ recommended).
- **Mechanism**: builds an `ffmpeg ... -f rawvideo pipe:` command per stream and talks to it over
  `subprocess.PIPE`. No OpenCV, no ctypes bindings.
- **Compatibility**: the public API mirrors `cv2.VideoCapture` / `cv2.VideoWriter`, plus GPU
  paths, ROI filters, low-latency streams, background (noblock) I/O and CUDA tensor export.
- **Python support**: declared `>=3.6`; verified here on CPython 3.13.

Public names (verified exports): `VideoCapture`, `VideoCaptureNV`, `VideoCaptureQSV`,
`VideoCaptureCAM`, `VideoCaptureStream`, `VideoCaptureStreamRT`, `VideoCapturePannels`,
`VideoWriter`, `VideoWriterNV`, `VideoWriterQSV`, `VideoWriterStreamRT`,
`noblock`, `ReadLiveLast`, `toCUDA`, `get_num_NVIDIA_GPUs`,
`VideoReader*` aliases, and the `FFmpeg*` implementation classes.

---

## 3. Environment setup & fast verification

```bash
# 1. ffmpeg must exist (import-time check)
ffmpeg -version && ffprobe -version

# 2. Install the package in editable mode from the repo root
python -m pip install -e .

# 3. Behavioral regression suite (prints JSON + "unexpected failures: N")
PYTHONPATH=. python tests/compat_suite.py --out /tmp/ffmpegcv_compat.json
```

`PYTHONPATH=.` is required: `tests/` has no `__init__.py`, so running the file directly does
not put the repo root on `sys.path`. Exit code is `1` iff any expectation failed.

A green run looks like:

```
===COMPAT_JSON_END===
unexpected failures: 0
```

---

## 4. Task recipes

All snippets assume `import ffmpegcv`, `import numpy as np`.

### 4.1 Read a video file

```python
with ffmpegcv.VideoCapture("in.mp4") as cap:      # aliases: ffmpegcv.VideoReader
    n = len(cap)                                  # frame count (files only)
    while True:
        ret, frame = cap.read()                   # -> (bool, np.ndarray | None)
        if not ret:
            break
    # or: for frame in cap: ...
```

Reader attributes: `fps`, `count`, `duration`, `codec`, `width`, `height`, `size` (w,h),
`origin_width`, `origin_height`, `crop_width`, `crop_height`, `pix_fmt`, `out_numpy_shape`,
`iframe`, `filename`, `ffmpeg_cmd`.

### 4.2 Write a video file

```python
with ffmpegcv.VideoWriter("out.mp4", None, 30) as out:   # codec None -> "h264"
    out.write(frame1)                                    # size fixed by first frame
    out.write(frame2)
```

- Choose `mp4`/`mkv`, not `avi`.
- `codec` is a **string** (`"h264"`, `"hevc"`, `"mpeg4"`, `"h264_nvenc"`). A cv2 fourcc int
  emits a `UserWarning` and is ignored.
- Never pass `cv2.VideoWriter_fourcc(...)`.
- `resize=(w, h)` scales output; `preset=` forwards to ffmpeg; `bitrate="1M"` sets `-b:v`.
- Every `write()` must receive the same shape as the first frame, else `AssertionError`.

### 4.3 Copy / transcode

```python
with ffmpegcv.VideoCapture("in.mp4") as cap, \
     ffmpegcv.VideoWriter("out.mp4", None, cap.fps) as out:
    for frame in cap:
        out.write(frame)
```

### 4.4 ROI: crop, resize, pad

```python
ffmpegcv.VideoCapture(f, crop_xywh=(10, 20, 640, 480))                    # x, y, w, h
ffmpegcv.VideoCapture(f, resize=(640, 480))                               # stretch
ffmpegcv.VideoCapture(f, resize=(640, 480), resize_keepratio=True)         # letterbox
ffmpegcv.VideoCapture(f, resize=(640, 480), resize_keepratio=True,
                      resize_keepratioalign="topleft")                     # center|topleft|topright|bottomleft|bottomright
ffmpegcv.VideoCapture(f, crop_xywh=(0, 0, 640, 480), resize=(512, 512))    # crop then resize
ffmpegcv.VideoCapture(f, infile_options="-re -stream_loop -1")             # raw ffmpeg input flags
```

On the CPU path, odd `crop_xywh` values are silently floored to even with a printed warning;
odd `resize` values raise `AssertionError`. The NV path asserts crop values are even.

### 4.5 Pixel formats & resulting shapes

`pix_fmt` is an **input** pixel format and defines the numpy layout:

| `pix_fmt`             | frame shape      | notes |
| --------------------- | ---------------- | ----- |
| `bgr24` (default)     | `(h, w, 3)`      | OpenCV-compatible |
| `rgb24`               | `(h, w, 3)`      | for PIL / torchvision |
| `gray`                | `(h, w, 1)`      | `extractplanes=y` |
| `yuv420p` / `yuvj420p`| `(h*3//2, w)`    | raw planar, for `toCUDA` |
| `nv12`                | `(h*3//2, w)`    | raw semi-planar, best with GPU |

### 4.6 GPU (NVIDIA NVENC/NVDEC, Intel QSV)

```python
ffmpegcv.VideoCaptureNV(f, gpu=0)                 # checks ffmpeg NVENC + a real GPU
ffmpegcv.VideoCaptureQSV(f)                       # experimental; crop/resize not implemented
ffmpegcv.VideoWriterNV("out.mp4", "h264", 30)     # codec "h264" -> "h264_nvenc"
ffmpegcv.VideoWriterQSV("out.mp4", "hevc", 30)
```

These call `_check_nvidia()` / `get_num_QSV_GPUs()` and raise `RuntimeError`/`AssertionError`
when the hardware or ffmpeg build lacks support. `gpu` is taken modulo the detected GPU count.

### 4.7 CUDA tensor export (`toCUDA`)

```python
cap = ffmpegcv.toCUDA(ffmpegcv.VideoCaptureNV(f, pix_fmt="nv12"), tensor_format="chw")
ret, t = cap.read()            # pycuda GPUArray, float32, (3, h, w) for "chw"
ret, t = cap.read_torch()      # torch.Tensor on cuda:0 (torch is optional)
ret, m = cap.read_cudamem()    # raw pycuda DeviceAllocation
ret, _ = cap.read_torch(buf)   # write into caller-owned tensor (no alloc)
```

Constraints: source `pix_fmt` **must** be `yuv420p` or `nv12`; output is always RGB float32.
`tensor_format` is `"chw"` (default) or `"hwc"`. Requires `pycuda` and a working CUDA device.

### 4.8 Camera (`VideoCaptureCAM`) — experimental

```python
from ffmpegcv.ffmpeg_reader_camera import query_camera_devices, query_camera_options

cap = ffmpegcv.VideoCaptureCAM(0)                       # by index
cap = ffmpegcv.VideoCaptureCAM("Integrated Camera")     # by name
opts = query_camera_options(0)                          # list of supported dicts
cap = ffmpegcv.VideoCaptureCAM(0, **opts[-1], crop_xywh=(0, 0, 640, 480))
```

- Backed by a producer thread + a 30-frame queue; it keeps running until `release()`.
- macOS/`avfoundation` cannot be queried: pass `camsize_wh`, `camfps`, `campix_fmt` explicitly.
- Linux/`v4l2` cannot report FPS: leave `camfps` unset.
- Cameras expose **no** `fps`/`count`/`duration`/`codec` attributes and no `__len__` — only
  `width`, `height`, `size`, `origin_*`, `crop_*`, `camfps`, `camcodec`, `campix_fmt`.
  (`VideoCaptureStream`, a subclass, does set `fps`/`count`/`duration`/`codec`.)
- Prefer `cv2.VideoCapture` for plain camera capture; use this for ROI/name lookup.

### 4.9 Streams / IP cameras

```python
cap = ffmpegcv.VideoCaptureStream(url, timeout=5)      # RTSP/RTP/RTMP/HTTP(S)
cap = ffmpegcv.VideoCaptureStreamRT(url)               # low-latency, buffered producer
cap = ffmpegcv.ReadLiveLast(ffmpegcv.VideoCaptureStreamRT, url)  # always-latest frame
out = ffmpegcv.VideoWriterStreamRT("rtmp://host/app/key")         # libx264 + zerolatency
```

`StreamRT`/`ReadLiveLast` never terminate on their own — bound the loop and always `release()`.
`VideoWriterStreamRT` accepts only `h264`/`libx264`/`x264`/`mpeg4` and `bgr24`/`rgb24`/`gray`.

### 4.10 Multi-panel reader

```python
cap = ffmpegcv.VideoCapturePannels(f, [[0, 0, 640, 480], [640, 0, 640, 480]])
ret, panels = cap.read()
# all panels equal size -> np.ndarray (n, h, w, c); mixed sizes -> list of arrays
```

### 4.11 Background (noblock) I/O

```python
cap = ffmpegcv.noblock(ffmpegcv.VideoCapture, f, pix_fmt="rgb24")   # multiprocessing
# cap.read() returns an already-buffered frame; work overlaps with prefetch
cap.release()
out = ffmpegcv.noblock(ffmpegcv.VideoWriter, "out.mp4", None, 30)
```

`noblock` only accepts `VideoCapture`, `VideoCaptureNV`, `VideoWriter`, `VideoWriterNV`;
anything else raises `ValueError`. Buffers are a `multiprocessing.Array` ring with
`NFRAME = 10` slots and a `Queue(maxsize=8)` of slot indices.

> **Spawn-safety:** `noblock` starts a `multiprocessing.Process`. On macOS/Windows (spawn start
> method) the caller **must** live in an importable module and call it under
> `if __name__ == "__main__":`. Running it from a REPL, `python - <<EOF` heredoc, or notebook
> fails with `FileNotFoundError: .../<stdin>`. Linux (`fork`) is unaffected.

---

## 5. Public API map

| Symbol | Signature (defaults) | Returns |
| ------ | -------------------- | ------- |
| `VideoCapture` | `(file, codec=None, pix_fmt="bgr24", crop_xywh=None, resize=None, resize_keepratio=True, resize_keepratioalign="center", infile_options=None)` | `FFmpegReader` |
| `VideoCaptureNV` | `(file, pix_fmt="bgr24", crop_xywh=None, resize=None, resize_keepratio=True, resize_keepratioalign="center", infile_options=None, gpu=0)` | `FFmpegReaderNV` |
| `VideoCaptureQSV` | same as NV | `FFmpegReaderQSV` |
| `VideoCaptureCAM` | `(camname, pix_fmt="bgr24", crop_xywh=None, resize=None, resize_keepratio=True, resize_keepratioalign="center", camsize_wh=None, camfps=None, camcodec=None, campix_fmt=None)` | `FFmpegReaderCAM` |
| `VideoCaptureStream` | `(stream_url, codec=None, pix_fmt="bgr24", …, timeout=None)` | `FFmpegReaderStream` |
| `VideoCaptureStreamRT` | `(stream_url, codec=None, pix_fmt="bgr24", …, gpu=None, timeout=None)` | CPU RT reader, or NV reader when `gpu` is an int |
| `VideoCapturePannels` | `(file, crop_xywh_l, codec=None, pix_fmt="bgr24", resize=None)` | `FFmpegReaderPannels` |
| `VideoWriter` | `(file, codec=None, fps=30, pix_fmt="bgr24", bitrate=None, resize=None, preset=None)` | `FFmpegWriter` |
| `VideoWriterNV` | `(file, codec=None, fps=30, pix_fmt="bgr24", gpu=0, bitrate=None, resize=None, preset=None)` | `FFmpegWriterNV` |
| `VideoWriterQSV` | same as NV | `FFmpegWriterQSV` |
| `VideoWriterStreamRT` | `(url, pix_fmt="bgr24", bitrate=None, resize=None, preset=None)` | `FFmpegWriterStreamRT` |
| `noblock` | `(fun, *args, **kwargs)` | `FFmpegReaderNoblock` or `FFmpegWriterNoblock` |
| `ReadLiveLast` | `(fun, *args, **kwargs)` | thread + reader hybrid |
| `toCUDA` | `(vid, gpu=0, tensor_format="chw")` | `FFmpegReaderCUDA` |
| `get_num_NVIDIA_GPUs` | `()` | `int` |

Common methods on every reader: `read() -> (bool, ndarray|None)`, `isOpened() -> bool`,
`release()`, `close()`, `__iter__`, `__next__`, `__enter__`/`__exit__`.
`__len__` is inherited from `FFmpegReader`, so it works for file readers, `VideoCapturePannels`
and `VideoCaptureStreamRT` (the latter asserts the stream count is known). Camera readers
(`VideoCaptureCAM`, `VideoCaptureStream`) define no `__len__`.
Writers add `write(img)`, plus `__del__` (auto-release).

`VideoReader*` names are exact aliases (`VideoReader is VideoCapture`, etc.).
`get_info` / `get_info_precise` / `VideoInfo` live in `ffmpegcv.video_info`;
`get_info` / `StreamInfo` live in `ffmpegcv.stream_info`.

---

## 6. Repository layout

```
.
├── AGENTS.md                     # this file — agent entry point
├── CLAUDE.md                     # short pointer to AGENTS.md (Claude Code)
├── llms.txt                      # machine-readable doc index
├── README.md / README_CN.md      # human docs (EN / zh-CN)
├── setup.py                      # packaging; reads README.md as long_description
├── docs/
│   ├── README.md                 # documentation index
│   └── agents/                   # deep-dive references for agents
│       ├── quickstart.md
│       ├── api-reference.md
│       ├── architecture.md
│       ├── troubleshooting.md
│       └── testing.md
├── ffmpegcv/
│   ├── __init__.py               # public factories + import-time ffmpeg check
│   ├── version.py                # __version__
│   ├── video_info.py             # ffprobe helpers, codec maps, process helpers
│   ├── stream_info.py            # ffprobe helpers for live streams
│   ├── ffmpeg_reader.py          # FFmpegReader / FFmpegReaderNV + filter builders
│   ├── ffmpeg_writer.py          # FFmpegWriter / FFmpegWriterNV
│   ├── ffmpeg_reader_camera.py   # FFmpegReaderCAM, device/option queries
│   ├── ffmpeg_reader_stream.py   # FFmpegReaderStream (+ NV variant)
│   ├── ffmpeg_reader_stream_realtime.py  # low-latency RT readers
│   ├── ffmpeg_writer_stream_realtime.py  # RTMP/RTSP writer
│   ├── ffmpeg_reader_qsv.py / ffmpeg_writer_qsv.py
│   ├── ffmpeg_reader_pannels.py  # multi-ROI split reader
│   ├── ffmpeg_noblock.py         # noblock() + ReadLiveLast
│   ├── ffmpeg_reader_noblock.py / ffmpeg_writer_noblock.py
│   ├── ffmpeg_reader_cuda.py     # toCUDA / pycuda kernels
│   └── py.typed                  # PEP 561 marker
└── tests/
    └── compat_suite.py           # behavioral regression suite (JSON output)
```

`.github/workflows/` only publishes to PyPI on release; there is no CI test job.

---

## 7. Rules for agents editing this repo

- **Keep the OpenCV-compatible surface stable.** Renaming a public symbol or changing a
  default breaks users; add aliases instead.
- **No new hard dependencies.** Only `numpy` (and optional `pycuda`). Do not import
  `cv2`/`torch` at module scope (`torch` is imported lazily inside `read_torch`).
- **Stay Python 3.6-compatible** in `ffmpegcv/` (no walrus, no `X | Y` annotations at
  runtime). `from __future__ import annotations` is not currently used.
- **Quote paths in ffmpeg commands** (`"{filename}"`); commands are built as shell strings
  but executed with `shell=False` via `shlex.split` — spaces in paths must stay quoted.
- **Every code path that opens a process must release it** (`release_process`,
  `release_process_writer`) — including error branches.
- **Update [`tests/compat_suite.py`](tests/compat_suite.py) when behavior changes**; it is the
  project's de-facto contract. Keep its JSON keys stable.
- When changing ffmpeg filter strings, check both the CPU (`get_videofilter_cpu`) and GPU
  (`get_videofilter_gpu`) builders in [`ffmpegcv/ffmpeg_reader.py`](ffmpegcv/ffmpeg_reader.py).

## 8. Definition of done for a change

1. `python -m pip install -e .` still succeeds.
2. `PYTHONPATH=. python tests/compat_suite.py` reports `unexpected failures: 0`
   (GPU-absent expectations are part of the suite and must stay non-fatal).
3. Any new public symbol is exported in [`ffmpegcv/__init__.py`](ffmpegcv/__init__.py) and
   documented in [docs/agents/api-reference.md](docs/agents/api-reference.md).
4. README examples that you touched still run verbatim.
