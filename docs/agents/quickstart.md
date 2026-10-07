# Quickstart for agents

Copy-paste recipes that are known to work with this repository (verified on
CPython 3.13 + ffmpeg 8.1 on macOS). For the rules and gotchas read
[AGENTS.md](../../AGENTS.md) first; for full signatures see
[api-reference.md](api-reference.md).

All snippets assume:

```python
import numpy as np
import ffmpegcv
```

## 0. Sanity check before writing any code

```bash
ffmpeg -version && ffprobe -version     # required at import time
PYTHONPATH=. python tests/compat_suite.py | tail -3
```

## 1. Read a file to RGB frames (deep-learning style)

```python
with ffmpegcv.VideoCapture("in.mp4", pix_fmt="rgb24") as cap:
    fps, n, (w, h) = cap.fps, len(cap), cap.size
    frames = [f for f in cap]          # each f: uint8 (h, w, 3)
```

## 2. Read only a crop and downscale (cheap on CPU/GPU)

```python
with ffmpegcv.VideoCapture("in.mp4",
                           crop_xywh=(320, 180, 640, 480),   # x, y, w, h
                           resize=(320, 240),
                           resize_keepratio=True) as cap:     # letterbox, align="center"
    for frame in cap:
        ...
```

## 3. Stream a file frame-by-frame without loading it all

```python
cap = ffmpegcv.VideoCapture("in.mp4")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        process(frame.copy())      # copy if you need to mutate / keep it
finally:
    cap.release()
```

## 4. Encode a frame generator to a video

```python
def frames():
    for _ in range(300):
        yield np.zeros((240, 320, 3), np.uint8)

with ffmpegcv.VideoWriter("out.mp4", None, 30) as out:   # h264
    for frame in frames():
        out.write(frame)
```

Real-time-ish webcam/video saving with a fixed size up front:

```python
out = ffmpegcv.VideoWriter("out.mp4", "h264", 30, resize=(640, 480))
for frame in frames_at_640x480:
    out.write(frame)      # every frame must already be 480x640x3
out.release()
```

## 5. Lossless-ish copy

```python
with ffmpegcv.VideoCapture("in.mkv") as cap, \
     ffmpegcv.VideoWriter("out.mp4", "h264", cap.fps, preset="ultrafast") as out:
    for frame in cap:
        out.write(frame)
```

## 6. NVIDIA acceleration

```python
with ffmpegcv.VideoCaptureNV("in.mp4", pix_fmt="nv12", gpu=0) as cap, \
     ffmpegcv.VideoWriterNV("out.mp4", "h264", cap.fps, gpu=0) as out:
    for frame in cap:
        out.write(frame)
```

If this raises `RuntimeError` ("not compiled with NVENC") or `AssertionError` ("No GPU
found"), the machine cannot do NV — fall back to the CPU classes.

## 7. Zero-copy frames into a torch model

```python
import torch
cap = ffmpegcv.toCUDA(ffmpegcv.VideoCaptureNV("in.mp4", pix_fmt="nv12"),
                      tensor_format="chw")
buf = torch.empty((3, cap.height, cap.width), dtype=torch.float32, device="cuda:0")
while True:
    ret, buf = cap.read_torch(buf)     # reuse memory, no allocation per frame
    if not ret:
        break
    out = model(buf.unsqueeze(0))
cap.release()
```

## 8. List cameras and open one

```python
from ffmpegcv.ffmpeg_reader_camera import query_camera_devices, query_camera_options

print(query_camera_devices())              # {0: (name, id_or_path), ...}
opts = query_camera_options(0)             # query_camera_options may print warnings on Linux/macOS
cap = ffmpegcv.VideoCaptureCAM(0, **opts[-1])
try:
    for _ in range(10):
        ret, frame = cap.read()
finally:
    cap.release()
```

## 9. Low-latency RTSP

```python
cap = ffmpegcv.ReadLiveLast(ffmpegcv.VideoCaptureStreamRT,
                            "rtsp://user:pass@host:554/Streaming/Channels/101")
try:
    for _ in range(100):
        ret, frame = cap.read()      # always the newest frame, no backlog
finally:
    cap.release()
```

## 10. Overlap I/O with model inference

```python
def main():
    cap = ffmpegcv.noblock(ffmpegcv.VideoCapture, "in.mp4", pix_fmt="rgb24")
    try:
        while True:
            ret, frame = cap.read()      # prefetched in a child process
            if not ret:
                break
            result = model(frame)
    finally:
        cap.release()

if __name__ == "__main__":               # required on macOS/Windows (spawn)
    main()
```

`noblock` uses `multiprocessing`, so on macOS/Windows the entry point must be an importable
module guarded by `if __name__ == "__main__":`. Calling it from a REPL or a `python - <<EOF`
heredoc fails with `FileNotFoundError: .../<stdin>`.

## 11. Probe metadata without opening a stream

```python
from ffmpegcv.video_info import get_info, get_info_precise

vi = get_info("in.mp4")
print(vi.width, vi.height, vi.fps, vi.count, vi.codec, vi.duration)

# exact frame count for containers that don't report nb_frames (mkv/flv/ts)
vi2 = get_info_precise("in.mkv")
```

## 12. Convert to images / a numpy stack

```python
with ffmpegcv.VideoCapture("in.mp4", pix_fmt="rgb24") as cap:
    stack = np.stack([f for f in cap])       # (N, h, w, 3) — memory heavy!

# write PNGs later with your own tooling (PIL, imageio, ...); ffmpegcv does not do images.
```
