# Testing & verification

The repository ships a single behavioral/regression script:
[`tests/compat_suite.py`](../../tests/compat_suite.py). It is not pytest-based and there is no
CI test job — this script **is** the contract.

## Run it

```bash
cd <repo root>
python -m pip install -e .              # or set PYTHONPATH
PYTHONPATH=. python tests/compat_suite.py --out /tmp/ffmpegcv_compat.json
```

Why `PYTHONPATH=.`? `tests/` has no `__init__.py`, so `python tests/compat_suite.py` puts
*`tests/`* on `sys.path` and cannot import the local `ffmpegcv` package.

Exit code is `0` when every expectation held, `1` otherwise. The last line is either
`unexpected failures: 0` (green) or a list of failing keys.

## What it does

1. Creates a deterministic source clip with
   `ffmpeg -f lavfi -i testsrc=size=320x240:rate=30 -t 1` (30 frames).
2. Runs ~60 checks across: metadata (`ffprobe`), every `pix_fmt`, ROI filters and all five
   alignments, context-manager/iterator protocol, writer codecs and options, `noblock`,
   `ReadLiveLast`, multi-panel reading, stream/RTSP command construction (with ffmpeg mocked),
   per-platform camera command construction (with `this_os` mocked), and GPU-unavailable paths.
3. Prints results as JSON between `===COMPAT_JSON_BEGIN===` / `===COMPAT_JSON_END===` and
   optionally writes the same payload to `--out`.

## Output schema

```jsonc
{
  "python": "3.13.2", "numpy": "2.2.3", "ffmpeg": "8.1.1", "ffmpegcv": "0.3.20",
  "results": {
    "cap_pixfmt_bgr24": { "ok": true, "value": { "n": 30, "shape": [240,320,3], "dtype": "uint8",
                                                 "hash0": "...", "len": 30, "isOpened_after": false } },
    "wr_bad_shape":     { "ok": true, "value": "AssertionError", "msg": "" }
  },
  "failed_expectations": []          // keys whose "ok" is false
}
```

Two helper semantics matter when reading/extending results:

- `run(key, fn)` — success means `fn()` returned normally; its return value is stored.
- `run_err(key, fn)` — success means `fn()` **raised**; the exception class name is stored in
  `"value"`. A `run_err` case is *green* on a machine without GPUs.

`failed_expectations` therefore contains only genuine surprises, and is empty on machines
with or without NVIDIA/QSV hardware.

## Determinism notes

- `hash0` / `hash_last` are **observations, not assertions**. They differ across ffmpeg
  versions (encoder defaults), so never hard-code them; compare within one environment only.
- Timing-, thread- and process-based checks (`noblock_*`, `readlivelast`, stream checks) assert
  counts/shapes, not exact frames.
- The suite mocks ffmpeg for camera/RTSP command-string checks so it never needs a real device.

## Adding a case

Inside an existing `test_*` function:

```python
def _my_case():
    with ffmpegcv.VideoCapture(src, pix_fmt="gray") as cap:
        ret, frame = cap.read()
        return {"ret": bool(ret), "shape": list(frame.shape)}

run("cap_my_case", _my_case)                 # expect success
run_err("cap_my_guard", lambda: ffmpegcv.VideoCapture("/nope.mp4"))   # expect an exception
```

Rules:

- Keep JSON keys and the top-level payload schema stable — downstream agents diff this output.
- Never let a case hang: file-based cases inherit a 1-second clip; stream/camera cases must not
  open real devices.
- Register a new `test_*` function in `main()`'s tuple if you add a whole section.
- Record failures through `run`/`run_err`; let `main()` collect them.

## Quick ad-hoc verification

```bash
# 1. import-time prerequisite
ffmpeg -version && ffprobe -version

# 2. smoke: write + read back one clip
PYTHONPATH=. python - <<'PY'
import numpy as np, ffmpegcv
with ffmpegcv.VideoWriter("/tmp/_smoke.mp4", None, 30) as out:
    for _ in range(10):
        out.write(np.zeros((240, 320, 3), np.uint8))
with ffmpegcv.VideoCapture("/tmp/_smoke.mp4") as cap:
    print("frames:", len(cap), "size:", cap.size, "fps:", cap.fps)
assert len(cap) == 10
PY
```

## Reporting results as an agent

When you finish a change, report: the exact command, `python`/`numpy`/`ffmpeg` versions, the
`failed_expectations` list, and any changed `results` keys. That is enough for a reviewer to
reproduce your run.
