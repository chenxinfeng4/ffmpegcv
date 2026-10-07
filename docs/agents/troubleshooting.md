# Troubleshooting

Symptom → cause → fix. Run `PYTHONPATH=. python tests/compat_suite.py` to see which of these
the local machine trips; GPU-absent failures are *expected* and part of the suite.

## Import / install

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `RuntimeError: The ffmpeg is not installed.` | `ffmpeg` or `ffprobe` not on `PATH`; `import ffmpegcv` runs this check | Install ffmpeg (`conda install ffmpeg`, `brew install ffmpeg`, `apt install ffmpeg`) and verify `ffmpeg -version && ffprobe -version`. |
| `ModuleNotFoundError: No module named 'ffmpegcv'` when running `python tests/compat_suite.py` | Running a script inside `tests/` puts `tests/` on `sys.path`, not the repo root | Run with `PYTHONPATH=.` from the repo root, or `pip install -e .`. |
| `ImportError: No module named 'pycuda'` / `cuda.init()` fails | `toCUDA` needs `pycuda`; `ffmpeg_reader_cuda` imports it at module import | `pip install ffmpegcv[cuda]` (or `pip install pycuda`). Do not call `toCUDA` on CPU-only machines. |
| `torch` import error | Not a dependency | `torch` is only imported inside `read_torch`; install torch if you use it, otherwise use `read()`/`read_cudamem()`. |

## GPU paths

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `RuntimeError: The ffmpeg is not compiled with NVENC support.` | ffmpeg build lacks nvenc/cuvid | Use the CPU classes, or install an ffmpeg built with NVIDIA support. |
| `RuntimeError: No NVIDIA GPU found.` | `get_num_NVIDIA_GPUs() == 0` | No NVIDIA GPU present; use CPU. |
| `AssertionError: No GPU found` | `FFmpegReaderNV.VideoReader` / `FFmpegReaderQSV.VideoReader` GPU guard | Use `VideoCapture`, or provide the hardware. |
| `AssertionError` from `VideoWriterNV` (`assert numGPU`) | NV encoder unavailable | Use `VideoWriter`, or a valid `h264_nvenc`/`hevc_nvenc` build. |
| `AssertionError: Cannot use multiple QSV gpu yet.` | QSV only supports device 0 | Pass `gpu=0`/omit. |
| QSV reader "works" on a machine without Intel GPU | Older ffmpeg builds printed option help even without a device | Current code rejects the `"not recognized"` message; update ffmpeg if you see false positives. |
| `Exception: No NV codec found for <name>` | Codec not in the map | Use a supported short name (`h264`, `hevc`, `mjpeg`, `mpeg2video`, `vp8`, `vp9`, `av1`, …) or the mapped name. |

## Reading & shapes

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `ValueError: assignment destination is read-only` | Frames are read-only views over the pipe buffer | `frame = frame.copy()` (or `np.array(frame)`) before mutating. |
| Frame shape unexpected | `pix_fmt` determines layout | `rgb24`/`bgr24`: `(h,w,3)`; `gray`: `(h,w,1)`; `yuv420p`/`nv12`: `(h*3//2, w)`. |
| `AssertionError: 'resize' must be even number` / `"resize must be a tuple of (width, height)"` | Odd or malformed size | Use two even ints, e.g. `resize=(640, 480)`. |
| `Warning 'crop_xywh' would be replaced into even numbers` | CPU path auto-floors odd crop values | Pass even `x, y, w, h`. |
| `AssertionError` in `assert all(n % 2 == 0 ...)` | NV reader requires even crop | Same as above. |
| Image looks stretched or has black bars unexpectedly | `resize_keepratio` defaults to **True** | Pass `resize_keepratio=False` to stretch; or pick one of the five `resize_keepratioalign` values. |
| `AssertionError: <path> not exists` | `VideoCapture` asserts the file exists | Check the path; streams/cameras are not opened via `VideoCapture`. |
| `AssertionError: height must be even` / `width must be even` (NV) | cuvid constraint | Crop/resize to even dimensions. |
| `TypeError: object of type 'FFmpegReaderCAM' has no len()` | Camera readers define no `__len__` | Track `iframe` yourself; camera readers also have no `fps`/`count`/`duration`/`codec`. |
| `AttributeError: 'FFmpegReaderCAM' object has no attribute 'fps'` | Camera factory never sets `fps` | Use the requested `camfps`, or read `camfps`. |
| `len(cap)` raises `AssertionError: The frame count is unknown for streams.` | `VideoCaptureStreamRT` (an `FFmpegReader`) has `count=None` | Do not rely on `len` for live sources. |
| `read()` returns `(False, None)` immediately | EOF, bad codec, stream unreachable, or ffmpeg failed | Reproduce with direct ffmpeg; errors are hidden by design (see below). |

### Seeing ffmpeg's error output

Readers run with `-loglevel error`/`warning` and send stderr to `DEVNULL`
(`run_async_reader`). To debug, take the failing object's `cap.ffmpeg_cmd` and run it by hand:

```python
print(cap.ffmpeg_cmd)      # copy/paste into a shell (it is already quoted for a shell)
```

## Writing

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `AssertionError` on the 2nd `write()` | Frame shape differs from the first frame | Resize/convert to the exact first-frame shape, or pass `resize=` to the writer. |
| `UserWarning: Codec should be a string ...` | Passed a `cv2.VideoWriter_fourcc` int | Pass a string codec such as `"h264"`. |
| `TypeError`/no output file, empty file | `release()` never called, process not flushed | Always `with` or call `release()`; `__del__` is a safety net, not a guarantee. |
| Video is too large / poor quality | No bitrate/preset control by default | Pass `bitrate="4M"` and `preset="medium"`/`"slow"`. |
| `AssertionError` in `VideoWriterStreamRT` | Codec/pix_fmt outside the whitelist | Use `h264`/`libx264`/`x264`/`mpeg4` and `bgr24`/`rgb24`/`gray`. |
| `Preset is auto configured in FFmpegWriterStreamRT` printed | `preset` is ignored by design | Don't pass `preset`. |

## Streams & cameras

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| Process never exits; program hangs after use | Camera/`StreamRT`/`ReadLiveLast`/`noblock` keep background workers alive | Always `release()` (use `try/finally`), then join child processes. |
| Frames lag behind by seconds | `VideoCaptureCAM`/`VideoCaptureStream` buffer 30 frames and take the newest only when full | Prefer `VideoCaptureStreamRT` + `ReadLiveLast` for lowest latency. |
| `query_camera_options` prints "CAN NOT query" and returns a stub | macOS `avfoundation` cannot enumerate options; Linux `v4l2` cannot report fps | Pass `camsize_wh`, `camfps`, `campix_fmt` explicitly (macOS requires all three). |
| No camera found on macOS | Index vs name mapping differs | `query_camera_devices()` returns `{name: (name, index)}` on mac; the reader accepts name or int. |
| `ValueError: The function is not supported as a Reader or Writer` | `noblock` given an unsupported factory | Only `VideoCapture`, `VideoCaptureNV`, `VideoWriter`, `VideoWriterNV` are supported. |
| `FileNotFoundError: .../<stdin>` from a multiprocessing child | macOS/Windows spawn re-imports `__main__`; the caller was a REPL/heredoc/notebook | Put the `noblock` code in a real module behind `if __name__ == "__main__":`. |
| RTSP stream stalls | UDP transport issues | Readers already force TCP/`prefer_tcp`; if it still stalls, test `ffplay -rtsp_transport tcp <url>`. |

## Behavior changes (version notes)

- `VideoCapture(..., codec=...)` is deprecated and emits `DeprecationWarning`; the codec is
  detected from the file. Remove the argument.
- The QSV probe was changed to reject `"Codec ... is not recognized by FFmpeg."` from ffmpeg 6+.
- `mkv`/`flv`/`ts` frame counts require a full scan and are therefore slow by design.
