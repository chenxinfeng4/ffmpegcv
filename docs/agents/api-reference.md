# API reference

Exact signatures as exported by [`ffmpegcv/__init__.py`](../../ffmpegcv/__init__.py).
`->` shows the concrete class returned. All `ffmpegcv.*` symbols are importable directly.

Reader options shared by every capture factory:

| Option | Type | Default | Meaning |
| ------ | ---- | ------- | ------- |
| `pix_fmt` | `str` | `"bgr24"` | Output pixel format — also the numpy layout. One of `bgr24`, `rgb24`, `gray`, `yuv420p`, `yuvj420p`, `nv12` (availability varies per reader). |
| `crop_xywh` | `(x, y, w, h)` | `None` | Crop before any resize. |
| `resize` | `(w, h)` | `None` | Output size; must be even. |
| `resize_keepratio` | `bool` | `True` | Letterbox instead of stretch. |
| `resize_keepratioalign` | `str` | `"center"` | `center`, `topleft`, `topright`, `bottomleft`, `bottomright`. |
| `infile_options` | `str` | `None` | Extra flags injected before `-i` (e.g. `"-re -stream_loop -1"`). |

Default `resize_keepratio=True` is a notable difference from cv2: `resize=(640,480)` alone
letterboxes. Pass `resize_keepratio=False` to stretch.

---

## Capture factories

### `VideoCapture`

```python
ffmpegcv.VideoCapture(
    file: str,
    codec: Optional[str] = None,          # DEPRECATED — ffprobe detects it; emits DeprecationWarning
    pix_fmt: str = "bgr24",
    crop_xywh: Optional[Sequence[int]] = None,
    resize: Optional[Sequence[int]] = None,
    resize_keepratio: bool = True,
    resize_keepratioalign: Optional[str] = "center",
    infile_options: Optional[str] = None,
) -> FFmpegReader
```

Alias: `ffmpegcv.VideoReader`. Asserts the file exists. Metadata comes from `ffprobe`
(`video_info.get_info`).

### `VideoCaptureNV`

```python
ffmpegcv.VideoCaptureNV(
    file, pix_fmt="bgr24", crop_xywh=None, resize=None,
    resize_keepratio=True, resize_keepratioalign="center",
    infile_options=None, gpu=0,
) -> FFmpegReaderNV
```

Alias: `VideoReaderNV`. Calls `_check_nvidia()` (ffmpeg NVENC/NVDEC + `get_num_NVIDIA_GPUs() > 0`),
then decodes through `<codec>_cuvid` + `-hwaccel cuda -hwaccel_device {gpu}`. Requires even
source dimensions. `gpu` is taken modulo the GPU count.

### `VideoCaptureQSV`

```python
ffmpegcv.VideoCaptureQSV(
    file, pix_fmt="bgr24", crop_xywh=None, resize=None,
    resize_keepratio=True, resize_keepratioalign="center",
    infile_options=None, gpu=0,
) -> FFmpegReaderQSV
```

Alias: `VideoReaderQSV`. Experimental. Only one QSV device is supported (`assert gpu in (None, 0)`),
and the module docstring notes ROI is not implemented.

### `VideoCaptureCAM`

```python
ffmpegcv.VideoCaptureCAM(
    camname: Union[int, str],
    pix_fmt: str = "bgr24",
    crop_xywh=None, resize=None, resize_keepratio=True, resize_keepratioalign="center",
    camsize_wh=None, camfps=None, camcodec=None, campix_fmt=None,
) -> FFmpegReaderCAM
```

Implement as: `camname` is an index, an exact device name, or a platform device
path (`@device_pnp_...` on Windows). Uses `dshow` (Windows), `v4l2` (Linux), `avfoundation`
(macOS). Implemented with a `ProducerThread` + `queue.Queue(maxsize=30)` and drops frames when
the consumer falls behind. Helpers in `ffmpegcv.ffmpeg_reader_camera`:
`query_camera_devices(verbose_dict=False)` and `query_camera_options(cam_id_name)`.

Not set on this reader (accessing them raises `AttributeError`): `fps`, `count`, `duration`,
`codec`, and there is no `__len__`. Available: `width`, `height`, `size`, `origin_width`,
`origin_height`, `crop_width`, `crop_height`, `camfps`, `camcodec`, `campix_fmt`, `camname`,
`camid`, `pix_fmt`, `out_numpy_shape`, `ffmpeg_cmd`, `iframe`.
`FFmpegReaderStream` (below) extends this class and *does* set `fps`/`count`/`duration`/`codec`.

### `VideoCaptureStream`

```python
ffmpegcv.VideoCaptureStream(
    stream_url, codec=None, pix_fmt="bgr24", crop_xywh=None, resize=None,
    resize_keepratio=True, resize_keepratioalign="center",
    infile_options=None, timeout=None,
) -> FFmpegReaderStream
```

Alias: `VideoReaderStream`. RTSP/RTP/RTMP/HTTP(S). `timeout` bounds the ffprobe call.
RTSP URLs get `-rtsp_flags prefer_tcp -pkt_size 736`. Uses a producer thread like the camera.

### `VideoCaptureStreamRT`

```python
ffmpegcv.VideoCaptureStreamRT(
    stream_url, codec=None, pix_fmt="bgr24", crop_xywh=None, resize=None,
    resize_keepratio=True, resize_keepratioalign="center",
    infile_options=None, gpu=None, timeout=None,
) -> Union[FFmpegReaderStreamRT, FFmpegReaderStreamRTNV]
```

Alias: `VideoReaderStreamRT`. `gpu=None` → CPU reader; any int → NVIDIA reader.
Adds `-fflags nobuffer -flags low_delay -strict experimental`. RTSP uses `rtsp_transport tcp`.
No producer thread: `read()` pulls directly from the pipe.

### `VideoCapturePannels`

```python
ffmpegcv.VideoCapturePannels(
    file: str,
    crop_xywh_l: List[Sequence[int]],
    codec=None,
    pix_fmt="bgr24",
    resize=None,
) -> FFmpegReaderPannels
```

Alias: `VideoReaderPannels`. One ffmpeg process splits the frame with
`filter_complex "split=N[...];[VSRC0]crop=..."` and emits all panels on stdout.
`read()` returns:
- `np.ndarray` `(N, h, w, c)` when all panels share a size;
- otherwise a **list** of arrays with individual shapes.

Missing from this reader: `resize_keepratio*` (resize does not preserve aspect). It still
inherits `__len__` from `FFmpegReader`, so `len(cap)` returns the probed frame count.

---

## Writer factories

### `VideoWriter`

```python
ffmpegcv.VideoWriter(
    file: str,
    codec: Optional[str] = None,     # None -> "h264"
    fps: float = 30,
    pix_fmt: str = "bgr24",
    bitrate: Optional[str] = None,   # e.g. "1M"
    resize: Optional[Sequence[int]] = None,
    preset: Optional[str] = None,    # e.g. "ultrafast"
) -> FFmpegWriter
```

A cv2 fourcc int is ignored with a `UserWarning`. Output pix_fmt is `yuv420p` unless the
subclass overrides `target_pix_fmt`. RTSP targets (`rtsp://`) add `-f rtsp`.

### `VideoWriterNV`

```python
ffmpegcv.VideoWriterNV(
    file, codec=None, fps=30, pix_fmt="bgr24", gpu=0,
    bitrate=None, resize=None, preset=None,
) -> FFmpegWriterNV
```

`codec=None` → `hevc_nvenc`; `"h264"` → `"h264_nvenc"`; only `h264_nvenc`/`hevc_nvenc` are
allowed. Default preset is `"fast"` (`"default"` under Google Colab).

### `VideoWriterQSV`

```python
ffmpegcv.VideoWriterQSV(file, codec=None, fps=30, pix_fmt="bgr24", gpu=0,
                        bitrate=None, resize=None, preset=None) -> FFmpegWriterQSV
```

`codec=None` → `hevc_qsv`; strings are mapped through `decoder_to_qsv`. Single device only.

### `VideoWriterStreamRT`

```python
ffmpegcv.VideoWriterStreamRT(
    url, pix_fmt="bgr24", bitrate=None, resize=None, preset=None
) -> FFmpegWriterStreamRT
```

Always `libx264`; asserts `pix_fmt in {bgr24, rgb24, gray}`. Forces `-f flv`,
`-tune zerolatency`, `-preset ultrafast`, `-g 50`, and `-f rtsp` when the URL starts with
`rtsp://`. Passing `preset` prints a notice and is ignored.

---

## Utilities

### `noblock`

```python
ffmpegcv.noblock(fun, *args, **kwargs) -> FFmpegReaderNoblock | FFmpegWriterNoblock
```

`fun` must be one of `VideoCapture`, `VideoCaptureNV`, `VideoWriter`, `VideoWriterNV`,
otherwise `ValueError`. Uses `multiprocessing.Process` + shared `Array` ring buffer
(`NFRAME = 10`, queue depth `8`). Reader `read()` blocks until the next prefetched frame;
`release()` joins the child.

### `ReadLiveLast`

```python
ffmpegcv.ReadLiveLast(fun, *args, **kwargs)
```

A `threading.Thread` subclass that also subclasses `FFmpegReader`. It continuously reads and
keeps only the latest frame (`queue.Queue(maxsize=1)`), so `read()` returns fresh data with no
backlog. Best combined with `VideoCaptureStreamRT`. Always `release()`.

### `toCUDA`

```python
ffmpegcv.toCUDA(vid: FFmpegReader, gpu: int = 0, tensor_format: str = "chw") -> FFmpegReaderCUDA
```

Wraps an existing reader whose `pix_fmt` is `yuv420p` or `nv12`. A CUDA kernel converts
YUV → RGB float32 directly on the device. Output shape is `(3, h, w)` for `"chw"` and
`(h, w, 3)` for `"hwc"`. Read methods:

| Method | Output |
| ------ | ------ |
| `read()` | `pycuda.gpuarray.GPUArray` (float32) |
| `read_cudamem()` | `pycuda.driver.DeviceAllocation` (raw) |
| `read_torch(out=None)` | `torch.Tensor` on `cuda:{gpu}`; pass `out` to reuse memory |

`torch` is imported lazily inside `read_torch`, so it is not a package dependency.
`read_*` also accept an explicit output buffer as the first argument.

### Metadata helpers (`ffmpegcv.video_info`)

```python
class VideoInfo(NamedTuple):
    width: int; height: int; fps: float; count: int; codec: str; duration: float

get_info(video: str) -> VideoInfo            # fast ffprobe; scans whole file for mkv/flv/ts
get_info_precise(video: str) -> VideoInfo    # reads first/last pts for an exact fps
get_num_NVIDIA_GPUs() -> int                  # exported at package top level too
get_num_QSV_GPUs() -> int
encoder_to_nvidia(codec) / decoder_to_nvidia(codec) / encoder_to_qsv(codec) / decoder_to_qsv(codec)
```

Codec mappers accept either the short name (`"h264"`) or an already-mapped name
(`"h264_nvenc"`), and raise `Exception("No NV codec found for ...")` otherwise.

### Stream metadata (`ffmpegcv.stream_info`)

```python
class StreamInfo(NamedTuple):
    width: int; height: int; fps: float; count: Optional[int]; codec: str; duration: Optional[float]

get_info(stream_url, timeout=None, duration_ms=100) -> StreamInfo
```

`count` and `duration` are always `None` for live streams; this is why stream readers have no
usable `__len__`.

---

## Reader / writer object protocol

`FFmpegReader` (and subclasses used for files):

```python
cap.iframe          # int, last frame index (-1 before first read)
cap.width, cap.height, cap.size          # output size; size == (width, height)
cap.origin_width, cap.origin_height      # pre-filter size
cap.crop_width, cap.crop_height
cap.fps, cap.count, cap.duration, cap.codec
cap.pix_fmt, cap.out_numpy_shape, cap.filename, cap.ffmpeg_cmd
cap.debug           # flag; ffmpeg stderr is otherwise redirected to DEVNULL
cap.read() -> (bool, np.ndarray | None)
cap.isOpened() -> bool
cap.release(); cap.close()
with cap: ...       # __enter__/__exit__
for frame in cap: ...  # __iter__/__next__
len(cap)            # -> count; asserts if count is None
```

`FFmpegWriter`:

```python
out.write(img)      # first call fixes size/shape and starts ffmpeg
out.isOpened(); out.release(); out.close()
with out: ...
# __del__ calls release()
```

Frames returned by readers are **non-writable views** over the raw pipe buffer:

```python
ret, frame = cap.read()
frame.flags.writeable   # False
frame[0, 0, 0] = 0      # ValueError: assignment destination is read-only
```
