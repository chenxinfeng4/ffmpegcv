import numpy as np
import warnings
import pprint
import select
import sys
from types import TracebackType
from typing import Any, Optional, Sequence, Tuple, Type

from .video_info import run_async, release_process_writer, get_num_NVIDIA_GPUs


IN_COLAB = "google.colab" in sys.modules


class FFmpegWriter:
    # Attributes initialized by the `VideoWriter` factory methods.
    fps: float
    codec: str
    pix_fmt: str
    filename: str
    bitrate: Optional[str]
    resize: Optional[Sequence[int]]
    preset: Optional[str]
    in_numpy_shape: Sequence[int]

    def __init__(self) -> None:
        self.iframe: int = -1
        self.size: Optional[Tuple[int, int]] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.waitInit: bool = True
        self._isopen: bool = True
        self.process: Any = None

    def __enter__(self) -> "FFmpegWriter":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()

    def __repr__(self) -> str:
        props = pprint.pformat(self.__dict__).replace("{", " ").replace("}", " ")
        return f"{self.__class__}\n" + props

    @staticmethod
    def VideoWriter(
        filename: str,
        codec: Optional[str],
        fps: float,
        pix_fmt: str,
        bitrate: Optional[str] = None,
        resize: Optional[Sequence[int]] = None,
        preset: Optional[str] = None
    ) -> "FFmpegWriter":
        if codec is None:
            codec = "h264"
        elif not isinstance(codec, str):
            codec = "h264"
            warnings.warn(
                "Codec should be a string. Eg `h264`, `h264_nvenc`. "
                "You may used CV2.VideoWriter_fourcc, which will be ignored.",
                UserWarning,
            )
        assert resize is None or len(resize) == 2

        vid = FFmpegWriter()
        vid.fps = fps
        vid.codec, vid.pix_fmt, vid.filename = codec, pix_fmt, filename
        vid.bitrate = bitrate
        vid.resize = resize
        vid.preset = preset
        return vid

    def _init_video_stream(self) -> None:
        assert self.resize is not None
        bitrate_str = f"-b:v {self.bitrate} " if self.bitrate else ""
        rtsp_str = f"-f rtsp" if self.filename.startswith("rtsp://") else ""
        filter_str = (
            ""
            if self.resize == self.size
            else f"-vf scale={self.resize[0]}:{self.resize[1]}"
        )
        target_pix_fmt = getattr(self, "target_pix_fmt", "yuv420p")
        preset_str = f"-preset {self.preset} " if self.preset else ""

        self.ffmpeg_cmd = (
            f"ffmpeg -y -loglevel error "
            f"-f rawvideo -pix_fmt {self.pix_fmt} -s {self.width}x{self.height} -r {self.fps} -i pipe: "
            f"{bitrate_str} "
            f"-r {self.fps} -c:v {self.codec} "
            f"{preset_str}"
            f"{filter_str} {rtsp_str} "
            f'-pix_fmt {target_pix_fmt} "{self.filename}"'
        )
        self.process = run_async(self.ffmpeg_cmd)

    def write(self, img: np.ndarray) -> None:
        if self.waitInit:
            if self.pix_fmt in ("nv12", "yuv420p", "yuvj420p"):
                height_15, width = img.shape[:2]
                assert width % 2 == 0 and height_15 * 2 % 3 == 0
                height = int(height_15 / 1.5)
            else:
                height, width = img.shape[:2]
            self.width, self.height = width, height
            self.in_numpy_shape = img.shape
            self.size = (width, height)
            self.resize = self.size if self.resize is None else tuple(self.resize)
            self._init_video_stream()
            self.waitInit = False

        self.iframe += 1
        assert self.in_numpy_shape == img.shape
        img = img.astype(np.uint8).tobytes()
        self.process.stdin.write(img)

        stderrreadable, _, _ = select.select([self.process.stderr], [], [], 0)
        if stderrreadable:
            data = self.process.stderr.read(1024)
            sys.stderr.buffer.write(data)

    def isOpened(self) -> bool:
        return self._isopen

    def release(self) -> None:
        self._isopen = False
        if hasattr(self, "process"):
            release_process_writer(self.process)

    def close(self) -> None:
        return self.release()


class FFmpegWriterNV(FFmpegWriter):
    gpu: int

    @staticmethod
    def VideoWriter(  # type: ignore[override]
        filename: str,
        codec: Optional[str],
        fps: float,
        pix_fmt: str,
        gpu: Optional[int],
        bitrate: Optional[str] = None,
        resize: Optional[Sequence[int]] = None,
        preset: Optional[str] = None
    ) -> "FFmpegWriterNV":
        numGPU = get_num_NVIDIA_GPUs()
        assert numGPU
        gpu = int(gpu) % numGPU if gpu is not None else 0
        if codec is None:
            codec = "hevc_nvenc"
        elif not isinstance(codec, str):
            codec = "hevc_nvenc"
            warnings.warn(
                "Codec should be a string. Eg `h264`, `h264_nvenc`. "
                "You may used CV2.VideoWriter_fourcc, which will be ignored.",
                UserWarning,
            )
        elif codec.endswith("_nvenc"):
            codec = codec
        else:
            codec = codec + "_nvenc"
        assert codec in [
            "hevc_nvenc",
            "h264_nvenc",
        ], "codec should be `hevc_nvenc` or `h264_nvenc`"
        assert resize is None or len(resize) == 2

        vid = FFmpegWriterNV()
        vid.fps = fps
        vid.codec, vid.pix_fmt, vid.filename = codec, pix_fmt, filename
        vid.gpu = gpu
        vid.bitrate = bitrate
        vid.resize = resize
        vid.preset = preset if preset is not None else ("default" if IN_COLAB else "fast")
        return vid

    def _init_video_stream(self) -> None:
        assert self.resize is not None
        bitrate_str = f"-b:v {self.bitrate} " if self.bitrate else ""
        rtsp_str = f"-f rtsp" if self.filename.startswith("rtsp://") else ""
        filter_str = (
            ""
            if self.resize == self.size
            else f"-vf scale={self.resize[0]}:{self.resize[1]}"
        )
        self.ffmpeg_cmd = (
            f"ffmpeg -y -loglevel error "
            f"-f rawvideo -pix_fmt {self.pix_fmt} -s {self.width}x{self.height} -r {self.fps} -i pipe: "
            f"-preset {self.preset} {bitrate_str} "
            f"-r {self.fps} -gpu {self.gpu} -c:v {self.codec} "
            f"{filter_str} {rtsp_str} "
            f'-pix_fmt yuv420p "{self.filename}"'
        )
        self.process = run_async(self.ffmpeg_cmd)
