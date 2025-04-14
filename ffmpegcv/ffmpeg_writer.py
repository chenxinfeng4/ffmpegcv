import numpy as np
import warnings
import pprint
import select
import sys
from .video_info import run_async, release_process_writer, get_num_NVIDIA_GPUs


IN_COLAB = "google.colab" in sys.modules


class FFmpegWriter:
    def __init__(self):
        self.iframe = -1
        self.size = None
        self.width, self.height = None, None
        self.waitInit = True
        self._isopen = True

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.release()

    def __del__(self):
        self.release()

    def __repr__(self):
        props = pprint.pformat(self.__dict__).replace("{", " ").replace("}", " ")
        return f"{self.__class__}\n" + props

    @staticmethod
    def VideoWriter(
        filename, codec, fps, pix_fmt, bitrate=None, resize=None, preset=None
    ):
        if codec is None:
            codec = "h264"
        elif not isinstance(codec, str):
            codec = "h264"
            warnings.simplefilter(
                """
                Codec should be a string. Eg `h264`, `h264_nvenc`. 
                You may used CV2.VideoWriter_fourcc, which will be ignored.
                """
            )
        assert resize is None or len(resize) == 2

        vid = FFmpegWriter()
        vid.fps = fps
        vid.codec, vid.pix_fmt, vid.filename = codec, pix_fmt, filename
        vid.bitrate = bitrate
        vid.resize = resize
        vid.preset = preset
        return vid

    def _init_video_stream(self):
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

    def write(self, img: np.ndarray):
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

    def isOpened(self):
        return self._isopen

    def release(self):
        self._isopen = False
        if hasattr(self, "process"):
            release_process_writer(self.process)

    def close(self):
        return self.release()


class FFmpegWriterNV(FFmpegWriter):
    @staticmethod
    def VideoWriter(
        filename, codec, fps, pix_fmt, gpu, bitrate=None, resize=None, preset=None
    ):
        numGPU = get_num_NVIDIA_GPUs()
        assert numGPU
        gpu = int(gpu) % numGPU if gpu is not None else 0
        if codec is None:
            codec = "hevc_nvenc"
        elif not isinstance(codec, str):
            codec = "hevc_nvenc"
            warnings.simplefilter(
                """
                Codec should be a string. Eg `h264`, `h264_nvenc`. 
                You may used CV2.VideoWriter_fourcc, which will be ignored.
                """
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
        vid.preset = preset if preset is not None else ("default" if IN_COLAB else "p2")
        return vid

    def _init_video_stream(self):
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
