import numpy as np
import warnings
import pprint
import sys
from .video_info import run_async, release_process, get_num_NVIDIA_GPUs


IN_COLAB = "google.colab" in sys.modules


class FFmpegWriter:
    def __init__(self):
        self.iframe = -1
        self.size = None
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
    def VideoWriter(filename, codec, fps, frameSize, pix_fmt, bitrate=None):
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

        vid = FFmpegWriter()
        vid.fps, vid.size = fps, frameSize
        vid.width, vid.height = vid.size if vid.size else (None, None)
        vid.codec, vid.pix_fmt, vid.filename = codec, pix_fmt, filename
        vid.bitrate = bitrate
        return vid

    def _init_video_stream(self):
        bitrate_str = f'-b:v {self.bitrate} ' if self.bitrate else ''
        self.ffmpeg_cmd = (f'ffmpeg -y -loglevel warning ' 
                f'-f rawvideo -pix_fmt {self.pix_fmt} -s {self.width}x{self.height} -r {self.fps} -i pipe: '
                f'{bitrate_str} '
                f'-r {self.fps} -c:v {self.codec} -pix_fmt yuv420p "{self.filename}"')
        self.process = run_async(self.ffmpeg_cmd)

    def write(self, img:np.ndarray):
        if self.waitInit:
            if self.size is None:
                self.size = (img.shape[1], img.shape[0])
            self.width, self.height = self.size
            self._init_video_stream()
            self.waitInit = False

        self.iframe += 1
        assert self.size == (img.shape[1], img.shape[0])
        img = img.astype(np.uint8).tobytes()
        self.process.stdin.write(img)

    def isOpened(self):
        return self._isopen

    def release(self):
        self._isopen = False
        if hasattr(self, "process"):
            release_process(self.process)

    def close(self):
        return self.release()


class FFmpegWriterNV(FFmpegWriter):
    @staticmethod
    def VideoWriter(filename, codec, fps, frameSize, pix_fmt, gpu, bitrate=None):
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

        vid = FFmpegWriterNV()
        vid.fps, vid.size = fps, frameSize
        vid.width, vid.height = vid.size if vid.size else (None, None)
        vid.codec, vid.pix_fmt, vid.filename = codec, pix_fmt, filename
        vid.gpu = gpu
        vid.waitInit = True
        vid.bitrate = bitrate
        return vid

    def _init_video_stream(self):
        bitrate_str = f'-b:v {self.bitrate} ' if self.bitrate else ''
        default_preset = 'default' if IN_COLAB else 'p2'
        self.preset = getattr(self, 'preset', default_preset)
        self.ffmpeg_cmd = (f'ffmpeg -y -loglevel warning '
            f'-f rawvideo -pix_fmt {self.pix_fmt} -s {self.width}x{self.height} -r {self.fps} -i pipe: '
            f'-preset {self.preset} {bitrate_str} '
            f'-r {self.fps} -gpu {self.gpu} -c:v {self.codec} -pix_fmt yuv420p "{self.filename}"')
        self.process = run_async(self.ffmpeg_cmd)
