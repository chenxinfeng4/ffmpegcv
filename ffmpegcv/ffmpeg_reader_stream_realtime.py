from .video_info import run_async
from queue import Queue
from ffmpegcv.ffmpeg_reader_camera import FFmpegReaderCAM, ProducerThread
import numpy as np

class FFmpegReaderStreamRT(FFmpegReaderCAM):
    def __init__(self):
        super().__init__()

    @staticmethod
    def VideoReader(
        stream_url,
        pix_fmt,
        camsize
    ):
        assert pix_fmt in ["rgb24", "bgr24", "yuv420p", "nv12","gray"]
        vid = FFmpegReaderStreamRT()
        vid.width, vid.height = camsize

        vid.ffmpeg_cmd = (
            f"ffmpeg -loglevel warning "
            '-rtsp_transport tcp '
            '-fflags nobuffer -flags low_delay -strict experimental '
            f' -i {stream_url}'
            f" -pix_fmt {pix_fmt}  -f rawvideo pipe:"
        )

        vid.size = (vid.width, vid.height)
        vid.pix_fmt = pix_fmt
        assert (not pix_fmt == "yuv420p") or (
            vid.height % 2 == 0 and vid.width % 2 == 0
        ), "yuv420p must be even"
        vid.out_numpy_shape = {
            "rgb24": (vid.height, vid.width, 3),
            "bgr24": (vid.height, vid.width, 3),
            "nv12": (int(vid.height * 1.5), vid.width),
            "yuv420p": (int(vid.height * 1.5), vid.width),
            "gray": (vid.height, vid.width),
        }[pix_fmt]
        vid.process = run_async(vid.ffmpeg_cmd)
        vid.isopened = True
        return vid

    def read(self):
        in_bytes = self.process.stdout.read(np.prod(self.out_numpy_shape))
        if not in_bytes:
            return False, None

        self.iframe += 1
        img = None
        img = np.frombuffer(in_bytes, np.uint8).reshape(self.out_numpy_shape)
        return True, img
