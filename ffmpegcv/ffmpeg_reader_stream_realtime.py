from .video_info import run_async
from queue import Queue
from ffmpegcv.ffmpeg_reader import FFmpegReader, get_videofilter_cpu, get_outnumpyshape
import numpy as np
from ffmpegcv.stream_info import get_info


class FFmpegReaderStreamRT(FFmpegReader):
    def __init__(self):
        super().__init__()

    @staticmethod
    def VideoReader(
        stream_url,
        codec,
        pix_fmt,
        crop_xywh,
        resize,
        resize_keepratio,
        resize_keepratioalign
    ):
        assert pix_fmt in ["rgb24", "bgr24", "yuv420p", "nv12","gray"]
        vid = FFmpegReaderStreamRT()
        videoinfo = get_info(stream_url)
        vid.origin_width = videoinfo.width
        vid.origin_height = videoinfo.height
        vid.fps = videoinfo.fps
        vid.codec = codec if codec else videoinfo.codec
        vid.count = videoinfo.count
        vid.duration = videoinfo.duration

        (vid.crop_width, vid.crop_height), (vid.width, vid.height), filteropt = get_videofilter_cpu(
                (vid.origin_width, vid.origin_height), pix_fmt, crop_xywh, resize, 
                resize_keepratio, resize_keepratioalign)
        vid.size = (vid.width, vid.height)

        rtsp_opt = '-rtsp_transport tcp ' if stream_url.startswith('rtsp://') else ''
        vid.ffmpeg_cmd = (
            f"ffmpeg -loglevel warning "
            f' {rtsp_opt} '
            '-fflags nobuffer -flags low_delay -strict experimental '
            f' -vcodec {vid.codec} -i {stream_url}'
            f" {filteropt} -pix_fmt {pix_fmt}  -f rawvideo pipe:"
        )

        vid.out_numpy_shape = get_outnumpyshape(vid.size, pix_fmt)
        return vid
    