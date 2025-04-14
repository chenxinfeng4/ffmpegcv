from .video_info import run_async
from queue import Queue
from ffmpegcv.stream_info import get_info
from ffmpegcv.ffmpeg_reader_camera import FFmpegReaderCAM, ProducerThread
from ffmpegcv.ffmpeg_reader import (
    get_videofilter_cpu,
    get_outnumpyshape,
    get_videofilter_gpu,
    get_num_NVIDIA_GPUs,
    decoder_to_nvidia,
)


class FFmpegReaderStream(FFmpegReaderCAM):
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
        resize_keepratioalign,
        timeout,
    ):

        vid = FFmpegReaderStream()
        videoinfo = get_info(stream_url, timeout)
        vid.origin_width = videoinfo.width
        vid.origin_height = videoinfo.height
        vid.fps = videoinfo.fps
        vid.codec = codec if codec else videoinfo.codec
        vid.count = videoinfo.count
        vid.duration = videoinfo.duration
        vid.pix_fmt = pix_fmt

        (
            (vid.crop_width, vid.crop_height),
            (vid.width, vid.height),
            filteropt,
        ) = get_videofilter_cpu(
            (vid.origin_width, vid.origin_height),
            pix_fmt,
            crop_xywh,
            resize,
            resize_keepratio,
            resize_keepratioalign,
        )
        vid.size = (vid.width, vid.height)

        rtsp_opt = '' if not stream_url.startswith('rtsp://') else '-rtsp_flags prefer_tcp -pkt_size 736 '
        vid.ffmpeg_cmd = (
            f"ffmpeg -loglevel warning "
            f" {rtsp_opt} "
            f" -vcodec {vid.codec} -i {stream_url} "
            f" {filteropt} -pix_fmt {pix_fmt}  -f rawvideo pipe:"
        )
        vid.out_numpy_shape = get_outnumpyshape(vid.size, pix_fmt)
        vid.process = run_async(vid.ffmpeg_cmd)

        vid.isopened = True

        # producer
        vid.step = 1
        vid.q = Queue(maxsize=30)
        producer = ProducerThread(vid, vid.q)
        producer.start()
        return vid


class FFmpegReaderStreamNV(FFmpegReaderCAM):
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
        resize_keepratioalign,
        gpu,
        timeout,
    ):
        numGPU = get_num_NVIDIA_GPUs()
        vid = FFmpegReaderStreamNV()
        videoinfo = get_info(stream_url, timeout)
        vid.origin_width = videoinfo.width
        vid.origin_height = videoinfo.height
        vid.fps = videoinfo.fps
        vid.codec = codec if codec else videoinfo.codec
        vid.codecNV = decoder_to_nvidia(vid.codec)
        vid.count = videoinfo.count
        vid.duration = videoinfo.duration
        vid.pix_fmt = pix_fmt

        (
            (vid.crop_width, vid.crop_height),
            (vid.width, vid.height),
            (cropopt, scaleopt, filteropt),
        ) = get_videofilter_gpu(
            (vid.origin_width, vid.origin_height),
            pix_fmt,
            crop_xywh,
            resize,
            resize_keepratio,
            resize_keepratioalign,
        )
        vid.size = (vid.width, vid.height)

        rtsp_opt = '' if not stream_url.startswith('rtsp://') else '-rtsp_flags prefer_tcp -pkt_size 736 '
        vid.ffmpeg_cmd = (
            f"ffmpeg -loglevel warning -hwaccel cuda -hwaccel_device {gpu} "
            f" {rtsp_opt} "
            f' -vcodec {vid.codecNV} {cropopt} {scaleopt} -i "{stream_url}" '
            f" {filteropt} -pix_fmt {pix_fmt} -f rawvideo pipe:"
        )
        vid.out_numpy_shape = get_outnumpyshape(vid.size, pix_fmt)
        vid.process = run_async(vid.ffmpeg_cmd)

        vid.isopened = True

        # producer
        vid.step = 1
        vid.q = Queue(maxsize=30)
        producer = ProducerThread(vid, vid.q)
        producer.start()
        return vid
