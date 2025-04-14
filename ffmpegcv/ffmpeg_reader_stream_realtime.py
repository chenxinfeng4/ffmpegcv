from ffmpegcv.ffmpeg_reader import (
    FFmpegReader,
    get_videofilter_cpu,
    get_outnumpyshape,
    get_videofilter_gpu,
    get_num_NVIDIA_GPUs,
    decoder_to_nvidia,
)
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
        resize_keepratioalign,
        timeout,
    ):
        vid = FFmpegReaderStreamRT()
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
            f"ffmpeg -loglevel error "
            f" {rtsp_opt} "
            "-fflags nobuffer -flags low_delay -strict experimental "
            f" -vcodec {vid.codec} -i {stream_url}"
            f" {filteropt} -pix_fmt {pix_fmt}  -f rawvideo pipe:"
        )

        vid.out_numpy_shape = get_outnumpyshape(vid.size, pix_fmt)
        return vid


class FFmpegReaderStreamRTNV(FFmpegReader):
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
        vid = FFmpegReaderStreamRTNV()
        videoinfo = get_info(stream_url, timeout)
        vid.origin_width = videoinfo.width
        vid.origin_height = videoinfo.height
        vid.fps = videoinfo.fps
        vid.codec = codec if codec else videoinfo.codec
        vid.codecNV = decoder_to_nvidia(vid.codec)
        vid.count = videoinfo.count
        vid.duration = videoinfo.duration
        vid.pix_fmt = pix_fmt

        numGPU = get_num_NVIDIA_GPUs()
        assert numGPU > 0, "No GPU found"
        gpu = int(gpu) % numGPU if gpu is not None else 0

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

        rtsp_opt = "-rtsp_transport tcp " if stream_url.startswith("rtsp://") else ""
        vid.ffmpeg_cmd = (
            f"ffmpeg -loglevel error -hwaccel cuda -hwaccel_device {gpu} "
            f" {rtsp_opt} "
            f"{cropopt} {scaleopt} "
            "-fflags nobuffer -flags low_delay -strict experimental "
            f' -vcodec {vid.codecNV} -i "{stream_url}" '
            f" {filteropt} -pix_fmt {pix_fmt} -f rawvideo pipe:"
        )

        vid.out_numpy_shape = get_outnumpyshape(vid.size, pix_fmt)
        return vid
