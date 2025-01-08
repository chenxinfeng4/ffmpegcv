import os
from ffmpegcv.ffmpeg_reader import FFmpegReader, get_videofilter_cpu, get_outnumpyshape
from .video_info import (
    get_info,
    get_num_QSV_GPUs,
    decoder_to_qsv,
)


class FFmpegReaderQSV(FFmpegReader):
    @staticmethod
    def VideoReader(
        filename,
        pix_fmt,
        crop_xywh,
        resize,
        resize_keepratio,
        resize_keepratioalign,
        gpu,
    ):
        """
        TODO: 1. only 1 gpu is recognized
        TODO: 2. 'crop_xywh', 'resize*' are not supported yet
        """
        assert os.path.exists(filename) and os.path.isfile(
            filename
        ), f"{filename} not exists"
        assert gpu is None or gpu == 0, "Cannot use multiple QSV gpu yet."
        numGPU = get_num_QSV_GPUs()
        assert numGPU > 0, "No GPU found"
        gpu = int(gpu) % numGPU if gpu is not None else 0
        assert (
            resize is None or len(resize) == 2
        ), "resize must be a tuple of (width, height)"

        vid = FFmpegReaderQSV()
        videoinfo = get_info(filename)
        vid.origin_width = videoinfo.width
        vid.origin_height = videoinfo.height
        vid.fps = videoinfo.fps
        vid.count = videoinfo.count
        vid.duration = videoinfo.duration
        vid.codecQSV = decoder_to_qsv(videoinfo.codec)
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

        vid.ffmpeg_cmd = (
            f"ffmpeg -loglevel warning "
            f' -vcodec {vid.codecQSV} -r {vid.fps} -i "{filename}" '
            f" {filteropt} -pix_fmt {pix_fmt} -r {vid.fps} -f rawvideo pipe:"
        )

        vid.out_numpy_shape = get_outnumpyshape(vid.size, pix_fmt)
        return vid
