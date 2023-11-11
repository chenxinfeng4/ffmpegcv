import os
from ffmpegcv.ffmpeg_reader import FFmpegReader
from .video_info import (
    get_info,
    get_num_QSV_GPUs,
    decoder_to_qsv,
)


class FFmpegReaderQSV(FFmpegReader):
    def _get_opts(
        vid, videoinfo, crop_xywh, resize, resize_keepratio, resize_keepratioalign, isgray
    ):
        vid.origin_width = videoinfo.width
        vid.origin_height = videoinfo.height
        vid.fps = videoinfo.fps
        vid.count = videoinfo.count
        vid.width, vid.height = vid.origin_width, vid.origin_height
        vid.codec = videoinfo.codec
        assert crop_xywh is None, 'Function not implemented yet'
        assert resize is None, 'Function not implemented yet'
        assert resize_keepratio is None or resize_keepratio==True, 'Function not implemented yet'
        assert resize_keepratioalign is None or resize_keepratioalign=="center", 'Function not implemented yet'
        assert vid.origin_height % 2 == 0, "height must be even"
        assert vid.origin_width % 2 == 0, "width must be even"
        if crop_xywh:
            pass
        else:
            crop_w, crop_h = vid.origin_width, vid.origin_height
            cropopt = ""

        vid.crop_width, vid.crop_height = crop_w, crop_h

        if resize is None or tuple(resize) == (vid.crop_width, vid.crop_height):
            scaleopt = ""
            filteropt = ""
        else:
            pass

        if isgray:
            if filteropt:
                filteropt=f'{filteropt},extractplanes=y'
            else:
                filteropt=f'-vf extractplanes=y'
        
        vid.size = (vid.width, vid.height)
        return cropopt, scaleopt, filteropt

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
        assert pix_fmt in ["rgb24", "bgr24", "yuv420p", "nv12", "gray"]
        assert gpu is None or gpu == 0, 'Cannot use multiple QSV gpu yet.'
        numGPU = get_num_QSV_GPUs()
        assert numGPU > 0, "No GPU found"
        gpu = int(gpu) % numGPU if gpu is not None else 0
        assert (
            resize is None or len(resize) == 2
        ), "resize must be a tuple of (width, height)"
        videoinfo = get_info(filename)
        vid = FFmpegReaderQSV()
        isgray = pix_fmt == "gray"
        cropopt, scaleopt, filteropt = vid._get_opts(
            videoinfo, crop_xywh, resize, resize_keepratio, resize_keepratioalign, isgray
        )
        vid.codecQSV = decoder_to_qsv(vid.codec)

        vid.ffmpeg_cmd = (
            f"ffmpeg -loglevel warning "
            f' -vcodec {vid.codecQSV} -r {vid.fps} -i "{filename}" '
            f" {filteropt} -pix_fmt {pix_fmt} -r {vid.fps} -f rawvideo pipe:"
        )

        vid.pix_fmt = pix_fmt
        assert (not pix_fmt == "yuv420p") or (
            vid.height % 2 == 0 and vid.width % 2 == 0
        ), "yuv420p must be even"
        vid.out_numpy_shape = {
            "rgb24": (vid.height, vid.width, 3),
            "bgr24": (vid.height, vid.width, 3),
            "yuv420p": (int(vid.height * 1.5), vid.width),
            "nv12": (int(vid.height * 1.5), vid.width),
            "gray": (vid.height, vid.width, 1)
        }[pix_fmt]
        return vid
