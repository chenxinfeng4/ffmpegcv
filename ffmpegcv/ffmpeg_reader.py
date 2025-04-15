import numpy as np
import pprint
import warnings
import os
import sys
import select
from .video_info import (
    run_async_reader as run_async,
    get_info,
    get_num_NVIDIA_GPUs,
    decoder_to_nvidia,
    release_process,
)


def get_videofilter_cpu(
    originsize: list,
    pix_fmt: str,
    crop_xywh: list,
    resize: list,
    resize_keepratio: bool,
    resize_keepratioalign: str,
):
    """
    ONGONING: common filter for video/cam/stream capture.
    """
    assert pix_fmt in ["rgb24", "bgr24", "yuv420p", "yuvj420p", "nv12", "gray"]
    origin_width, origin_height = originsize
    if crop_xywh:
        crop_w, crop_h = crop_xywh[2:]
        if not all([n % 2 == 0] for n in crop_xywh):
            print("Warning 'crop_xywh' would be replaced into even numbers")
            crop_xywh = [int(n//2*2) for n in crop_xywh]
        assert crop_w <= origin_width and crop_h <= origin_height
        x, y, w, h = crop_xywh
        cropopt = f"crop={w}:{h}:{x}:{y}"
    else:
        crop_w, crop_h = origin_width, origin_height
        cropopt = ""

    crop_wh = (crop_w, crop_h)
    if resize is None or tuple(resize) == crop_wh:
        scaleopt = ""
        padopt = ""
        final_size_wh = crop_wh
    else:
        final_size_wh = (dst_width, dst_height) = resize
        assert all([n % 2 == 0] for n in resize), "'resize' must be even number"
        if not resize_keepratio:
            scaleopt = f"scale={dst_width}x{dst_height}"
            padopt = ""
        else:
            re_width, re_height = crop_w / (crop_h / dst_height), dst_height
            if re_width > dst_width:
                re_width, re_height = dst_width, crop_h / (crop_w / dst_width)
            re_width, re_height = int(re_width), int(re_height)
            scaleopt = f"scale={re_width}x{re_height}"
            if resize_keepratioalign is None:
                resize_keepratioalign = "center"
            paddings = {
                "center": ((dst_width - re_width) // 2, (dst_height - re_height) // 2,),
                "topleft": (0, 0),
                "topright": (dst_width - re_width, 0),
                "bottomleft": (0, dst_height - re_height),
                "bottomright": (dst_width - re_width, dst_height - re_height),
            }
            assert (
                resize_keepratioalign in paddings
            ), 'resize_keepratioalign must be one of "center"(mmpose), "topleft"(mmdetection), "topright", "bottomleft", "bottomright"'
            xpading, ypading = paddings[resize_keepratioalign]
            padopt = f"pad={dst_width}:{dst_height}:{xpading}:{ypading}:black"

    pix_fmtopt = "extractplanes=y" if pix_fmt == "gray" else ""
    if any([cropopt, scaleopt, padopt, pix_fmtopt]):
        filterstr = ",".join(x for x in [cropopt, scaleopt, padopt, pix_fmtopt] if x)
        filteropt = f"-vf {filterstr}"
    else:
        filteropt = ""
    return crop_wh, final_size_wh, filteropt


def get_videofilter_gpu(
    originsize: list,
    pix_fmt: str,
    crop_xywh: list,
    resize: list,
    resize_keepratio: bool,
    resize_keepratioalign: str,
):
    assert pix_fmt in ["rgb24", "bgr24", "yuv420p", "yuvj420p", "nv12", "gray"]
    origin_width, origin_height = originsize
    if crop_xywh:
        crop_w, crop_h = crop_xywh[2:]
        assert all([n % 2 == 0] for n in crop_xywh), "'crop_xywh' must be even number"
        assert crop_w <= origin_width and crop_h <= origin_height
        x, y, w, h = crop_xywh
        top, bottom, left, right = (
            y,
            origin_height - (y + h),
            x,
            origin_width - (x + w),
        )  # crop length
        cropopt = f"-crop {top}x{bottom}x{left}x{right}"
    else:
        crop_w, crop_h = origin_width, origin_height
        cropopt = ""

    crop_wh = (crop_w, crop_h)
    filteropt = ""
    scaleopt = ""
    if resize is None or tuple(resize) == crop_wh:
        final_size_wh = crop_wh
    else:
        final_size_wh = (dst_width, dst_height) = resize
        assert all([n % 2 == 0] for n in resize), "'resize' must be even number"
        if not resize_keepratio:
            scaleopt = f"-resize {dst_width}x{dst_height}"
        else:
            re_width, re_height = crop_w / (crop_h / dst_height), dst_height
            if re_width > dst_width:
                re_width, re_height = dst_width, crop_h / (crop_w / dst_width)
            re_width, re_height = int(re_width), int(re_height)
            scaleopt = f"-resize {re_width}x{re_height}"
            if resize_keepratioalign is None:
                resize_keepratioalign = "center"
            paddings = {
                "center": ((dst_width - re_width) // 2, (dst_height - re_height) // 2,),
                "topleft": (0, 0),
                "topright": (dst_width - re_width, 0),
                "bottomleft": (0, dst_height - re_height),
                "bottomright": (dst_width - re_width, dst_height - re_height),
            }
            assert (
                resize_keepratioalign in paddings
            ), 'resize_keepratioalign must be one of "center"(mmpose), "topleft"(mmdetection), "topright", "bottomleft", "bottomright"'
            xpading, ypading = paddings[resize_keepratioalign]
            padopt = f"pad={dst_width}:{dst_height}:{xpading}:{ypading}:black"
            filteropt = f"-vf {padopt}"

    if pix_fmt == "gray":
        if filteropt:
            filteropt = f"{filteropt},extractplanes=y"
        else:
            filteropt = f"-vf extractplanes=y"

    return crop_wh, final_size_wh, [cropopt, scaleopt, filteropt]


def get_outnumpyshape(size_wh: list, pix_fmt: str) -> tuple:
    width, height = size_wh
    assert (not pix_fmt == "yuv420p") or (
        height % 2 == 0 and width % 2 == 0
    ), "yuv420p must be even"
    out_numpy_shape = {
        "rgb24": (height, width, 3),
        "bgr24": (height, width, 3),
        "yuv420p": (int(height * 1.5), width),
        "yuvj420p": (int(height * 1.5), width),
        "nv12": (int(height * 1.5), width),
        "gray": (height, width, 1),
    }[pix_fmt]
    return out_numpy_shape


class FFmpegReader:
    def __init__(self):
        self.iframe = -1
        self.width = None
        self.height = None
        self.size = (None, None)
        self.waitInit = True
        self.process = None
        self._isopen = True
        self.debug = False
        self.out_numpy_shape = (None, None, None)

    def __repr__(self):
        props = pprint.pformat(self.__dict__).replace("{", " ").replace("}", " ")
        return f"{self.__class__}\n" + props

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.release()

    def __len__(self):
        return self.count

    def __iter__(self):
        return self

    def __next__(self):
        ret, img = self.read()
        if ret:
            return img
        else:
            raise StopIteration

    @staticmethod
    def VideoReader(
        filename,
        codec,
        pix_fmt,
        crop_xywh,
        resize,
        resize_keepratio,
        resize_keepratioalign,
    ):
        assert os.path.exists(filename) and os.path.isfile(
            filename
        ), f"{filename} not exists"

        vid = FFmpegReader()
        videoinfo = get_info(filename)
        vid.origin_width = videoinfo.width
        vid.origin_height = videoinfo.height
        vid.fps = videoinfo.fps
        vid.count = videoinfo.count
        vid.duration = videoinfo.duration
        vid.pix_fmt = pix_fmt
        vid.codec = videoinfo.codec
        
        if codec is not None:
            warnings.warn(
                "The 'codec' parameter is auto detected and will be removed " 
                "in future versions. Please refrain from using this parameter.",
                DeprecationWarning
            )

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
            f"ffmpeg -loglevel error "
            f' -vcodec {vid.codec} -r {vid.fps} -i "{filename}" '
            f" {filteropt} -pix_fmt {pix_fmt} -r {vid.fps} -f rawvideo pipe:"
        )
        vid.out_numpy_shape = get_outnumpyshape(vid.size, pix_fmt)
        return vid

    def read(self):
        if self.waitInit:
            self.process = run_async(self.ffmpeg_cmd)
            self.waitInit = False
        
        in_bytes = self.process.stdout.read(np.prod(self.out_numpy_shape))
        # check if ffmpeg process error
        # if self.process.stderr.readable():
        #     print('---a')
        #     data = self.process.stderr.read()
        #     sys.stderr.buffer.write(data)
        #     print('---f')

        if not in_bytes:
            self.release()
            return False, None
        self.iframe += 1
        img = np.frombuffer(in_bytes, np.uint8).reshape(self.out_numpy_shape)

        return True, img

    def isOpened(self):
        return self._isopen

    def release(self):
        self._isopen = False
        release_process(self.process, forcekill=True)

    def close(self):
        return self.release()


class FFmpegReaderNV(FFmpegReader):
    def _get_opts(
        vid,
        videoinfo,
        crop_xywh,
        resize,
        resize_keepratio,
        resize_keepratioalign,
        isgray,
    ):
        vid.origin_width = videoinfo.width
        vid.origin_height = videoinfo.height
        vid.fps = videoinfo.fps
        vid.count = videoinfo.count
        vid.duration = videoinfo.duration
        vid.width, vid.height = vid.origin_width, vid.origin_height
        vid.codec = videoinfo.codec
        assert vid.origin_height % 2 == 0, "height must be even"
        assert vid.origin_width % 2 == 0, "width must be even"
        if crop_xywh:
            crop_w, crop_h = crop_xywh[2:]
            vid.width, vid.height = crop_w, crop_h
            x, y, w, h = crop_xywh
            top, bottom, left, right = (
                y,
                vid.origin_height - (y + h),
                x,
                vid.origin_width - (x + w),
            )  # crop length
            cropopt = f"-crop {top}x{bottom}x{left}x{right}"
        else:
            crop_w, crop_h = vid.origin_width, vid.origin_height
            cropopt = ""

        vid.crop_width, vid.crop_height = crop_w, crop_h

        if resize is None or tuple(resize) == (vid.crop_width, vid.crop_height):
            scaleopt = ""
            filteropt = ""
        else:
            vid.width, vid.height = dst_width, dst_height = resize
            if not resize_keepratio:
                scaleopt = f"-resize {dst_width}x{dst_height}"
                filteropt = ""
            else:
                re_width, re_height = crop_w / (crop_h / dst_height), dst_height
                if re_width > dst_width:
                    re_width, re_height = dst_width, crop_h / (crop_w / dst_width)
                re_width, re_height = int(re_width), int(re_height)
                scaleopt = f"-resize {re_width}x{re_height}"
                if resize_keepratioalign is None:
                    resize_keepratioalign = "center"
                paddings = {
                    "center": (
                        (dst_width - re_width) // 2,
                        (dst_height - re_height) // 2,
                    ),
                    "topleft": (0, 0),
                    "topright": (dst_width - re_width, 0),
                    "bottomleft": (0, dst_height - re_height),
                    "bottomright": (dst_width - re_width, dst_height - re_height),
                }
                assert (
                    resize_keepratioalign in paddings
                ), 'resize_keepratioalign must be one of "center"(mmpose), "topleft"(mmdetection), "topright", "bottomleft", "bottomright"'
                xpading, ypading = paddings[resize_keepratioalign]
                padopt = f"pad={dst_width}:{dst_height}:{xpading}:{ypading}:black"
                filteropt = f"-vf {padopt}"

        if isgray:
            if filteropt:
                filteropt = f"{filteropt},extractplanes=y"
            else:
                filteropt = f"-vf extractplanes=y"

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
        assert os.path.exists(filename) and os.path.isfile(
            filename
        ), f"{filename} not exists"
        assert pix_fmt in ["rgb24", "bgr24", "yuv420p", "yuvj420p", "nv12", "gray"]
        numGPU = get_num_NVIDIA_GPUs()
        assert numGPU > 0, "No GPU found"
        gpu = int(gpu) % numGPU if gpu is not None else 0
        assert (
            resize is None or len(resize) == 2
        ), "resize must be a tuple of (width, height)"
        videoinfo = get_info(filename)
        vid = FFmpegReaderNV()
        isgray = pix_fmt == "gray"
        cropopt, scaleopt, filteropt = vid._get_opts(
            videoinfo,
            crop_xywh,
            resize,
            resize_keepratio,
            resize_keepratioalign,
            isgray,
        )
        vid.codecNV = decoder_to_nvidia(vid.codec)

        vid.ffmpeg_cmd = (
            f"ffmpeg -loglevel error -hwaccel cuda -hwaccel_device {gpu} "
            f' -vcodec {vid.codecNV} {cropopt} {scaleopt} -r {vid.fps} -i "{filename}" '
            f" {filteropt} -pix_fmt {pix_fmt} -r {vid.fps} -f rawvideo pipe:"
        )

        vid.pix_fmt = pix_fmt
        vid.out_numpy_shape = get_outnumpyshape(vid.size, pix_fmt)
        return vid
