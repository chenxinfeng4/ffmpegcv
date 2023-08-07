from .video_info import run_async
from queue import Queue
from ffmpegcv.stream_info import get_info
from ffmpegcv.ffmpeg_reader_camera import FFmpegReaderCAM, ProducerThread


class FFmpegReaderStream(FFmpegReaderCAM):
    def __init__(self):
        super().__init__()

    @staticmethod
    def VideoReader(
        stream_url,
        pix_fmt,
        crop_xywh,
        resize,
        resize_keepratio,
        resize_keepratioalign,
    ):
        assert pix_fmt in ["rgb24", "bgr24", "yuv420p", "nv12", "gray"]
        vid = FFmpegReaderStream()
        videoinfo = get_info(stream_url)
        vid.width, vid.height = videoinfo.width, videoinfo.height
        vid.fps = videoinfo.fps
        vid.codec = videoinfo.codec
        vid.count = videoinfo.count
        vid.duration = videoinfo.duration

        vid.origin_width, vid.origin_height = vid.width, vid.height
        if crop_xywh:
            crop_w, crop_h = crop_xywh[2:]
            vid.width, vid.height = crop_w, crop_h
            x, y, w, h = crop_xywh
            cropopt = f"crop={w}:{h}:{x}:{y}"
        else:
            crop_w, crop_h = vid.origin_width, vid.origin_height
            cropopt = ""

        vid.crop_width, vid.crop_height = crop_w, crop_h

        if resize is None or resize == (vid.crop_width, vid.crop_height):
            scaleopt = ""
            padopt = ""
        else:
            vid.width, vid.height = dst_width, dst_height = resize
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

        pix_fmtopt = 'extractplanes=y' if pix_fmt=='gray' else ''
        if any([cropopt, scaleopt, padopt, pix_fmtopt]):
            filterstr = ",".join(x for x in [cropopt, scaleopt, padopt, pix_fmtopt] if x)
            filteropt = f"-vf {filterstr}"
        else:
            filteropt = ""

        vid.ffmpeg_cmd = (
            f"ffmpeg -loglevel warning "
            f' -vcodec {vid.codec} -i {stream_url} '
            f" {filteropt} -pix_fmt {pix_fmt}  -f rawvideo pipe:"
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
            "gray": (vid.height, vid.width, 1)
        }[pix_fmt]
        vid.process = run_async(vid.ffmpeg_cmd)

        vid.isopened = True

        # producer
        vid.step = 1
        vid.q = Queue(maxsize=30)
        producer = ProducerThread(vid, vid.q)
        producer.start()
        return vid
