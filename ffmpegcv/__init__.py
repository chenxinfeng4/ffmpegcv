from .ffmpeg_reader import FFmpegReader, FFmpegReaderNV
from .ffmpeg_writer import FFmpegWriter, FFmpegWriterNV
from .ffmpeg_reader_camera import FFmpegReaderCAM
from .ffmpeg_reader_stream import FFmpegReaderStream
from .ffmpeg_reader_stream_realtime import FFmpegReaderStreamRT, FFmpegReaderStreamRTNV
from .ffmpeg_writer_stream_realtime import FFmpegWriterStreamRT
from .ffmpeg_reader_qsv import FFmpegReaderQSV
from .ffmpeg_writer_qsv import FFmpegWriterQSV
from .ffmpeg_reader_pannels import FFmpegReaderPannels
from .ffmpeg_noblock import noblock, ReadLiveLast
from .video_info import get_num_NVIDIA_GPUs
import shutil
from subprocess import DEVNULL, check_output

from .version import __version__


def _check():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "The ffmpeg is not installed. \n\n"
            "Please install ffmpeg via:\n    "
            "conda install ffmpeg"
        )


_check()

_check_nvidia_init = None


def _check_nvidia():
    global _check_nvidia_init
    run = lambda x: check_output(x, shell=True, stderr=DEVNULL)
    if _check_nvidia_init is None:
        calling_output = run("ffmpeg -h encoder=hevc_nvenc")
        if "AVOptions" not in calling_output.decode("utf-8"):
            raise RuntimeError(
                "The ffmpeg is not compiled with NVENC support.\n\n"
                "Please re-compile ffmpeg following the instructions at:\n    "
                "https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/"
            )

        calling_output = run("ffmpeg -h decoder=hevc_cuvid")
        if "AVOptions" not in calling_output.decode("utf-8"):
            raise RuntimeError(
                "The ffmpeg is not compiled with NVENC support.\n\n"
                "Please re-compile ffmpeg following the instructions at:\n    "
                "https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/"
            )

        if get_num_NVIDIA_GPUs() == 0:
            raise RuntimeError(
                "No NVIDIA GPU found.\n\n"
                "Please use a NVIDIA GPU card listed at:\n    "
                "https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new"
            )

        _check_nvidia_init = True

    return True


def VideoCapture(
    file,
    codec=None,
    pix_fmt="bgr24",
    crop_xywh=None,
    resize=None,
    resize_keepratio=True,
    resize_keepratioalign="center",
) -> FFmpegReader:
    """
    Alternative to cv2.VideoCapture

    Parameters
    ----------
    file : str
        Path to video file.
    codec : str
        Codec to use. Optional. Default is `None`.
    pix_fmt : str
        Pixel format. ['bgr24' | 'rgb24']. Optional. Default is 'bgr24'.
    crop_xywh : tuple
        Crop the frame. (x, y, width, height). Optional. Default is `None`.
    resize  : tuple
        Resize the video to the given size. Optional. Default is `None`.
    resize_keepratio : bool
        Keep the aspect ratio and the border is black. Optional. Default is `True`.
    resize_keepratioalign : str
        Align the image to the `center`, `topleft`, `topright`, `bottomleft` or
        `bottomright`. Optional. Default is 'center'.

    Examples
    --------
    opencv
    ```
    cap = cv2.VideoCapture(file)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pass
    ```

    ffmpegcv
    ```
    cap = ffmpegcv.VideoCapture(file)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pass
    ```

    Or use iterator
    ```
    cap = ffmpegcv.VideoCapture(file)
    for frame in cap:
        pass
    counts = len(cap)
    ```

    Use GPU to accelerate decoding
    ```
    cap_cpu = ffmpegcv.VideoCapture(file)
    cap_gpu = ffmpegcv.VideoCaptureNV(file)
    ```

    Use rgb24 instead of bgr24
    ```
    cap = ffmpegcv.VideoCapture(file, pix_fmt='rgb24')
    ```

    Crop video.
    ```python
    cap = ffmpegcv.VideoCapture(file, crop_xywh=(0, 0, 640, 480))
    ```

    Resize the video to the given size
    ```
    cap = ffmpegcv.VideoCapture(file, resize=(640, 480))
    ```

    Resize and keep the aspect ratio with black border
    ```
    cap = ffmpegcv.VideoCapture(file, resize=(640, 480), resize_keepratio=True)
    ```

    Crop and then resize the video.
    ```python
    cap = ffmpegcv.VideoCapture(file, crop_xywh=(0, 0, 640, 480), resize=(512, 512))
    ```
    Author: Chenxinfeng 2022-04-16, cxf529125853@163.com
    """
    return FFmpegReader.VideoReader(
        file, codec, pix_fmt, crop_xywh, resize, resize_keepratio, resize_keepratioalign
    )


VideoReader = VideoCapture


def VideoWriter(
    file, codec=None, fps=30, pix_fmt="bgr24", bitrate=None, resize=None, preset=None
) -> FFmpegWriter:
    """
    Alternative to cv2.VideoWriter

    Parameters
    ----------
    file : str
        Path to video file.
    codec : str
        Codec to use. Optional. Default is `None` (x264).
    fps : number
        Frames per second. Optional. Default is 30.
    pix_fmt : str
        Pixel format of input. ['bgr24' | 'rgb24']. Optional. Default is 'bgr24'.
    bitrate : str
        Bitrate of output video. Optional. Default is `None`.
    resize : tuple
        Frame size of output. (width, height). Optional. Default is `None`.
    preset : str
        Preset of ffmpeg. Optional. Default is `None`.
    Examples
    --------
    opencv
    ```
    out = cv2.VideoWriter('outpy.avi',
                          cv2.VideoWriter_fourcc('M','J','P','G'),
                          10,
                          (w, h))
    out.write(frame1)
    out.write(frame2)
    out.release()
    ```

    ffmpegcv
    ```
    out = ffmpegcv.VideoWriter('outpy.avi', None, 10)
    out.write(frame1)
    out.write(frame2)
    out.release()
    ```

    frameSize is decided by the size of the first frame
    ```
    out = ffmpegcv.VideoWriter('outpy.avi', None, 10)
    ```

    Use GPU to accelerate encoding
    ```
    out_cpu = ffmpegcv.VideoWriter('outpy.avi', None, 10)
    out_gpu = ffmpegcv.VideoWriter('outpy.avi', 'h264_nvenc', 10)
    ```

    Use rgb24 instead of bgr24
    ```
    out = ffmpegcv.VideoWriter('outpy.avi', None, 10, pix_fmt='rgb24')
    out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ```

    Author: Chenxinfeng 2022-04-16, cxf529125853@163.com
    """
    return FFmpegWriter.VideoWriter(
        file, codec, fps, pix_fmt, bitrate, resize, preset=preset
    )


def VideoCaptureNV(
    file,
    pix_fmt="bgr24",
    crop_xywh=None,
    resize=None,
    resize_keepratio=True,
    resize_keepratioalign="center",
    gpu=0,
) -> FFmpegReaderNV:
    """
    `ffmpegcv.VideoCaptureNV` is a gpu version for `ffmpegcv.VideoCapture`.
    """
    _check_nvidia()
    return FFmpegReaderNV.VideoReader(
        file, pix_fmt, crop_xywh, resize, resize_keepratio, resize_keepratioalign, gpu
    )


VideoReaderNV = VideoCaptureNV


def VideoCaptureQSV(
    file,
    pix_fmt="bgr24",
    crop_xywh=None,
    resize=None,
    resize_keepratio=True,
    resize_keepratioalign="center",
    gpu=0,
) -> FFmpegReaderQSV:
    """
    `ffmpegcv.VideoCaptureQSV` is a gpu version for `ffmpegcv.VideoCapture`.
    """
    return FFmpegReaderQSV.VideoReader(
        file, pix_fmt, crop_xywh, resize, resize_keepratio, resize_keepratioalign, gpu
    )


VideoReaderQSV = VideoCaptureQSV


def VideoWriterNV(
    file,
    codec=None,
    fps=30,
    pix_fmt="bgr24",
    gpu=0,
    bitrate=None,
    resize=None,
    preset=None,
) -> FFmpegWriterNV:
    """
    `ffmpegcv.VideoWriterNV` is a gpu version for `ffmpegcv.VideoWriter`.
    """
    _check_nvidia()
    return FFmpegWriterNV.VideoWriter(
        file, codec, fps, pix_fmt, gpu, bitrate, resize, preset=preset
    )


def VideoWriterQSV(
    file,
    codec=None,
    fps=30,
    pix_fmt="bgr24",
    gpu=0,
    bitrate=None,
    resize=None,
    preset=None,
) -> FFmpegWriterQSV:
    """
    `ffmpegcv.VideoWriterQSV` is a gpu version for `ffmpegcv.VideoWriter`.
    """
    return FFmpegWriterQSV.VideoWriter(
        file, codec, fps, pix_fmt, gpu, bitrate, resize, preset=preset
    )


def VideoWriterStreamRT(
    url, pix_fmt="bgr24", bitrate=None, resize=None, preset=None
) -> FFmpegWriterStreamRT:
    return FFmpegWriterStreamRT.VideoWriter(
        url, "libx264", pix_fmt, bitrate, resize, preset
    )


def VideoCaptureCAM(
    camname,
    pix_fmt="bgr24",
    crop_xywh=None,
    resize=None,
    resize_keepratio=True,
    resize_keepratioalign="center",
    camsize_wh=None,
    camfps=None,
    camcodec=None,
    campix_fmt=None,
) -> FFmpegReaderCAM:
    """
    Alternative to cv2.VideoCapture

    Parameters
    ----------
    file : see ffmpegcv.VideoReader
    codec : see ffmpegcv.VideoReader
    pix_fmt : see ffmpegcv.VideoReader
    crop_xywh : see ffmpegcv.VideoReader
    resize  : see ffmpegcv.VideoReader
    resize_keepratio : see ffmpegcv.VideoReader
    resize_keepratioalign : see ffmpegcv.VideoReader
    camsize_wh: tuple or None
        Camera resolution (width, height). e.g (800, 600)
    camfps: float or None
        Camera framerate. e.g. 30.
    camcodec: str or None
        Camera codec. e.g. 'mjpeg' or 'h264'.
    campix_fmt: str or None
        Camera pixel format. e.g. 'rgb24' or 'yuv420p'.
        Just set one of `camcodec` or `campix_fmt`.
    Examples
    --------
    opencv
    ```
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pass
    ```

    ffmpegcv
    ```
    cap = ffmpegcv.VideoCaptureCAM(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pass
    ```

    Or use camera name
    ```
    cap = ffmpegcv.VideoCaptureCAM("Integrated Camera")
    ```

    Use full camera parameter
    ```
    cap = ffmpegcv.VideoCaptureCAM('FaceTime HD Camera',
                                    camsize_wh = (1280,720),
                                    camfps = 30,
                                    campix_fmt = 'nv12')
    ```

    Use camera with ROI operations
    ```
    cap = ffmpegcv.VideoCaptureCAM("Integrated Camera",
                                    crop_xywh = (0, 0, 640, 480),
                                    resize = (512, 512),
                                    resize_keepratio = True)
    ```
    Author: Chenxinfeng 2023-05-11, cxf529125853@163.com
    """
    return FFmpegReaderCAM.VideoReader(
        camname,
        pix_fmt,
        crop_xywh,
        resize,
        resize_keepratio,
        resize_keepratioalign,
        camsize_wh=camsize_wh,
        camfps=camfps,
        camcodec=camcodec,
        campix_fmt=campix_fmt,
    )


VideoReaderCAM = VideoCaptureCAM


def VideoCaptureStream(
    stream_url,
    codec=None,
    pix_fmt="bgr24",
    crop_xywh=None,
    resize=None,
    resize_keepratio=True,
    resize_keepratioalign="center",
    timeout=None,
) -> FFmpegReaderStream:
    """
    Alternative to cv2.VideoCapture

    Parameters
    ----------
    stream_url : RTSP, RTP, RTMP, HTTP, HTTPS url
    codec : see ffmpegcv.VideoReader
    pix_fmt : see ffmpegcv.VideoReader
    crop_xywh : see ffmpegcv.VideoReader
    resize  : see ffmpegcv.VideoReader
    resize_keepratio : see ffmpegcv.VideoReader
    resize_keepratioalign : see ffmpegcv.VideoReader
    timeout : waits in seconds for stream video to connect

    Examples
    --------
    opencv
    ```
    stream_url = 'http://devimages.apple.com.edgekey.net/streaming/examples/bipbop_4x3/gear2/prog_index.m3u8'
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print('Cannot open the stream')
        exit(-1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pass
    ```

    ffmpegcv
    ```
    cap = ffmpegcv.VideoCaptureStream(stream_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pass
    ```

    Author: Chenxinfeng 2023-05-31, cxf529125853@163.com
    """
    return FFmpegReaderStream.VideoReader(
        stream_url,
        codec,
        pix_fmt,
        crop_xywh,
        resize,
        resize_keepratio,
        resize_keepratioalign,
        timeout,
    )


VideoReaderStream = VideoCaptureStream


def VideoCaptureStreamRT(
    stream_url,
    codec=None,
    pix_fmt="bgr24",
    crop_xywh=None,
    resize=None,
    resize_keepratio=True,
    resize_keepratioalign="center",
    gpu=None,
    timeout=None,
) -> FFmpegReaderStreamRT:
    if gpu is None:
        return FFmpegReaderStreamRT.VideoReader(
            stream_url,
            codec,
            pix_fmt,
            crop_xywh,
            resize,
            resize_keepratio,
            resize_keepratioalign,
            timeout=timeout,
        )
    else:
        return FFmpegReaderStreamRTNV.VideoReader(
            stream_url,
            codec,
            pix_fmt,
            crop_xywh,
            resize,
            resize_keepratio,
            resize_keepratioalign,
            gpu=gpu,
            timeout=timeout,
        )


VideoReaderStreamRT = VideoCaptureStreamRT


def VideoCapturePannels(
    file: str, crop_xywh_l: list, codec=None, pix_fmt="bgr24", resize=None
):
    """
    Alternative to cv2.VideoCapture

    Parameters
    ----------
    file : str
        Path to video file.
    crop_xywh_l : list of crop_xywh
        Crop the frame. [(x0, y0, w0, h0), (x1, y1, w1, h1), ...].
    codec : str
        Codec to use. Optional. Default is `None`.
    pix_fmt : str
        Pixel format. ['bgr24' | 'rgb24' | 'gray']. Optional. Default is 'bgr24'.
    resize  : tuple as (w,h)
        Resize the pannels to identical size. Optional. Default is `None`.
        Does not keep the ratio.


    Examples
    --------


    ffmpegcv
    ```
    w,h = 1280,800
    cap = ffmpegcv.VideoCapturePannels(file,
        [[0,0,w,h], [w,0,w,h],[0,h,w,h], [w,h,w,h]])
    while True:
        ret, frame_pannels = cap.read()
        if not ret:
            break
        print(frame_pannels.shape)  # shape=(4,h,w,3)
    ```

    Author: Chenxinfeng 2024-02-15, cxf529125853@163.com
    """

    return FFmpegReaderPannels.VideoReader(
        file,
        crop_xywh_l,
        codec,
        pix_fmt,
        resize,
    )


VideoReaderPannels = VideoCapturePannels


def toCUDA(vid: FFmpegReader, gpu: int = 0, tensor_format: str = "chw") -> FFmpegReader:
    """
    Convert frames to CUDA tensor float32 in 'chw' or 'hwc' format.
    """
    from ffmpegcv.ffmpeg_reader_cuda import FFmpegReaderCUDA

    return FFmpegReaderCUDA(vid, gpu, tensor_format)
