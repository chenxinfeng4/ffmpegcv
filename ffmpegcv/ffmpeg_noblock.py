from .ffmpeg_reader_noblock import FFmpegReaderNoblock
from .ffmpeg_writer_noblock import FFmpegWriterNoblock
from typing import Callable
import ffmpegcv


def noblock(fun:Callable, *v_args, **v_kargs):
    readerfuns = (ffmpegcv.VideoCapture, ffmpegcv.VideoCaptureNV)
    writerfuns = (ffmpegcv.VideoWriter, ffmpegcv.VideoWriterNV)

    if fun in readerfuns:
        proxyfun = FFmpegReaderNoblock(fun, *v_args, **v_kargs)
    elif fun in writerfuns:
        proxyfun = FFmpegWriterNoblock(fun, *v_args, **v_kargs)
    else:
        raise ValueError('The function is not supported as a Reader or Writer')
    
    return proxyfun
