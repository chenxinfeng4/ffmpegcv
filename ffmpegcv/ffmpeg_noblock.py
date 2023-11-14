from .ffmpeg_reader_noblock import FFmpegReaderNoblock
from .ffmpeg_writer_noblock import FFmpegWriterNoblock
from typing import Callable
import ffmpegcv
import threading
import numpy as np


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


class ReadLiveLast(threading.Thread, ffmpegcv.FFmpegReader):
    def __init__(self, fun, *args, **kvargs):
        threading.Thread.__init__(self)
        ffmpegcv.FFmpegReader.__init__(self)
        self.vid = vid = fun(*args, **kvargs)
        self.out_numpy_shape = vid.out_numpy_shape
        self.width, self.height = vid.width, vid.height
        self.img = np.zeros(self.out_numpy_shape, dtype=np.uint8)
        self.ret = True
        self._isopen = True
        self.start()

    def read(self):
        if self.ret:
            self.iframe += 1
        return self.ret, self.img
    
    def release(self):
        self._isopen = False
        self.vid.release()

    def run(self):
        while self._isopen:
            self.ret, self.img = self.vid.read()
            if self.ret:
                break