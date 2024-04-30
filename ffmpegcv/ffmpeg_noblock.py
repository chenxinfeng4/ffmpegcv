from .ffmpeg_reader_noblock import FFmpegReaderNoblock
from .ffmpeg_writer_noblock import FFmpegWriterNoblock
from typing import Callable
import ffmpegcv
import threading
import numpy as np
import queue


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
        props_name = ['width', 'height', 'fps', 'count', 'codec', 'ffmpeg_cmd',
                      'size', 'pix_fmt', 'out_numpy_shape', 'iframe', 
                      'duration', 'origin_width', 'origin_height']
        for name in props_name:
            setattr(self, name, getattr(vid, name, None))

        self.img = np.zeros(self.out_numpy_shape, dtype=np.uint8)
        self.ret = True
        self._isopen = True
        self._q = queue.Queue(maxsize=1) # synchronize new frame
        self._lock = threading.Lock()
        self.start()

    def read(self):
        if self.ret:
            self._q.get()  # if reading too freq, then wait until new frame
            self.iframe += 1
        return self.ret, self.img
    
    def release(self):
        with self._lock:
            self._isopen = False
            self.vid.release()

    def run(self):
        while True:
            with self._lock:
                if self._isopen:
                    self.ret, self.img = self.vid.read()
                else:
                    break
            if not self._q.full():
                self._q.put(None)
            if not self.ret:
                break