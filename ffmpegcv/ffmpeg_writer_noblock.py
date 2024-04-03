from multiprocessing import Queue, Process, Array
import numpy as np
from .ffmpeg_writer import FFmpegWriter

NFRAME = 10

class FFmpegWriterNoblock(FFmpegWriter):
    def __init__(self,
                 vwriter_fun,
                 *vwriter_args, **vwriter_kwargs):
        super().__init__()
        vid:FFmpegWriter = vwriter_fun(*vwriter_args, **vwriter_kwargs)
        vid.release()

        props_name = ['width', 'height', 'fps', 'codec', 'pix_fmt',
                      'filename', 'size', 'bitrate']
        for name in props_name:
            setattr(self, name, getattr(vid, name, None))

        self.vwriter_fun = vwriter_fun
        self.vwriter_args = vwriter_args
        self.vwriter_kwargs = vwriter_kwargs
        self.q = Queue(maxsize=(NFRAME-2)) #buffer index, gluttonous snake NO biting its own tail
        self.waitInit = True
        self.process = None

    def write(self, img:np.ndarray):
        if self.waitInit:
            if self.size is None:
                self.size = (img.shape[1], img.shape[0])
            else:
                assert tuple(self.size) == (img.shape[1], img.shape[0])
            self.in_numpy_shape = img.shape
            self._init_share_array()
            process = Process(target=child_process, 
                            args=(self.shared_array, self.q, self.in_numpy_shape,
                                self.vwriter_fun, self.vwriter_args, self.vwriter_kwargs))
            process.start()
            self.process = process
            self.width, self.height = self.size
            self.waitInit = False

        self.iframe += 1
        data_id = self.iframe % NFRAME
        self.np_array[data_id] = img
        self.q.put(data_id)

    def _init_share_array(self):
        self.shared_array = Array('b', int(NFRAME*np.prod(self.in_numpy_shape)))
        self.np_array = np.frombuffer(self.shared_array.get_obj(), dtype=np.uint8).reshape((NFRAME,*self.in_numpy_shape))

    def release(self):
        if self.process is not None and self.process.is_alive():
            self.q.put(None)
            self.process.join()


def child_process(shared_array, q:Queue, in_numpy_shape, vwriter_fun, vwriter_args, vwriter_kwargs):
    vid = vwriter_fun(*vwriter_args, **vwriter_kwargs)
    np_array = np.frombuffer(shared_array.get_obj(), dtype=np.uint8).reshape((NFRAME,*in_numpy_shape))
    with vid:
        while True:
            data_id = q.get()
            if data_id is None:
                break
            else:
                img = np_array[data_id]
                vid.write(img)
