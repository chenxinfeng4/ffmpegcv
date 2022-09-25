import numpy as np
import pprint
import os
from .video_info import (run_async, release_process)

from threading import Thread
from queue import Queue

class ProducerThread(Thread):
    def __init__(self, vid, q):
        super(ProducerThread,self).__init__()
        self.vid = vid
        self.q = q

    def run(self):
        while True:
            if not self.vid.isopened:
                break
            ret, img = self.vid.read_()

            try:
                self.q.put_nowait((ret, img)) #give up frames 
            except Exception:
                pass
            continue


class FFmpegReaderCAM:
    def __init__(self):
        self.iframe = -1

    def __repr__(self):
        props = pprint.pformat(self.__dict__).replace('{',' ').replace('}',' ')
        return f'{self.__class__}\n'  + props

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.release()

    def __iter__(self):
        return self

    def __next__(self):
        ret, img = self.read()
        if ret:
            return img
        else:
            raise StopIteration

    @staticmethod
    def VideoReader(camname, camsize, pix_fmt, crop_xywh,
                    resize, resize_keepratio, resize_keepratioalign):
        assert pix_fmt in ['rgb24', 'bgr24', 'yuv420p']

        vid = FFmpegReaderCAM()
        assert len(camsize)==2
        vid.width, vid.height = camsize
        vid.origin_width, vid.origin_height = vid.width, vid.height
        if crop_xywh:
            crop_w, crop_h = crop_xywh[2:]
            vid.width, vid.height = crop_w, crop_h
            x, y, w, h = crop_xywh
            cropopt = f'crop={w}:{h}:{x}:{y}'
        else:
            crop_w, crop_h = vid.origin_width, vid.origin_height
            cropopt = ''

        vid.crop_width, vid.crop_height = crop_w, crop_h

        if resize is None or resize == (vid.crop_width, vid.crop_height):
            scaleopt = ''
            padopt = ''
        else:
            vid.width, vid.height = dst_width, dst_height = resize
            if not resize_keepratio:
                scaleopt = f'scale={dst_width}x{dst_height}'
                padopt = ''
            else:
                re_width, re_height = crop_w/(crop_h / dst_height) , dst_height
                if re_width > dst_width:
                    re_width, re_height = dst_width, crop_h/(crop_w / dst_width)
                re_width, re_height = int(re_width), int(re_height)
                scaleopt = f'scale={re_width}x{re_height}'
                if resize_keepratioalign is None: resize_keepratioalign = 'center'
                paddings = {'center': ((dst_width - re_width) // 2, (dst_height - re_height) // 2),
                            'topleft': (0, 0),
                            'topright': (dst_width - re_width, 0),
                            'bottomleft': (0, dst_height - re_height), 
                            'bottomright': (dst_width - re_width, dst_height - re_height)}
                assert resize_keepratioalign in paddings, 'resize_keepratioalign must be one of "center"(mmpose), "topleft"(mmdetection), "topright", "bottomleft", "bottomright"'
                xpading, ypading = paddings[resize_keepratioalign]
                padopt = f'pad={dst_width}:{dst_height}:{xpading}:{ypading}:black'
        
        if any([cropopt, scaleopt, padopt]):
            filterstr = ','.join(x for x in [cropopt, scaleopt, padopt] if x)
            filteropt = f'-vf {filterstr}'
        else:
            filteropt = ''

        args = (f'ffmpeg -loglevel warning '
                f' -f dshow -video_size {vid.origin_width}x{vid.origin_height} '
                f' -re -i video="{camname}" '
                f' {filteropt} -pix_fmt {pix_fmt} -f rawvideo pipe:')

        vid.size = (vid.width, vid.height)
        vid.pix_fmt = pix_fmt
        assert (not pix_fmt=='yuv420p') or (vid.height % 2 == 0 and vid.width % 2 == 0), 'yuv420p must be even'
        vid.out_numpy_shape = {'rgb24': (vid.height, vid.width, 3),
                            'bgr24': (vid.height, vid.width, 3),
                            'yuv420p': (int(vid.height * 1.5), vid.width)}[pix_fmt]
        vid.process = run_async(args)

        vid.isopened = True
        
        # producer
        vid.q = Queue(maxsize=2)
        producer = ProducerThread(vid, vid.q)
        producer.start()
        return vid

    def read_(self):
        in_bytes = self.process.stdout.read(np.prod(self.out_numpy_shape))
        if not in_bytes:
            self.release()
            return False, None
        self.iframe += 1
        img = None
        img = np.frombuffer(in_bytes, np.uint8).reshape(self.out_numpy_shape)
        return True, img

    def read(self):
        ret, img = self.q.get()
        return ret, img

    def release(self):
        self.isopened = False
        release_process(self.process)

    def close(self):
        return self.release()

