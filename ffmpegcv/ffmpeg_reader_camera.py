import numpy as np
import pprint
from .video_info import (run_async, release_process)
import re
import subprocess
from threading import Thread
from queue import Queue


def quary_camera_divices() -> dict:
    command = 'ffmpeg -hide_banner -list_devices true -f dshow -i dummy'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    dshowliststr = stderr.decode('utf-8')
    dshowliststr = dshowliststr.split('DirectShow audio devices')[0]
    pattern = re.compile(r'\[[^\]]*?\]  "([^"]*)"')
    matches = pattern.findall(dshowliststr)
    id_device_map = {i:device for i, device in enumerate(matches)}
    if len(id_device_map)==0:
        print('No camera divice found')
    return id_device_map


def quary_camera_options(cam_id_name) -> str:
    if isinstance(cam_id_name, int):
        id_device_map = quary_camera_divices()
        camname = id_device_map[cam_id_name]
    elif isinstance(cam_id_name, str):
        camname = cam_id_name
    else:
        raise ValueError('Not valid camname')
    command = f'ffmpeg -hide_banner -f dshow -list_options true -i video="{camname}"'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    dshowliststr = stderr.decode('utf-8').replace('\r\n','\n').replace('\r', '\n')
    dshowlist = [s for s in dshowliststr.split('\n') if 'fps=' in s]
    from collections import OrderedDict
    unique_dshowlist = list(OrderedDict.fromkeys(dshowlist))
    outlist = []
    for text in unique_dshowlist:
        cam_options = dict()
        cam_options['camcodec'] = re.search(r"vcodec=(\w+)", text).group(1) if 'vcodec' in text else None
        cam_options['campix_fmt'] = re.search(r"pixel_format=(\w+)", text).group(1) if 'pixel_format' in text else None
        camsize_wh = re.search(r"min s=(\w+)", text).group(1)
        cam_options['camsize_wh'] = tuple(int(v) for v in camsize_wh.split('x'))
        camfps = float(re.search(r"fps=(\w+)", text).group(1))
        cam_options['camfps'] = int(camfps) if round(camfps)==camfps else camfps
        outlist.append(cam_options)
    return outlist


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
                self.q.put_nowait((ret, img)) #drop frames 
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
    def VideoReader(cam_id_name, pix_fmt, crop_xywh,
                    resize, resize_keepratio, resize_keepratioalign,
                    camsize_wh=None, camfps=None, camcodec=None, 
                    campix_fmt=None, step=1):
        assert pix_fmt in ['rgb24', 'bgr24', 'yuv420p', 'nv12']

        vid = FFmpegReaderCAM()
        if isinstance(cam_id_name, int):
            id_device_map = quary_camera_divices()
            camname = id_device_map[cam_id_name]
        elif isinstance(cam_id_name, str):
            camname = cam_id_name
        else:
            raise ValueError('Not valid camname')
        vid.camname = camname

        if camsize_wh is None:
            cam_options = quary_camera_options(camname)
            resolutions = [c['camsize_wh'] for c in cam_options]
            camsize_wh = max(resolutions, key=lambda x: sum(x))

        assert len(camsize_wh)==2
        vid.width, vid.height = camsize_wh
        
        opt_camfps = f' -framerate {camfps} ' if camfps else ''
        vid.camfps = camfps if camfps else None
        opt_camcodec = f' -vcodec {camcodec} ' if camcodec else ''
        vid.camcodec = camcodec if camcodec else None
        opt_campix_fmt = f' -pixel_format {campix_fmt} ' if campix_fmt else ''
        vid.campix_fmt = campix_fmt if campix_fmt else None

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
                f' {opt_camfps} {opt_camcodec} {opt_campix_fmt} '
                f' -i video="{camname}" '
                f' {filteropt} -pix_fmt {pix_fmt} -f rawvideo pipe:')

        vid.size = (vid.width, vid.height)
        vid.pix_fmt = pix_fmt
        assert (not pix_fmt=='yuv420p') or (vid.height % 2 == 0 and vid.width % 2 == 0), 'yuv420p must be even'
        vid.out_numpy_shape = {'rgb24': (vid.height, vid.width, 3),
                            'bgr24': (vid.height, vid.width, 3),
                            'yuv420p': (int(vid.height * 1.5), vid.width),
                            'nv12': (int(vid.height * 1.5), vid.width),
                            }[pix_fmt]
        vid.process = run_async(args)

        vid.isopened = True
        
        # producer
        assert step>=1 and isinstance(step, int)
        vid.step=step
        vid.q = Queue(maxsize=30)
        producer = ProducerThread(vid, vid.q)
        producer.start()
        return vid

    def read_(self):
        for i in range(self.step):
            in_bytes = self.process.stdout.read(np.prod(self.out_numpy_shape))
        if not in_bytes:
            self.release()
            return False, None
        
        self.iframe += 1
        img = None
        img = np.frombuffer(in_bytes, np.uint8).reshape(self.out_numpy_shape)
        return True, img
    
    def read_gray(self):
        # It's an experimental function
        # return 'ret, img_gray'
        # img_gray: Height x Width x 1
        assert self.pix_fmt in ('nv12', 'yuv420p')
        ret, img = self.read()
        if not ret: return False, None
        assert img.shape==(int(self.height * 1.5), self.width)
        img_gray = img[:self.height, :, None]
        return True, img_gray

    def read(self):
        ret, img = self.q.get()
        return ret, img

    def release(self):
        self.isopened = False
        release_process(self.process)

    def close(self):
        return self.release()
