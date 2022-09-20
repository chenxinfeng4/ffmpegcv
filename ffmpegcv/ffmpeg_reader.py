import numpy as np
import pprint
import os
from .video_info import (run_async, get_info, get_num_NVIDIA_GPUs, 
                        decoder_to_nvidia, release_process)


class FFmpegReader:
    def __init__(self):
        self.iframe = -1

    def __repr__(self):
        props = pprint.pformat(self.__dict__).replace('{',' ').replace('}',' ')
        return f'{self.__class__}\n'  + props

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
    def VideoReader(filename, codec, pix_fmt, crop_xywh,
                    resize, resize_keepratio, resize_keepratioalign):
        assert os.path.exists(filename) and os.path.isfile(filename), f'{filename} not exists'
        assert pix_fmt in ['rgb24', 'bgr24', 'yuv420p']

        vid = FFmpegReader()
        videoinfo = get_info(filename)
        vid.width = videoinfo.width
        vid.height = videoinfo.height
        vid.fps = videoinfo.fps
        vid.count = videoinfo.count
        vid.origin_width, vid.origin_height = vid.width, vid.height
        vid.codec = codec if codec else videoinfo.codec
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
                f' -vcodec {vid.codec} -r {vid.fps} -i "{filename}" '
                f' {filteropt} -pix_fmt {pix_fmt} -r {vid.fps} -f rawvideo pipe:')

        vid.process = run_async(args)
        vid.size = (vid.width, vid.height)
        vid.pix_fmt = pix_fmt
        assert (not pix_fmt=='yuv420p') or (vid.height % 2 == 0 and vid.width % 2 == 0), 'yuv420p must be even'
        vid.out_numpy_shape = {'rgb24': (vid.height, vid.width, 3),
                            'bgr24': (vid.height, vid.width, 3),
                            'yuv420p': (int(vid.height * 1.5), vid.width)}[pix_fmt]
        return vid

    def read(self):
        in_bytes = self.process.stdout.read(np.prod(self.out_numpy_shape))
        if not in_bytes:
            self.release()
            return False, None
        self.iframe += 1
        img = None
        img = np.frombuffer(in_bytes, np.uint8).reshape(self.out_numpy_shape)
        return True, img

    def release(self):
        release_process(self.process)

    def close(self):
        return self.release()


class FFmpegReaderNV(FFmpegReader):
    def _get_opts(vid, videoinfo, crop_xywh, resize, 
                  resize_keepratio, resize_keepratioalign):
        vid.origin_width = videoinfo.width
        vid.origin_height = videoinfo.height
        vid.fps = videoinfo.fps
        vid.count = videoinfo.count
        vid.width, vid.height = vid.origin_width, vid.origin_height
        vid.codec = videoinfo.codec
        assert vid.origin_height %2 == 0, 'height must be even'
        assert vid.origin_width %2 == 0, 'width must be even'
        if crop_xywh:
            crop_w, crop_h = crop_xywh[2:]
            vid.width, vid.height = crop_w, crop_h
            x, y, w, h = crop_xywh
            top, bottom, left, right = y, vid.origin_height - (y + h), x, vid.origin_width - (x + w)  #crop length
            cropopt = f'-crop {top}x{bottom}x{left}x{right}'
        else:
            crop_w, crop_h = vid.origin_width, vid.origin_height
            cropopt = ''

        vid.crop_width, vid.crop_height = crop_w, crop_h

        if resize is None or resize == (vid.crop_width, vid.crop_height):
            scaleopt = ''
            filteropt = ''
        else:
            vid.width, vid.height = dst_width, dst_height = resize
            if not resize_keepratio:
                scaleopt = f'-resize {dst_width}x{dst_height}'
                filteropt = ''
            else:
                re_width, re_height = crop_w/(crop_h / dst_height) , dst_height
                if re_width > dst_width:
                    re_width, re_height = dst_width, crop_h/(crop_w / dst_width)
                re_width, re_height = int(re_width), int(re_height)
                scaleopt = f'-resize {re_width}x{re_height}'
                if resize_keepratioalign is None: resize_keepratioalign = 'center'
                paddings = {'center': ((dst_width - re_width) // 2, (dst_height - re_height) // 2),
                            'topleft': (0, 0),
                            'topright': (dst_width - re_width, 0),
                            'bottomleft': (0, dst_height - re_height), 
                            'bottomright': (dst_width - re_width, dst_height - re_height)}
                assert resize_keepratioalign in paddings, 'resize_keepratioalign must be one of "center"(mmpose), "topleft"(mmdetection), "topright", "bottomleft", "bottomright"'
                xpading, ypading = paddings[resize_keepratioalign]
                padopt = f'pad={dst_width}:{dst_height}:{xpading}:{ypading}:black'
                filteropt = f'-vf {padopt}'

        vid.size = (vid.width, vid.height)
        return cropopt, scaleopt, filteropt

    @staticmethod
    def VideoReader(filename, pix_fmt, crop_xywh, 
                    resize, resize_keepratio, resize_keepratioalign, 
                    gpu):
        assert os.path.exists(filename) and os.path.isfile(filename), f'{filename} not exists'
        assert pix_fmt in ['rgb24', 'bgr24', 'yuv420p']
        numGPU = get_num_NVIDIA_GPUs()
        assert numGPU>0, 'No GPU found'
        gpu = int(gpu) % numGPU if gpu is not None else 0
        assert resize is None or len(resize) == 2, 'resize must be a tuple of (width, height)'
        videoinfo = get_info(filename)
        vid = FFmpegReaderNV()
        cropopt, scaleopt, filteropt = vid._get_opts(videoinfo, crop_xywh, resize, 
            resize_keepratio, resize_keepratioalign)
        vid.codecNV = decoder_to_nvidia(vid.codec)

        args = (f'ffmpeg -loglevel warning -hwaccel cuda -hwaccel_device {gpu} '
                f' -vcodec {vid.codecNV} {cropopt} {scaleopt} -r {vid.fps} -i "{filename}" '
                f' {filteropt} -pix_fmt {pix_fmt} -r {vid.fps} -f rawvideo pipe:')

        vid.process = run_async(args)
        vid.pix_fmt = pix_fmt
        assert (not pix_fmt=='yuv420p') or (vid.height % 2 == 0 and vid.width % 2 == 0), 'yuv420p must be even'
        vid.out_numpy_shape = {'rgb24': (vid.height, vid.width, 3),
                            'bgr24': (vid.height, vid.width, 3),
                            'yuv420p': (int(vid.height * 1.5), vid.width)}[pix_fmt]
        return vid
