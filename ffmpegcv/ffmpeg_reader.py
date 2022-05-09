import numpy as np
import pprint
from .video_info import (run_async, get_info, get_num_NVIDIA_GPUs, 
                        decoder_to_nvidia, release_process)


class FFmpegReader:
    def __repr__(self):
        props = pprint.pformat(self.__dict__).replace('{',' ').replace('}',' ')
        return f'{self.__class__}\n'  + props
    
    def __len__(self):
        return self.count

    def __iter__(self):
        return self

    def __next__(self):
        ret, img = self.read()
        if ret:
            return img
        else:
            self.release()
            raise StopIteration

    @staticmethod
    def VideoReader(filename, codec, pix_fmt, crop_xywh,
                    resize, resize_keepratio, resize_keepratioalign):
        assert pix_fmt in ['rgb24', 'bgr24']

        vid = FFmpegReader()
        videoinfo = get_info(filename)
        vid.width = videoinfo.width
        vid.height = videoinfo.height
        vid.fps = fps = videoinfo.fps
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
        return vid

    def read(self):
        in_bytes = self.process.stdout.read(self.height * self.width * 3)
        if not in_bytes:
            self.release()
            return False, None
        img = None
        img = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
        return True, img

    def release(self):
        release_process(self.process)


class FFmpegReaderNV(FFmpegReader):
    @staticmethod
    def VideoReader(filename, pix_fmt, crop_xywh, 
                    resize, resize_keepratio, resize_keepratioalign, 
                    gpu):
        assert pix_fmt in ['rgb24', 'bgr24']
        numGPU = get_num_NVIDIA_GPUs()
        assert numGPU>0, 'No GPU found'
        gpu = int(gpu) % numGPU if gpu is not None else 0
        assert resize is None or len(resize) == 2, 'resize must be a tuple of (width, height)'
        videoinfo = get_info(filename)
        vid = FFmpegReaderNV()
        vid.origin_width = videoinfo.width
        vid.origin_height = videoinfo.height
        vid.fps = videoinfo.fps
        vid.count = videoinfo.count
        vid.width, vid.height = vid.origin_width, vid.origin_height
        vid.codec = videoinfo.codec
        vid.codecNV = decoder_to_nvidia(vid.codec)
        
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
        
        args = (f'ffmpeg -loglevel warning -hwaccel cuda -hwaccel_device {gpu} '
                f' -vcodec {vid.codecNV} {cropopt} {scaleopt} -r {vid.fps} -i "{filename}" '
                f' {filteropt} -pix_fmt {pix_fmt} -r {vid.fps} -f rawvideo pipe:')

        vid.process = run_async(args)
        vid.size = (vid.width, vid.height)
        return vid
