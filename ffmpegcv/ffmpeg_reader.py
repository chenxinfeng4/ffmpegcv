import numpy as np
import warnings
import subprocess
from .video_info import get_info, get_num_NVIDIA_GPUs, decoder_to_nvidia

def run_async(args):
    quiet = True
    stderr_stream = subprocess.DEVNULL if quiet else None
    return subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_stream, shell=True
    )


class FFmpegReader:
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
    def VideoReader(filename, codec, pix_fmt, resize, resize_keepratio):
        assert pix_fmt in ['rgb24', 'bgr24']

        vid = FFmpegReader()
        videoinfo = get_info(filename)
        vid.width = videoinfo.width
        vid.height = videoinfo.height
        vid.fps = fps = videoinfo.fps
        vid.count = videoinfo.count
        vid.origin_width, vid.origin_height = vid.width, vid.height

        codecopt = '-c:v ' + codec if codec else ''

        if resize:
            vid.width, vid.height = dst_width, dst_height = resize
            if resize_keepratio:
                re_width, re_height = vid.origin_width/(vid.origin_height / dst_height) , dst_height
                if re_width > dst_width:
                    re_width, re_height = dst_width, vid.origin_height/(vid.origin_width / dst_width)
                re_width, re_height = int(re_width), int(re_height)
                scaleopt = '-vf scale=%d:%d' % (re_width, re_height)
                xpading, ypading = (dst_width - re_width) // 2, (dst_height - re_height) // 2
                padopt = f'pad={dst_width}:{dst_height}:{xpading}:{ypading}:black'
                filteropt = f'{scaleopt},{padopt}'
            else:
                filteropt = '-vf scale=%d:%d' % (dst_width, dst_height)
        else:
            filteropt = ''

        args = (f'ffmpeg {codecopt} -r {fps} -i "{filename}" '
                f'{filteropt} -pix_fmt {pix_fmt} '
                f'-r {fps} -f rawvideo pipe:')
        vid.process = run_async(args)
        return vid

    def read(self):
        in_bytes = self.process.stdout.read(self.height * self.width * 3)
        if not in_bytes:
            return False, None
        img = None
        img = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
        return True, img

    def release(self):
        self.process.terminate()
        self.process.wait()


class FFmpegReaderNV(FFmpegReader):
    @staticmethod
    def VideoReader(filename, pix_fmt, crop_xywh, resize, resize_keepratio, gpu):
        assert pix_fmt in ['rgb24', 'bgr24']
        numGPU = get_num_NVIDIA_GPUs()
        assert numGPU>0, 'No GPU found'
        gpu = int(gpu) % numGPU if gpu is not None else 0
        assert len(resize) == 2, 'resize must be a tuple of (width, height)'
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

        if resize:
            vid.width, vid.height = dst_width, dst_height = resize
            if resize_keepratio:
                re_width, re_height = crop_w/(crop_h / dst_height) , dst_height
                if re_width > dst_width:
                    re_width, re_height = dst_width, crop_h/(crop_w / dst_width)
                re_width, re_height = int(re_width), int(re_height)
                scaleopt = f'-vf scale_cuda={re_width}:{re_height},hwdownload,format=nv12'
                xpading, ypading = (dst_width - re_width) // 2, (dst_height - re_height) // 2
                padopt = f'pad={dst_width}:{dst_height}:{xpading}:{ypading}:black'
                filteropt = f'{scaleopt},{padopt}'
            else:
                filteropt = f'-vf scale_cuda={dst_width}:{dst_height},hwdownload,format=nv12'
        else:
            filteropt = ''
            
        args = (f'ffmpeg -hwaccel cuda -hwaccel_device {vid.gpu} -hwaccel_output_format cuda '
                f' -vcodec {vid.codecNV} {cropopt} -r {vid.fps} -i "{filename}" '
                f' {filteropt} -pix_fmt {pix_fmt} -r {vid.fps} -f rawvideo pipe:')

        vid.process = run_async(args)
        return vid
