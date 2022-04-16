import numpy as np
import cv2
import subprocess


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
        vidcv = cv2.VideoCapture(filename)

        vid = FFmpegReader()
        vid.width = int(vidcv.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid.height = int(vidcv.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid.fps = fps = int(vidcv.get(cv2.CAP_PROP_FPS))
        vid.count = int(vidcv.get(cv2.CAP_PROP_FRAME_COUNT))
        vid.origin_width, vid.origin_height = vid.width, vid.height
        vidcv.release()
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