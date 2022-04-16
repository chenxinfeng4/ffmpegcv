import numpy as np
import subprocess
import warnings

def run_async(args):
    quiet = False
    stderr_stream = subprocess.DEVNULL if quiet else None
    return subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_stream, shell=True
    )

class FFmpegWriter:
    def __del__(self):
        if hasattr(self, 'process'):
            self.release()

    @staticmethod
    def VideoWriter(filename, codec, fps, frameSize, pix_fmt):
        if codec is None:
            codec = 'libx264'
        elif not isinstance(codec, str):
            codec = 'libx264'
            warnings.simplefilter('''
                Codec should be a string. Eg `h264`, `h264_nvenc`. 
                You may used CV2.VideoWriter_fourcc, which will be ignored.
                ''')

        width, height = frameSize
        vid = FFmpegWriter()
        vid.fps, vid.width, vid.height = fps, width, height
        args = (f'ffmpeg -y -loglevel warning ' 
                f'-f rawvideo -pix_fmt {pix_fmt} -s {width}x{height} -r {fps} -i pipe: '
                f'-r {fps} -c:v {codec} -pix_fmt yuv420p "{filename}"')

        vid.process = run_async(args)
        return vid

    def write(self, img):
        assert img.shape[:2] == (self.height, self.width)
        img = img.astype(np.uint8).tobytes()
        self.process.stdin.write(img)

    def release(self):
        self.process.stdin.close()
        self.process.wait()
