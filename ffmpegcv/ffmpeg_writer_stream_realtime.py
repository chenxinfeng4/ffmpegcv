from .video_info import run_async
from ffmpegcv.ffmpeg_writer import FFmpegWriter


class FFmpegWriterStreamRT(FFmpegWriter):
    @staticmethod
    def VideoWriter(filename:str, codec, pix_fmt, bitrate=None) -> FFmpegWriter:
        assert codec in ['h264', 'libx264', 'x264']
        assert pix_fmt in ['bgr24', 'rgb24']
        assert filename.startswith('rtmp://'), 'currently only support rtmp'
        vid = FFmpegWriterStreamRT()
        vid.filename = filename
        vid.codec = codec
        vid.pix_fmt = pix_fmt
        vid.bitrate = bitrate
        return vid

    def _init_video_stream(self):
        bitrate_str = f'-b:v {self.bitrate} ' if self.bitrate else ''
        self.ffmpeg_cmd = (f'ffmpeg -loglevel warning ' 
                f'-f rawvideo -pix_fmt {self.pix_fmt} -s {self.width}x{self.height} -i pipe: '
                f'{bitrate_str} -f flv '
                f' -tune zerolatency -preset ultrafast '
                f' -c:v {self.codec} "{self.filename}"')
        self.process = run_async(self.ffmpeg_cmd)
