import warnings
from ffmpegcv.ffmpeg_writer import FFmpegWriter
from .video_info import (
    run_async,
    get_num_QSV_GPUs,
    decoder_to_qsv,
)


class FFmpegWriterQSV(FFmpegWriter):
    @staticmethod
    def VideoWriter(filename, codec, fps, frameSize, pix_fmt, gpu, bitrate=None):
        assert gpu is None or gpu == 0, 'Cannot use multiple QSV gpu yet.'
        numGPU = get_num_QSV_GPUs()
        assert numGPU
        gpu = int(gpu) % numGPU if gpu is not None else 0
        if codec is None:
            codec = "hevc_qsv"
        elif not isinstance(codec, str):
            codec = "hevc_qsv"
            warnings.simplefilter(
                """
                Codec should be a string. Eg `h264`, `hevc`. 
                You may used CV2.VideoWriter_fourcc, which will be ignored.
                """
            )
        else:
            codec = decoder_to_qsv(codec)

        vid = FFmpegWriterQSV()
        vid.fps, vid.size = fps, frameSize
        vid.width, vid.height = vid.size if vid.size else (None, None)
        vid.codec, vid.pix_fmt, vid.filename = codec, pix_fmt, filename
        vid.gpu = gpu
        vid.waitInit = True
        vid.bitrate = bitrate
        return vid

    def _init_video_stream(self):
        bitrate_str = f'-b:v {self.bitrate} ' if self.bitrate else ''
        self.ffmpeg_cmd = (f'ffmpeg -y -loglevel warning '
            f'-f rawvideo -pix_fmt {self.pix_fmt} -s {self.width}x{self.height} -r {self.fps} -i pipe: '
            f' {bitrate_str} '
            f'-r {self.fps} -c:v {self.codec} -pix_fmt yuv420p "{self.filename}"')
        self.process = run_async(self.ffmpeg_cmd)