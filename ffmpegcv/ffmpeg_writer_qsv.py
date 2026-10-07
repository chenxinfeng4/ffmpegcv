import warnings
from typing import Optional, Sequence

from ffmpegcv.ffmpeg_writer import FFmpegWriter
from .video_info import (
    run_async,
    get_num_QSV_GPUs,
    decoder_to_qsv,
)


class FFmpegWriterQSV(FFmpegWriter):
    gpu: int

    @staticmethod
    def VideoWriter(  # type: ignore[override]
        filename: str,
        codec: Optional[str],
        fps: float,
        pix_fmt: str,
        gpu: Optional[int],
        bitrate: Optional[str] = None,
        resize: Optional[Sequence[int]] = None,
        preset: Optional[str] = None
    ) -> "FFmpegWriterQSV":
        assert gpu is None or gpu == 0, 'Cannot use multiple QSV gpu yet.'
        numGPU = get_num_QSV_GPUs()
        assert numGPU
        gpu = int(gpu) % numGPU if gpu is not None else 0
        if codec is None:
            codec = "hevc_qsv"
        elif not isinstance(codec, str):
            codec = "hevc_qsv"
            warnings.warn(
                "Codec should be a string. Eg `h264`, `hevc`. "
                "You may used CV2.VideoWriter_fourcc, which will be ignored.",
                UserWarning,
            )
        else:
            codec = decoder_to_qsv(codec)
        assert resize is None or len(resize) == 2

        vid = FFmpegWriterQSV()
        vid.fps = fps
        vid.codec, vid.pix_fmt, vid.filename = codec, pix_fmt, filename
        vid.gpu = gpu
        vid.bitrate = bitrate
        vid.resize = resize
        vid.preset = preset
        return vid
