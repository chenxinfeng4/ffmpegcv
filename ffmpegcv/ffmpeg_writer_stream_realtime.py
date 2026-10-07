from typing import Optional, Sequence

from .video_info import run_async
from ffmpegcv.ffmpeg_writer import FFmpegWriter


class FFmpegWriterStreamRT(FFmpegWriter):
    @staticmethod
    def VideoWriter(  # type: ignore[override]
        filename: str,
        codec: str,
        pix_fmt: str,
        bitrate: Optional[str] = None,
        resize: Optional[Sequence[int]] = None,
        preset: Optional[str] = None
    ) -> "FFmpegWriterStreamRT":
        assert codec in ["h264", "libx264", "x264", "mpeg4"]
        assert pix_fmt in ["bgr24", "rgb24", "gray"]
        # assert filename.startswith('rtmp://'), 'currently only support rtmp'
        assert resize is None or len(resize) == 2
        vid = FFmpegWriterStreamRT()
        vid.filename = filename
        vid.codec = codec
        vid.pix_fmt = pix_fmt
        vid.bitrate = bitrate
        vid.resize = resize
        if preset is not None:
            print("Preset is auto configured in FFmpegWriterStreamRT")
        vid.preset = "ultrafast"
        return vid

    def _init_video_stream(self) -> None:
        assert self.resize is not None
        bitrate_str = f"-b:v {self.bitrate} " if self.bitrate else ""
        rtsp_str = f"-f rtsp" if self.filename.startswith("rtsp://") else ""
        filter_str = (
            ""
            if self.resize == self.size
            else f"-vf scale={self.resize[0]}:{self.resize[1]}"
        )
        self.ffmpeg_cmd = (
            f"ffmpeg -loglevel warning "
            f"-f rawvideo -pix_fmt {self.pix_fmt} -s {self.width}x{self.height} -i pipe: "
            f"{bitrate_str} -f flv -rtsp_transport tcp "
            f" -tune zerolatency -preset {self.preset} "
            f"{filter_str} {rtsp_str} "
            f' -c:v {self.codec} -g 50 -pix_fmt yuv420p "{self.filename}"'
        )
        self.process = run_async(self.ffmpeg_cmd)
