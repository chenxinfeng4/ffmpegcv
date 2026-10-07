import subprocess
import json
import shlex
from typing import Any, Dict, NamedTuple, Optional


class StreamInfo(NamedTuple):
    width: int
    height: int
    fps: float
    count: Optional[int]  # type: ignore[assignment]  # shadows tuple.count
    codec: str
    duration: Optional[float]


def get_info(stream_url: str, timeout: Optional[float] = None, duration_ms: int = 100) -> StreamInfo:
    rtsp_opt = '' if not stream_url.startswith('rtsp://') else '-rtsp_flags prefer_tcp -pkt_size 736 '
    analyze_duration = f'-analyzeduration {duration_ms * 1000}'
    cmd = (f'ffprobe -v quiet -print_format json=compact=1 {rtsp_opt} {analyze_duration} '
           f'-select_streams v:0 -show_format -show_streams "{stream_url}"')
    output = subprocess.check_output(shlex.split(cmd), shell=False, timeout=timeout)
    data: Dict[str, Any] = json.loads(output)
    vinfo: Dict[str, Any] = data['streams'][0]

    outinfo: Dict[str, Any] = dict()
    outinfo["width"] = int(vinfo["width"])
    outinfo["height"] = int(vinfo["height"])
    if 'avg_frame_rate' in vinfo and vinfo['avg_frame_rate']!= '0/0':
        outinfo["fps"] = eval(vinfo["avg_frame_rate"])
    else:
        outinfo["fps"] = eval(vinfo["r_frame_rate"])
    outinfo["count"] = None
    outinfo["codec"] = vinfo["codec_name"]
    outinfo["duration"] = None
    streaminfo = StreamInfo(**outinfo)

    return streaminfo


if __name__ == "__main__":
    stream_url = "http://devimages.apple.com.edgekey.net/streaming/examples/bipbop_4x3/gear2/prog_index.m3u8"
    streaminfo = get_info(stream_url)
    print(streaminfo)
