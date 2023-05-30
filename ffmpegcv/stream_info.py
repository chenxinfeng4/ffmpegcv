import subprocess
from collections import namedtuple
import xml.etree.ElementTree as ET


def get_info(stream_url):
    cmd = 'ffprobe -v quiet -print_format xml -select_streams v:0 -show_format -show_streams "{}"'.format(stream_url)
    output = subprocess.check_output(cmd, shell=True)
    root = ET.fromstring(output)
    assert (root[0].tag, root[0][0].tag) == ("streams", "stream")
    vinfo = root[0][0].attrib

    StreamInfo = namedtuple(
        "StreamInfo", ["width", "height", "fps", "count", "codec", "duration"]
    )
    outinfo = dict()
    outinfo['width'] = int(vinfo['width'])
    outinfo['height'] = int(vinfo['height'])
    outinfo['fps'] = eval(vinfo['r_frame_rate'])
    outinfo['count'] = None
    outinfo['codec'] = vinfo['codec_name']
    outinfo['duration'] = None
    streaminfo = StreamInfo(**outinfo)

    return streaminfo


if __name__ == '__main__':
    stream_url = 'http://devimages.apple.com.edgekey.net/streaming/examples/bipbop_4x3/gear2/prog_index.m3u8'
    streaminfo = get_info(stream_url)
    print(streaminfo)
