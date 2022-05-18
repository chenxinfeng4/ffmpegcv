import subprocess
from subprocess import Popen, PIPE
import re
from collections import namedtuple
import xml.etree.ElementTree as ET


def get_info(video):
    cmd = 'ffprobe -v quiet -print_format xml -select_streams v:0 -show_format -show_streams "{}"'.format(video)
    output = subprocess.check_output(cmd, shell=True)
    root = ET.fromstring(output)
    assert (root[0].tag, root[0][0].tag) == ('streams', 'stream')
    vinfo = root[0][0].attrib

    VideoInfo = namedtuple('VideoInfo', ['width', 'height', 'fps', 'count', 
                                        'codec', 'duration'])
    outinfo = dict()
    outinfo['width'] = int(vinfo['width'])
    outinfo['height'] = int(vinfo['height'])
    outinfo['fps'] = eval(vinfo['r_frame_rate'])
    outinfo['count'] = int(vinfo['nb_frames'])
    outinfo['codec'] = vinfo['codec_name']
    outinfo['duration'] = float(vinfo['duration'])
    videoinfo = VideoInfo(**outinfo)

    return videoinfo


def get_num_NVIDIA_GPUs():
    cmd = 'ffmpeg -f lavfi -i nullsrc -c:v h264_nvenc -gpu list -f null -'
    p = Popen(cmd.split(), shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate(b"")
    p.stdin.close()
    p.stdout.close()
    p.terminate()
    pattern = re.compile(r'GPU #\d+ - < NVIDIA')
    nv_info = pattern.findall(stderr.decode())
    num_gpu = len(nv_info)
    return num_gpu


def encoder_to_nvidia(codec):
    codec_map = {'h264'      : 'h264_nvenc',
                 'hevc'      : 'hevc_nvenc'}

    if codec in codec_map:
        return codec_map[codec]
    elif codec in codec_map.values():
        return codec
    else:
        raise Exception('No NV codec found for %s' % codec)
    

def decoder_to_nvidia(codec):
    codec_map = {'av1'       : 'av1_cuvid',
                 'h264'      : 'h264_cuvid',
                 'x264'      : 'h264_cuvid',    
                 'hevc'      : 'hevc_cuvid',
                 'x265'      : 'hevc_cuvid',
                 'h265'      : 'hevc_cuvid',
                 'mjpeg'     : 'mjpeg_cuvid',
                 'mpeg1video': 'mpeg1_cuvid',
                 'mpeg2video': 'mpeg2_cuvid',
                 'mpeg4'     : 'mpeg4_cuvid',
                 'vp1'       : 'vp1_cuvid',
                 'vp8'       : 'vp8_cuvid',
                 'vp9'       : 'vp9_cuvid'}
    
    if codec in codec_map:
        return codec_map[codec]
    elif codec in codec_map.values():
        return codec
    else:
        raise Exception('No NV codec found for %s' % codec)


def run_async(args):
    quiet = True
    stderr_stream = subprocess.DEVNULL if quiet else None
    bufsize = -1
    return Popen(
        args, stdin=PIPE, stdout=PIPE, stderr=stderr_stream, 
        shell=isinstance(args, str), bufsize=bufsize
    )


def release_process(process):
    process.stdin.close()
    process.stdout.close()
    process.terminate()
    process.wait()
