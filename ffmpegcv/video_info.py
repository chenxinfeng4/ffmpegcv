import subprocess
from subprocess import Popen, PIPE
import re
from collections import namedtuple
import xml.etree.ElementTree as ET

scan_the_whole = {'mkv', 'flv', 'ts'} #scan the whole file to the count, slow

_inited_get_num_NVIDIA_GPUs = False
_inited_get_num_QSV_GPUs = False
_num_NVIDIA_GPUs = -1
_num_QSV_GPUs = -1


def get_info(video:str):
    do_scan_the_whole = video.split('.')[-1] in scan_the_whole

    def ffprobe_info_(do_scan_the_whole):
        use_count_packets = '-count_packets' if do_scan_the_whole else ''
        cmd = 'ffprobe -v quiet -print_format xml -select_streams v:0 {} -show_format -show_streams "{}"'.format(use_count_packets, video)
        output = subprocess.check_output(cmd, shell=True)
        root = ET.fromstring(output)
        assert (root[0].tag, root[0][0].tag) == ("streams", "stream")
        vinfo = root[0][0].attrib
        return vinfo
    
    vinfo = ffprobe_info_(do_scan_the_whole)

    if 'nb_frames' not in vinfo:
        do_scan_the_whole = True
        vinfo = ffprobe_info_(do_scan_the_whole)

    VideoInfo = namedtuple(
        "VideoInfo", ["width", "height", "fps", "count", "codec", "duration"]
    )
    outinfo = dict()
    outinfo['width'] = int(vinfo['width'])
    outinfo['height'] = int(vinfo['height'])
    outinfo['fps'] = eval(vinfo['r_frame_rate'])
    outinfo['count'] = int(vinfo['nb_read_packets' if do_scan_the_whole
                         else 'nb_frames']) #nb_read_packets | nb_frames
    outinfo['codec'] = vinfo['codec_name']
    outinfo['duration'] = (float(vinfo['duration']) if 'duration' in vinfo
                            else outinfo['count']/outinfo['fps'])
    videoinfo = VideoInfo(**outinfo)

    return videoinfo


def get_num_NVIDIA_GPUs():
    global _num_NVIDIA_GPUs, _inited_get_num_NVIDIA_GPUs
    if not _inited_get_num_NVIDIA_GPUs:
        cmd = "ffmpeg -f lavfi -i nullsrc -c:v h264_nvenc -gpu list -f null -"
        p = Popen(cmd.split(), shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate(b"")
        p.stdin.close()
        p.stdout.close()
        p.terminate()
        pattern = re.compile(r"GPU #\d+ - < ")
        nv_info = pattern.findall(stderr.decode())
        _num_NVIDIA_GPUs = len(nv_info)
        _inited_get_num_NVIDIA_GPUs = True
    return _num_NVIDIA_GPUs


def get_num_QSV_GPUs():
    global _num_QSV_GPUs, _inited_get_num_QSV_GPUs
    if not _inited_get_num_QSV_GPUs:
        cmd = "ffmpeg -hide_banner -f qsv -h encoder=h264_qsv"
        p = Popen(cmd.split(), shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate(b"")
        _num_QSV_GPUs = 1 if len(stdout) > 50 else 0
        _inited_get_num_QSV_GPUs = True
    return _num_QSV_GPUs


def encoder_to_nvidia(codec):
    codec_map = {"h264": "h264_nvenc", "hevc": "hevc_nvenc"}

    if codec in codec_map:
        return codec_map[codec]
    elif codec in codec_map.values():
        return codec
    else:
        raise Exception("No NV codec found for %s" % codec)


def encoder_to_qsv(codec):
    codec_map = {
        "h264": "h264_qsv",
        "hevc": "hevc_qsv",
        "mjpeg": "mjpeg_qsv",
        "mpeg2video": "mpeg2_qsv",
        "vp9": "vp9_qsv"}

    if codec in codec_map:
        return codec_map[codec]
    elif codec in codec_map.values():
        return codec
    else:
        raise Exception("No QSV codec found for %s" % codec)


def decoder_to_nvidia(codec):
    codec_map = {
        "av1": "av1_cuvid",
        "h264": "h264_cuvid",
        "x264": "h264_cuvid",
        "hevc": "hevc_cuvid",
        "x265": "hevc_cuvid",
        "h265": "hevc_cuvid",
        "mjpeg": "mjpeg_cuvid",
        "mpeg1video": "mpeg1_cuvid",
        "mpeg2video": "mpeg2_cuvid",
        "mpeg4": "mpeg4_cuvid",
        "vp1": "vp1_cuvid",
        "vp8": "vp8_cuvid",
        "vp9": "vp9_cuvid",
    }

    if codec in codec_map:
        return codec_map[codec]
    elif codec in codec_map.values():
        return codec
    else:
        raise Exception("No NV codec found for %s" % codec)


def decoder_to_qsv(codec):
    codec_map = {
        "av1": "av1_qsv",
        "h264": "h264_qsv",
        "hevc": "hevc_qsv",
        "mjpeg": "mjpeg_qsv",
        "mpeg2video": "mpeg2_qsv",
        "vc1": "vc1_qsv",
        "vp8": "vp8_qsv",
        "vp9": "vp9_qsv"}

    if codec in codec_map:
        return codec_map[codec]
    elif codec in codec_map.values():
        return codec
    else:
        raise Exception("No QSV codec found for %s" % codec)


def run_async(args):
    quiet = True
    stderr_stream = subprocess.DEVNULL if quiet else None
    bufsize = -1
    return Popen(
        args,
        stdin=PIPE,
        stdout=PIPE,
        stderr=stderr_stream,
        shell=isinstance(args, str),
        bufsize=bufsize,
    )


def release_process(process):
    if hasattr(process, "stdin"):
        process.stdin.close()
    if hasattr(process, "stdout"):
        process.stdout.close()
    if hasattr(process, "terminate"):
        process.terminate()
    if hasattr(process, "wait"):
        process.wait()
