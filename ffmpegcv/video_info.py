import subprocess
from subprocess import Popen, PIPE
import re
from collections import namedtuple
import json
import shlex
import platform

scan_the_whole = {"mkv", "flv", "ts"}  # scan the whole file to the count, slow

_is_windows = platform.system() == "Windows"
_inited_get_num_NVIDIA_GPUs = False
_inited_get_num_QSV_GPUs = False
_num_NVIDIA_GPUs = -1
_num_QSV_GPUs = -1


def get_info(video: str):
    do_scan_the_whole = video.split(".")[-1] in scan_the_whole

    def ffprobe_info_(do_scan_the_whole):
        use_count_packets = '-count_packets' if do_scan_the_whole else ''
        cmd = 'ffprobe -v quiet -print_format json=compact=1 -select_streams v:0 {}  -show_streams "{}"'.format(
            use_count_packets, video)

        output = subprocess.check_output(shlex.split(cmd), shell=False)
        data: dict = json.loads(output)
        vinfo: dict = data['streams'][0]
        return vinfo

    vinfo = ffprobe_info_(do_scan_the_whole)

    if "nb_frames" not in vinfo:
        do_scan_the_whole = True
        vinfo = ffprobe_info_(do_scan_the_whole)

    # VideoInfo = namedtuple(
    #     "VideoInfo", ["width", "height", "fps", "count", "codec", "duration", "pix_fmt"]
    # )
    VideoInfo = namedtuple(
        "VideoInfo", ["width", "height", "fps", "count", "codec", "duration"]
    )
    outinfo = dict()
    outinfo["width"] = int(vinfo["width"])
    outinfo["height"] = int(vinfo["height"])
    outinfo["fps"] = eval(vinfo["r_frame_rate"])
    outinfo["count"] = int(
        vinfo["nb_read_packets" if do_scan_the_whole else "nb_frames"]
    )  # nb_read_packets | nb_frames
    outinfo["codec"] = vinfo["codec_name"]
    # outinfo['pix_fmt'] = vinfo['pix_fmt']

    outinfo["duration"] = (
        float(vinfo["duration"])
        if "duration" in vinfo
        else outinfo["count"] / outinfo["fps"]
    )
    videoinfo = VideoInfo(**outinfo)

    return videoinfo


def get_info_precise(video: str):
    videoinfo = get_info(video)
    cmd = (
        "ffprobe -v error -select_streams v:0 -show_entries frame=pts_time "
        f' -of default=noprint_wrappers=1:nokey=1 -read_intervals 0%+#1,99999% "{video}"'
    )
    output = subprocess.check_output(
        shlex.split(cmd), shell=False, stderr=subprocess.DEVNULL
    )
    pts_start, *_, pts_end = output.decode().split()
    pts_start, pts_end = float(pts_start), float(pts_end)
    videoinfod = videoinfo._asdict()
    duration_ = pts_end - pts_start
    videoinfod["fps"] = round((videoinfo.count - 1) / duration_, 3)
    videoinfod["duration"] = round(duration_ + 1 / videoinfod["fps"], 3)
    videoinfo_precise = videoinfo.__class__(*videoinfod.values())
    return videoinfo_precise


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
        "vp9": "vp9_qsv",
    }

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
        "vp9": "vp9_qsv",
    }

    if codec in codec_map:
        return codec_map[codec]
    elif codec in codec_map.values():
        return codec
    else:
        raise Exception("No QSV codec found for %s" % codec)


def run_async(args):
    bufsize = -1
    if isinstance(args, str):
        args = shlex.split(args)
    return Popen(
        args,
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        shell=False,
        bufsize=bufsize,
    )

def run_async_reader(args):
    bufsize = -1
    if isinstance(args, str):
        args = shlex.split(args)

    return Popen(
        args,
        stdin=None,
        stdout=PIPE,
        stderr=subprocess.DEVNULL,
        shell=False,
        bufsize=bufsize,
    )


def release_process(process: Popen, forcekill=False):
    if hasattr(process, "stdin") and process.stdin is not None:
        process.stdin.close()
    if hasattr(process, "stdout") and process.stdout is not None:
        process.stdout.close()
    if hasattr(process, "stderr") and process.stderr is not None:
        process.stderr.close()
    if forcekill and hasattr(process, "terminate") and not _is_windows:
        process.terminate()
    if forcekill and hasattr(process, "wait"):
        process.wait()


def release_process_writer(process: Popen):
    if hasattr(process, "stdin"):
        process.stdin.close()
    if hasattr(process, "stdout"):
        process.stdout.close()
    if hasattr(process, "wait"):
        process.wait()
