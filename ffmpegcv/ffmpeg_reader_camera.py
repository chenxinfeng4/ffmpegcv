import numpy as np
import pprint
from .video_info import run_async, release_process
import re
import subprocess
from threading import Thread
from queue import Queue
import sys
import os
from types import TracebackType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from ffmpegcv.ffmpeg_reader import get_videofilter_cpu, get_outnumpyshape


class platform:
    win: int = 0
    linux: int = 1
    mac: int = 2
    other: int = 3


if sys.platform.startswith("linux"):
    this_os = platform.linux
elif sys.platform.startswith("win32"):
    this_os = platform.win
elif sys.platform.startswith("darwin"):
    this_os = platform.mac
else:
    this_os = platform.other


def _query_camera_divices_mac() -> Dict[int, Tuple[str, int]]:
    # run the command 'ffmpeg -f avfoundation -list_devices true -i "" '
    command = 'ffmpeg -hide_banner -f avfoundation -list_devices true -i ""'
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    # parse the output into a dictionary
    lines = stderr.decode("utf-8").split("AVFoundation audio devices:")[0].split("\n")
    id_device_map = dict()
    device_id_pattern = re.compile(r"\[[^\]]*?\] \[(\d*)\]")
    device_name_pattern = re.compile(r".*\] (.*)")
    for line in lines[1:-1]:
        id_match = re.search(device_id_pattern, line)
        name_match = re.search(device_name_pattern, line)
        if id_match is None or name_match is None:
            continue
        device_id = int(id_match.group(1))
        device_name = name_match.group(1)
        id_device_map[device_id] = (device_name, device_id)
    return id_device_map


def _query_camera_divices_win() -> Dict[int, Tuple[str, str]]:
    command = "ffmpeg -hide_banner -list_devices true -f dshow -i dummy"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    dshowliststr = stderr.decode("utf-8")
    dshowliststr = dshowliststr.split("DirectShow audio devices")[0]
    pattern = re.compile(r'\[*?\] *"([^"]*)"')
    matches = pattern.findall(dshowliststr)
    alternative_pattern = re.compile(r'Alternative name "(.*)"')
    alternative_names = alternative_pattern.findall(dshowliststr)
    assert len(matches) == len(alternative_names)
    id_device_map = {
        i: device for i, device in enumerate(zip(matches, alternative_names))
    }
    if len(id_device_map) == 0:
        print("No camera divice found")
    return id_device_map


def _query_camera_divices_linux() -> Dict[int, Tuple[str, str]]:
    "edit from https://github.com/p513817/python-get-cam-name/blob/master/get_cam_name.py"
    root = "/sys/class/video4linux"
    cam_info: List[Tuple[str, str]] = []

    for index in sorted([file for file in os.listdir(root)]):
        # Get Camera Name From /sys/class/video4linux/<video*>/name
        real_index_file = os.path.realpath("/sys/class/video4linux/" + index + "/index")
        with open(real_index_file, "r") as name_file:
            _index = name_file.read().rstrip()
            if _index != "0":
                continue

        real_file = os.path.realpath("/sys/class/video4linux/" + index + "/name")
        with open(real_file, "r") as name_file:
            name = name_file.read().rstrip()
            name = name.split(":")[0]

        # Setup Each Camera and Index ( video* )
        cam_info.append((name, "/dev/" + index))

    id_device_map = {i: vname for i, vname in enumerate(cam_info)}
    return id_device_map


def query_camera_devices(verbose_dict: bool = False) -> Dict[Any, Any]:
    result: Dict[Any, Any]
    if this_os == platform.linux:
        result = _query_camera_divices_linux()
    elif this_os == platform.mac:
        result = _query_camera_divices_mac()
    else:
        result = _query_camera_divices_win()
    if verbose_dict:
        dict_by_v0 = {v[0]: v for v in result.values()}
        dict_by_v1 = {v[1]: v for v in result.values()}
        result.update(dict_by_v0)  # type: ignore[arg-type]
        result.update(dict_by_v1)  # type: ignore[arg-type]

    return result


def _query_camera_options_mac(cam_id_name: Any) -> List[Dict[str, Any]]:
    print(
        "\033[33m"
        + "FFmpeg& FFmpegcv CAN NOT query the camera options in MAC platform."
        + "\033[0m"
    )
    print("Please find the proper parameter other way.")
    return [{"camsize_wh": None, "camfps": None}]


def _query_camera_options_linux(cam_id_name: Any) -> List[Dict[str, Any]]:
    print(
        "\033[33m"
        + "FFmpeg& FFmpegcv CAN NOT query the camera FPS in Linux platform."
        + "\033[0m"
    )
    print("Please find the proper parameter other way.")
    camname = query_camera_devices(verbose_dict=True)[cam_id_name][1]
    command = f'ffmpeg -hide_banner -f v4l2 -list_formats all -i "{camname}"'
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    lines = stderr.decode("utf-8").split("\n")
    lines = [l for l in lines if "v4l2" in l]
    outlist = []
    for line in lines:
        _, vcodec, *_, resolutions = line.split(":")
        vcodec = vcodec.strip()
        israw = "Raw" in line
        camcodec = None if israw else vcodec
        campix_fmt = vcodec if israw else None
        resolutions = resolutions.strip()

        camsize_wh_l = [tuple(map(int, r.split("x"))) for r in resolutions.split()]
        outlist.extend(
            [
                {
                    "camcodec": camcodec,
                    "campix_fmt": campix_fmt,
                    "camsize_wh": wh,
                    "camfps": None,
                }
                for wh in camsize_wh_l
            ]
        )
    return outlist


def _query_camera_options_win(cam_id_name: Union[int, str]) -> List[Dict[str, Any]]:
    if isinstance(cam_id_name, int):
        id_device_map = query_camera_devices()
        camname = id_device_map[cam_id_name][1]
    elif isinstance(cam_id_name, str):
        camname = cam_id_name
    else:
        raise ValueError("Not valid camname")
    command = f'ffmpeg -hide_banner -f dshow -list_options true -i video="{camname}"'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    dshowliststr = stderr.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    dshowlist = [s for s in dshowliststr.split("\n") if "fps=" in s]
    from collections import OrderedDict

    unique_dshowlist = list(OrderedDict.fromkeys(dshowlist))
    outlist = []
    for text in unique_dshowlist:
        cam_options = dict()
        cam_options["camcodec"] = (
            re.search(r"vcodec=(\w+)", text).group(1) if "vcodec" in text else None  # type: ignore[union-attr]
        )
        cam_options["campix_fmt"] = (
            re.search(r"pixel_format=(\w+)", text).group(1)  # type: ignore[union-attr]
            if "pixel_format" in text
            else None
        )
        camsize_wh = re.search(r"min s=(\w+)", text).group(1)  # type: ignore[union-attr]
        cam_options["camsize_wh"] = tuple(int(v) for v in camsize_wh.split("x"))
        camfps = float(re.findall(r"fps=([\d.]+)", text)[-1])
        cam_options["camfps"] = int(camfps) if int(camfps) == camfps else camfps
        outlist.append(cam_options)
    return outlist


def query_camera_options(cam_id_name: Any) -> List[Dict[str, Any]]:
    return {
        platform.linux: _query_camera_options_linux,
        platform.mac: _query_camera_options_mac,
        platform.win: _query_camera_options_win,
    }[this_os](cam_id_name)


class ProducerThread(Thread):
    def __init__(self, vid: "FFmpegReaderCAM", q: Queue) -> None:
        super(ProducerThread, self).__init__()
        self.vid = vid
        self.q = q

    def run(self) -> None:
        q = self.q
        while True:
            if not self.vid.isOpened():
                break
            ret, img = self.vid.read_()

            if q.full():
                q.get() # drop frames
            q.put((ret, img))  


class FFmpegReaderCAM:
    # Attributes initialized by the `VideoReader` factory methods.
    camname: Optional[str]
    camid: Optional[int]
    origin_width: int
    origin_height: int
    width: int
    height: int
    camfps: Optional[float]
    camcodec: Optional[str]
    campix_fmt: Optional[str]
    pix_fmt: str
    crop_width: Any
    crop_height: Any
    size: Any
    out_numpy_shape: Any
    ffmpeg_cmd: str
    process: Any
    step: int
    q: Queue

    def __init__(self) -> None:
        self.iframe: int = -1
        self._isopen: bool = True

    def __repr__(self) -> str:
        props = pprint.pformat(self.__dict__).replace("{", " ").replace("}", " ")
        return f"{self.__class__}\n" + props

    def __enter__(self) -> "FFmpegReaderCAM":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.release()

    def __iter__(self) -> "FFmpegReaderCAM":
        return self

    def __next__(self) -> np.ndarray:
        ret, img = self.read()
        if ret:
            return img
        else:
            raise StopIteration

    @staticmethod
    def VideoReader(
        cam_id_name: Union[int, str],
        pix_fmt: str,
        crop_xywh: Optional[Sequence[int]],
        resize: Optional[Sequence[int]],
        resize_keepratio: bool,
        resize_keepratioalign: Optional[str],
        camsize_wh: Optional[Sequence[int]] = None,
        camfps: Optional[float] = None,
        camcodec: Optional[str] = None,
        campix_fmt: Optional[str] = None,
        step: int = 1,
    ) -> "FFmpegReaderCAM":
        vid = FFmpegReaderCAM()
        if this_os == platform.mac:
            # use cam_id as the device marker
            if isinstance(cam_id_name, str):
                id_device_map = query_camera_devices()
                camname = cam_id_name
                id_device_map.update({v[0]: v for v in id_device_map.values()})
                camid = id_device_map[cam_id_name][1]
            else:
                camname = None
                camid = cam_id_name
        elif this_os == platform.linux:
            id_device_map = query_camera_devices(verbose_dict=True)
            camname = id_device_map[cam_id_name][-1]
            camid = None
        else:
            if isinstance(cam_id_name, int):
                id_device_map = query_camera_devices()
                camname = id_device_map[cam_id_name][1]
                camid = cam_id_name
            else:
                camname = cam_id_name
                camid = None

        vid.camname = camname
        vid.camid = camid

        if camsize_wh is None:
            cam_options = query_camera_options(camname)
            resolutions = [c["camsize_wh"] for c in cam_options]
            camsize_wh = max(resolutions, key=lambda x: sum(x))

        assert len(camsize_wh) == 2
        vid.origin_width, vid.origin_height = camsize_wh

        opt_camfps = f" -framerate {camfps} " if camfps else ""
        vid.camfps = camfps if camfps else None

        opt_camcodec_ = {
            platform.linux: "input_format",
            platform.mac: "",
            platform.win: "vcodec",
        }[this_os]
        opt_camcodec = f" -{opt_camcodec_} {camcodec} " if camcodec else ""
        vid.camcodec = camcodec if camcodec else None
        vid.pix_fmt = pix_fmt

        opt_campix_fmt_ = {
            platform.linux: "input_format",
            platform.mac: "pixel_format",
            platform.win: "pixel_format",
        }[this_os]
        opt_campix_fmt = f" -{opt_campix_fmt_} {campix_fmt} " if campix_fmt else ""
        vid.campix_fmt = campix_fmt if campix_fmt else None

        opt_camname = {
            platform.linux: f'"{camname}"',
            platform.win: f'video="{camname}"',
            platform.mac: f"{camid}:none",
        }[this_os]

        (vid.crop_width, vid.crop_height), (vid.width, vid.height), filteropt = get_videofilter_cpu(
                (vid.origin_width, vid.origin_height), pix_fmt, crop_xywh, resize, resize_keepratio, resize_keepratioalign)
        vid.size = (vid.width, vid.height)

        opt_driver_ = {
            platform.linux: "v4l2",
            platform.mac: "avfoundation",
            platform.win: "dshow",
        }[this_os]

        vid.ffmpeg_cmd = (
            f"ffmpeg -loglevel warning "
            f" -f {opt_driver_} "
            f" -video_size {vid.origin_width}x{vid.origin_height} "
            f" {opt_camfps} {opt_camcodec} {opt_campix_fmt} "
            f" -i {opt_camname} "
            f" {filteropt} -pix_fmt {pix_fmt} -f rawvideo pipe:"
        )

        vid.out_numpy_shape = get_outnumpyshape(vid.size, pix_fmt)
        vid.process = run_async(vid.ffmpeg_cmd)

        # producer
        assert step >= 1 and isinstance(step, int)
        vid.step = step
        vid.q = Queue(maxsize=30)
        producer = ProducerThread(vid, vid.q)
        producer.start()
        return vid

    def read_(self) -> Tuple[bool, Optional[np.ndarray]]:
        for i in range(self.step):
            in_bytes = self.process.stdout.read(np.prod(self.out_numpy_shape))
        if not in_bytes:
            self.release()
            return False, None

        self.iframe += 1
        img = None
        img = np.frombuffer(in_bytes, np.uint8).reshape(self.out_numpy_shape)
        return True, img

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # the producer thread buffers frames; drain them even after EOF/release
        if not self._isopen and self.q.empty():
            return False, None
        ret, img = self.q.get()
        return ret, img

    def isOpened(self) -> bool:
        return self._isopen
    
    def release(self) -> None:
        self._isopen = False
        release_process(self.process)

    def close(self) -> None:
        return self.release()
