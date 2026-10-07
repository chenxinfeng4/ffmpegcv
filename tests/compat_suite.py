#!/usr/bin/env python
"""Compatibility / behavioral regression suite for ffmpegcv.

Runs against any Python >= 3.6 with numpy + ffmpeg installed.
Collects deterministic observations into JSON and prints them between markers.

Usage:
    python tests/compat_suite.py --out /tmp/result.json
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np

import ffmpegcv

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

RESULTS = {}
TMP = tempfile.mkdtemp(prefix="ffmpegcv_compat_")
W, H, FPS = 320, 240, 30


def fhash(arr):
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def run(key, fn):
    """Execute fn(), store its JSON-able result under key."""
    try:
        RESULTS[key] = {"ok": True, "value": fn()}
    except Exception as exc:  # noqa: BLE001 - we want every failure recorded
        RESULTS[key] = {"ok": False,
                        "error": "%s: %s" % (type(exc).__name__, str(exc).replace(TMP, "<tmp>"))}


def run_err(key, fn):
    """Execute fn() that is expected to fail; store the exception type."""
    try:
        fn()
        RESULTS[key] = {"ok": False, "error": "no exception raised"}
    except Exception as exc:  # noqa: BLE001
        RESULTS[key] = {"ok": True, "value": type(exc).__name__,
                        "msg": str(exc).replace(TMP, "<tmp>")[:120]}


def make_video(path, w=W, h=H, fps=FPS, dur=1):
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=%dx%d:rate=%d" % (w, h, fps), "-t", str(dur),
         "-pix_fmt", "yuv420p", path],
        check=True,
    )


def probe(path):
    out = subprocess.check_output([
        FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=width,height,codec_name,nb_frames,nb_read_packets",
        "-of", "json", path,
    ])
    st = json.loads(out)["streams"][0]
    return {
        "width": int(st.get("width", 0)),
        "height": int(st.get("height", 0)),
        "codec": st.get("codec_name"),
        "nb_frames": st.get("nb_frames"),
        "nb_read_packets": st.get("nb_read_packets"),
    }


def read_all(cap):
    frames = []
    for frame in cap:
        frames.append(frame)
    return frames


# ---------------------------------------------------------------- test cases

def test_info(src):
    def _info():
        vi = ffmpegcv.video_info.get_info(src)
        return {
            "width": vi.width, "height": vi.height, "fps": vi.fps,
            "count": vi.count, "codec": vi.codec, "duration": round(vi.duration, 3),
            "type": type(vi).__name__, "fields": list(vi._fields),
        }

    def _precise():
        vi = ffmpegcv.video_info.get_info_precise(src)
        return {"fps": round(vi.fps, 3), "count": vi.count,
                "type": type(vi).__name__, "fields": list(vi._fields)}

    def _stream():
        vi = ffmpegcv.stream_info.get_info(src)
        return {"width": vi.width, "height": vi.height, "fps": vi.fps,
                "count": vi.count, "codec": vi.codec, "duration": vi.duration,
                "fields": list(vi._fields)}

    def _maps():
        from ffmpegcv.video_info import (encoder_to_nvidia, encoder_to_qsv,
                                         decoder_to_nvidia, decoder_to_qsv)
        return {
            "encoder_to_nvidia(h264)": encoder_to_nvidia("h264"),
            "decoder_to_nvidia(hevc)": decoder_to_nvidia("hevc"),
            "decoder_to_nvidia(passthrough)": decoder_to_nvidia("h264_cuvid"),
            "encoder_to_qsv(hevc)": encoder_to_qsv("hevc"),
            "decoder_to_qsv(mjpeg)": decoder_to_qsv("mjpeg"),
        }

    run("info_video", _info)
    run("info_precise", _precise)
    run("info_stream", _stream)
    run("codec_maps", _maps)
    run_err("codec_map_invalid", lambda: ffmpegcv.video_info.decoder_to_nvidia("bogus"))
    run("gpu_counts", lambda: {"nvidia": ffmpegcv.get_num_NVIDIA_GPUs(),
                               "qsv": ffmpegcv.video_info.get_num_QSV_GPUs()})
    run("info_type_exported", lambda: {
        "video_info.VideoInfo": hasattr(ffmpegcv.video_info, "VideoInfo"),
        "stream_info.StreamInfo": hasattr(ffmpegcv.stream_info, "StreamInfo"),
    })


def test_capture_pixfmts(src):
    for fmt in ["bgr24", "rgb24", "gray", "yuv420p", "nv12"]:
        def _f(fmt=fmt):
            cap = ffmpegcv.VideoCapture(src, pix_fmt=fmt)
            frames = read_all(cap)
            cap.release()
            return {"n": len(frames), "shape": list(frames[0].shape),
                    "dtype": str(frames[0].dtype), "hash0": fhash(frames[0]),
                    "hash_last": fhash(frames[-1]), "len": len(cap),
                    "isOpened_after": cap.isOpened()}
        run("cap_pixfmt_%s" % fmt, _f)


def test_capture_filters(src):
    cases = {
        "crop": dict(crop_xywh=(10, 20, 100, 80)),
        "resize": dict(resize=(64, 48)),
        "resize_noratio": dict(resize=(64, 48), resize_keepratio=False),
        "crop_resize": dict(crop_xywh=(0, 0, 200, 100), resize=(64, 64)),
        "rgb_resize": dict(pix_fmt="rgb24", resize=(80, 60)),
        "gray_crop": dict(pix_fmt="gray", crop_xywh=(0, 0, 100, 100)),
    }
    for align in ["center", "topleft", "topright", "bottomleft", "bottomright"]:
        cases["keepratio_%s" % align] = dict(
            resize=(100, 100), resize_keepratio=True, resize_keepratioalign=align)

    for name, kw in cases.items():
        def _f(kw=kw):
            cap = ffmpegcv.VideoCapture(src, **kw)
            ret, frame = cap.read()
            cap.release()
            return {"ret": bool(ret), "shape": list(frame.shape), "hash": fhash(frame),
                    "size": list(cap.size)}
        run("cap_filter_%s" % name, _f)

    def _odd_crop():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cap = ffmpegcv.VideoCapture(src, crop_xywh=(1, 3, 101, 81))
            ret, frame = cap.read()
            cap.release()
        return {"warned": "Warning" in buf.getvalue(), "shape": list(frame.shape),
                "size": list(cap.size)}
    run("cap_odd_crop", _odd_crop)

    def _infile_options():
        cap = ffmpegcv.VideoCapture(src, infile_options="-threads 1")
        ret, frame = cap.read()
        cap.release()
        return {"ret": bool(ret), "shape": list(frame.shape)}
    run("cap_infile_options", _infile_options)


def test_capture_protocol(src):
    def _ctx_len():
        with ffmpegcv.VideoCapture(src) as cap:
            n = 0
            for _ in cap:
                n += 1
            return {"n": n, "len": len(cap), "size": list(cap.size),
                    "fps": cap.fps, "repr_has_class": "FFmpegReader" in repr(cap)}
    run("cap_context_manager", _ctx_len)

    def _deprecation():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cap = ffmpegcv.VideoCapture(src, codec="h264")
            cap.release()
        return {"deprecation": sorted({w.category.__name__ for w in caught})}
    run("cap_codec_deprecation", _deprecation)

    def _aliases():
        return {"VideoReader is VideoCapture": ffmpegcv.VideoReader is ffmpegcv.VideoCapture,
                "VideoReaderNV is VideoCaptureNV": ffmpegcv.VideoReaderNV is ffmpegcv.VideoCaptureNV,
                "VideoReaderStreamRT is VideoCaptureStreamRT":
                    ffmpegcv.VideoReaderStreamRT is ffmpegcv.VideoCaptureStreamRT,
                "VideoReaderPannels is VideoCapturePannels":
                    ffmpegcv.VideoReaderPannels is ffmpegcv.VideoCapturePannels}
    run("public_aliases", _aliases)

    def _missing():
        return "raises"
    run_err("cap_missing_file", lambda: ffmpegcv.VideoCapture(os.path.join(TMP, "nope.mp4")))


def test_writer(src):
    def _writer(name, **kw):
        out = os.path.join(TMP, "out_%s.%s" % (name, kw.pop("ext", "mp4")))
        n = kw.pop("n", 10)
        w = ffmpegcv.VideoWriter(out, **kw)
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:, :, 1] = 128
        for i in range(n):
            frame[0, 0, 0] = i
            w.write(frame)
        w.release()
        return {"isOpened": w.isOpened(), "probe": probe(out)}

    run("wr_h264_default", lambda: _writer("h264"))
    run("wr_mpeg4", lambda: _writer("mpeg4", codec="mpeg4"))
    run("wr_resize", lambda: _writer("resize", resize=(160, 120)))
    run("wr_rgb24", lambda: _writer("rgb24", pix_fmt="rgb24"))
    run("wr_bitrate", lambda: _writer("bitrate", bitrate="1M"))
    run("wr_preset", lambda: _writer("preset", preset="ultrafast"))

    def _fourcc():
        out = os.path.join(TMP, "out_fourcc.mp4")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            w = ffmpegcv.VideoWriter(out, 1234, 30)
            for _ in range(3):
                w.write(np.zeros((H, W, 3), np.uint8))
            w.release()
        return {"probe": probe(out),
                "warnings": sorted({c.category.__name__ for c in caught})}
    run("wr_fourcc_int", _fourcc)

    def _bad_shape():
        out = os.path.join(TMP, "out_bad.mp4")
        w = ffmpegcv.VideoWriter(out, None, 30)
        w.write(np.zeros((H, W, 3), np.uint8))
        try:
            w.write(np.zeros((H // 2, W, 3), np.uint8))
        finally:
            w.release()
        return "no error"
    run_err("wr_bad_shape", _bad_shape)

    def _ctx():
        out = os.path.join(TMP, "out_ctx.mp4")
        with ffmpegcv.VideoWriter(out, None, 30) as w:
            for _ in range(3):
                w.write(np.zeros((H, W, 3), np.uint8))
        return {"probe": probe(out)}
    run("wr_context_manager", _ctx)


def test_noblock(src):
    def _reader():
        nb = ffmpegcv.noblock(ffmpegcv.VideoCapture, src)
        n = 0
        while True:
            ret, frame = nb.read()
            if not ret:
                break
            n += 1
        nb.release()
        return {"n": n}
    run("noblock_reader", _reader)

    def _reader_first():
        nb = ffmpegcv.noblock(ffmpegcv.VideoCapture, src)
        ret, frame = nb.read()
        proc = getattr(nb, "process", None)
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join()
        return {"ret": bool(ret), "shape": list(frame.shape)}
    run("noblock_reader_first", _reader_first)

    def _writer():
        out = os.path.join(TMP, "out_noblock.mp4")
        nb = ffmpegcv.noblock(ffmpegcv.VideoWriter, out, None, 30)
        for _ in range(10):
            nb.write(np.zeros((H, W, 3), np.uint8))
        nb.release()
        return {"probe": probe(out)}
    run("noblock_writer", _writer)

    def _live_last():
        rll = ffmpegcv.ReadLiveLast(ffmpegcv.VideoCapture, src)
        ret, frame = rll.read()
        rll.release()
        return {"ret": bool(ret), "shape": list(frame.shape)}
    run("readlivelast", _live_last)

    def _unsupported():
        ffmpegcv.noblock(sorted)
    run_err("noblock_unsupported", _unsupported)


def test_pannels(src):
    def _similar():
        cap = ffmpegcv.VideoCapturePannels(
            src, [[0, 0, W // 2, H], [W // 2, 0, W // 2, H]])
        ret, frames = cap.read()
        cap.release()
        return {"ret": bool(ret), "shape": list(frames.shape)}
    run("pannels_similar", _similar)

    def _dissimilar():
        cap = ffmpegcv.VideoCapturePannels(
            src, [[0, 0, 100, 100], [0, 0, 200, 120], [10, 10, 60, 40]])
        ret, frames = cap.read()
        cap.release()
        return {"ret": bool(ret), "n": len(frames),
                "shapes": [list(f.shape) for f in frames]}
    run("pannels_dissimilar", _dissimilar)

    def _gray():
        cap = ffmpegcv.VideoCapturePannels(
            src, [[0, 0, 160, 120], [160, 0, 160, 120]], pix_fmt="gray")
        ret, frames = cap.read()
        cap.release()
        return {"ret": bool(ret), "shape": list(frames.shape)}
    run("pannels_gray", _gray)


def test_stream(src):
    def _stream_local():
        cap = ffmpegcv.VideoCaptureStream(src)
        ret, frame = cap.read()
        n = 1
        while cap.read()[0]:
            n += 1
        cap.release()
        return {"ret": bool(ret), "shape": list(frame.shape), "n": n}
    run("stream_local_file", _stream_local)

    def _stream_rt_local():
        cap = ffmpegcv.VideoCaptureStreamRT(src)
        ret, frame = cap.read()
        n = 1 if ret else 0
        while cap.read()[0]:
            n += 1
        cap.release()
        return {"n": n, "first_ret": bool(ret),
                "shape": list(frame.shape) if frame is not None else None}
    run("stream_rt_local_file", _stream_rt_local)

    # command construction with mocked ffprobe/ffmpeg (rtsp branch)
    def _mocked_cmd(module_name, cls_name, kwargs, url):
        import ffmpegcv.ffmpeg_reader_stream as smod
        import ffmpegcv.ffmpeg_reader_stream_realtime as rmod
        from collections import namedtuple
        Info = namedtuple("Info", "width height fps count codec duration")
        info = Info(320, 240, 30.0, None, "h264", None)
        target = smod if module_name == "stream" else rmod
        patches = []
        if hasattr(target, "ProducerThread"):
            class _P(object):
                def __init__(self, *a, **k):
                    pass

                def start(self):
                    pass
            patches.append(("ProducerThread", target.ProducerThread, _P))
        if hasattr(target, "run_async"):
            patches.append(("run_async", target.run_async, lambda cmd: None))
        patches.append(("get_info", target.get_info, lambda *a, **k: info))
        try:
            for name, _old, new in patches:
                setattr(target, name, new)
            cls = getattr(target, cls_name)
            vid = cls.VideoReader(url, **kwargs)
            cmd = vid.ffmpeg_cmd
        finally:
            for name, old, _new in patches:
                setattr(target, name, old)
        return cmd

    def _rtsp_stream():
        cmd = _mocked_cmd("stream", "FFmpegReaderStream",
                          dict(codec=None, pix_fmt="bgr24", crop_xywh=None, resize=None,
                               resize_keepratio=True, resize_keepratioalign="center",
                               infile_options=None, timeout=None),
                          "rtsp://example.com/live")
        return {"has_rtsp_flags": "prefer_tcp" in cmd, "has_vcodec": "-vcodec" in cmd}
    run("stream_rtsp_cmd", _rtsp_stream)

    def _rt_cmd():
        cmd = _mocked_cmd("rt", "FFmpegReaderStreamRT",
                          dict(codec=None, pix_fmt="bgr24", crop_xywh=None, resize=None,
                               resize_keepratio=True, resize_keepratioalign="center",
                               infile_options=None, timeout=None),
                          "rtsp://example.com/live")
        return {"has_rtsp_transport": "rtsp_transport tcp" in cmd,
                "has_low_delay": "low_delay" in cmd}
    run("stream_rt_rtsp_cmd", _rt_cmd)

    def _stream_attr_contract():
        import ffmpegcv.ffmpeg_reader_stream as smod
        from collections import namedtuple
        Info = namedtuple("Info", "width height fps count codec duration")
        info = Info(320, 240, 30.0, None, "h264", None)

        class _P(object):
            def __init__(self, *a, **k):
                pass

            def start(self):
                pass
        orig = (smod.ProducerThread, smod.run_async, smod.get_info)
        try:
            smod.ProducerThread = _P
            smod.run_async = lambda cmd: None
            smod.get_info = lambda *a, **k: info
            vid = smod.FFmpegReaderStream.VideoReader(
                "rtsp://example.com/live", None, "bgr24", None, None, True,
                "center", None, None)
        finally:
            smod.ProducerThread, smod.run_async, smod.get_info = orig
        return {a: hasattr(vid, a) for a in
                ("fps", "count", "duration", "codec", "isopened", "width", "size")}
    run("stream_attr_contract", _stream_attr_contract)


def test_camera_platforms():
    import ffmpegcv.ffmpeg_reader_camera as cam

    saved = (cam.this_os, cam.query_camera_devices, cam.query_camera_options,
             cam.run_async, cam.ProducerThread)

    class _P(object):
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

    def _build(os_id, cam_id_name, devices, options):
        cam.this_os = os_id
        cam.query_camera_devices = lambda verbose_dict=False: devices
        cam.query_camera_options = lambda name: options
        cam.run_async = lambda cmd: None
        cam.ProducerThread = _P
        vid = cam.FFmpegReaderCAM.VideoReader(
            cam_id_name, "bgr24", None, None, True, "center",
            camsize_wh=(640, 480), camfps=30, camcodec=None, campix_fmt=None)
        attrs = {a: hasattr(vid, a) for a in (
            "width", "height", "size", "origin_width", "origin_height",
            "crop_width", "crop_height", "camfps", "camcodec", "campix_fmt",
            "pix_fmt", "ffmpeg_cmd", "fps", "count", "duration", "codec", "isopened")}
        return {"cmd": vid.ffmpeg_cmd, "size": list(vid.size),
                "origin": [vid.origin_width, vid.origin_height], "attrs": attrs}

    try:
        linux = _build(cam.platform.linux, "cam0",
                       {"cam0": ("USB Camera", "/dev/video0")},
                       [{"camsize_wh": (640, 480), "camfps": None,
                         "camcodec": None, "campix_fmt": "mjpeg"}])
        run("camera_linux_cmd", lambda: {
            "ok": True, "has_v4l2": "-f v4l2" in linux["cmd"],
            "has_size": "-video_size 640x480" in linux["cmd"], "size": linux["size"]})

        win = _build(cam.platform.win, 0,
                     {0: ("USB Camera", "@device_pnp_xxx")},
                     [{"camsize_wh": (640, 480), "camfps": 30,
                       "camcodec": "mjpeg", "campix_fmt": None}])
        run("camera_win_cmd", lambda: {
            "ok": True, "has_dshow": "-f dshow" in win["cmd"],
            "has_video": 'video="USB Camera"' in win["cmd"]})

        mac = _build(cam.platform.mac, "FaceTime HD Camera",
                     {0: ("FaceTime HD Camera", 0)},
                     [{"camsize_wh": (640, 480), "camfps": None}])
        run("camera_mac_cmd", lambda: {
            "ok": True, "has_avfoundation": "-f avfoundation" in mac["cmd"],
            "has_input": "0:none" in mac["cmd"]})
        # documented contract: cameras expose no fps/count/duration/codec/isopened
        run("camera_attr_contract", lambda: linux["attrs"])
    finally:
        (cam.this_os, cam.query_camera_devices, cam.query_camera_options,
         cam.run_async, cam.ProducerThread) = saved

    run("camera_query_devices", lambda: sorted(
        str(k) for k in cam.query_camera_devices().keys())[:5])


def test_gpu_unavailable(src):
    run_err("nv_capture", lambda: ffmpegcv.VideoCaptureNV(src))
    run_err("nv_writer", lambda: ffmpegcv.VideoWriterNV(os.path.join(TMP, "o.mp4")))
    run_err("qsv_capture", lambda: ffmpegcv.VideoCaptureQSV(src))
    run_err("qsv_writer", lambda: ffmpegcv.VideoWriterQSV(os.path.join(TMP, "o2.mp4")))
    run_err("cuda_reader", lambda: ffmpegcv.toCUDA(ffmpegcv.VideoCapture(src)))


def test_public_surface():
    names = sorted(n for n in dir(ffmpegcv) if not n.startswith("_"))
    run("public_names", lambda: names)
    run("version", lambda: ffmpegcv.__version__)


def cleanup():
    shutil.rmtree(TMP, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = os.path.join(TMP, "in.mp4")
    make_video(src)

    try:
        for fn in (test_info, test_capture_pixfmts, test_capture_filters,
                   test_capture_protocol, test_writer, test_noblock, test_pannels,
                   test_stream, test_camera_platforms, test_gpu_unavailable,
                   test_public_surface):
            needs_src = fn not in (test_camera_platforms, test_public_surface)
            try:
                fn(src) if needs_src else fn()
            except Exception as exc:  # noqa: BLE001
                RESULTS["_section_%s" % fn.__name__] = {
                    "ok": False,
                    "error": "%s: %s" % (type(exc).__name__, str(exc).replace(TMP, "<tmp>"))}
    finally:
        cleanup()

    failures = sorted(k for k, v in RESULTS.items() if not v.get("ok"))
    payload = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "ffmpeg": subprocess.check_output([FFMPEG, "-version"]).decode().split()[2],
        "ffmpegcv": ffmpegcv.__version__,
        "results": RESULTS,
        "failed_expectations": failures,
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
    print("===COMPAT_JSON_BEGIN===")
    print(json.dumps(payload, indent=1, sort_keys=True))
    print("===COMPAT_JSON_END===")
    print("unexpected failures:", len(failures))
    for name in failures:
        print("  FAIL", name, RESULTS[name].get("error"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
