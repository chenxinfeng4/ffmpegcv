# FFMPEGCV is an alternative to OPENCV for video read and write.
![Python versions](https://img.shields.io/badge/Python-3.6%2B-blue.svg)
[![PyPI version](https://img.shields.io/pypi/v/ffmpegcv.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/ffmpegcv/)
[![PyPI downloads](https://img.shields.io/pypi/dm/ffmpegcv.svg)](https://pypistats.org/packages/ffmpegcv)
![Code size](https://shields.io/github/languages/code-size/chenxinfeng4/ffmpegcv
)
![Last Commit](https://shields.io/github/last-commit/chenxinfeng4/ffmpegcv)

English Version | [中文版本](./README_CN.md)

The ffmpegcv provide Video Reader and Video Witer with ffmpeg backbone, which are faster and powerful than cv2.

- The ffmpegcv is api **compatible** to open-cv. 
- The ffmpegcv can use **GPU accelerate** encoding and decoding*.
- The ffmpegcv support much more video **codecs** v.s. open-cv.
- The ffmpegcv support **RGB** & BGR & GRAY format as you like.
- The ffmpegcv can support ROI operations.You can **crop**, **resize** and **pad** the ROI.

In all, ffmpegcv is just similar to opencv api. But is has more codecs and does't require opencv installed.

## Functions:
- `VideoWriter`: Write a video file.
- `VideoCapture`: Read a video file.
- `VideoCaptureNV`: Read a video file by NVIDIA GPU.
- `VideoCaptureQSV`: Read a video file by Intel QuickSync Video.
- `VideoCaptureCAM`: Read a camera.
- `VideoCaptureStream`: Read a RTP/RTSP/RTMP/HTTP stream.
- `noblock`: Read/Write a video file in background using mulitprocssing.

## Install
You need to download ffmpeg before you can use ffmpegcv.
```
 #1A. LINUX: sudo apt install ffmpeg
 #1B. MAC (No NVIDIA GPU): brew install ffmpeg
 #1C. WINDOWS: download ffmpeg and add to the path
 #1D. CONDA: conda install ffmpeg=6.0.0     #don't use the default 4.x.x version
 
 #2. python
 pip install ffmpegcv
 ```

## When should choose `ffmpegcv` other than `opencv`:
- The `opencv` is hard to install. The ffmpegcv only requires `numpy` and `FFmpeg`, works across Mac/Windows/Linux platforms.
- The `opencv` packages too much image processing toolbox. You just want a simple video/camero IO with GPU accessible.
- The `opencv` didn't support `h264`/`h265` and other video writers.
- You want to **crop**, **resize** and **pad** the video/camero ROI.

## Basic example
Read a video by CPU, and rewrite it by GPU.
```python
vidin = ffmpegcv.VideoCapture(vfile_in)
vidout = ffmpegcv.VideoWriterNV(vfile_out, 'h264', vidin.fps)

with vidin, vidout:
    for frame in vidin:
        cv2.imshow('image', frame)
        vidout.write(frame)
```

Read the camera.
```python
# by device ID
cap = ffmpegcv.VideoCaptureCAM(0)
# by device name
cap = ffmpegcv.VideoCaptureCAM("Integrated Camera")
```

## GPU Accelation
- Support **NVIDIA** card only, test in x86_64 only.
- Works in **Windows**, **Linux** and **Anaconda**.
- Works in the **Google Colab** notebook. 
- Infeasible in the **MacOS**. That ffmpeg didn't supports NVIDIA at all.

> \* The ffmegcv GPU reader is a bit slower than CPU reader, but much faster when use ROI operations (crop, resize, pad).

## Codecs

| Codecs      | OpenCV-reader | ffmpegcv-cpu-r     | gpu-r  | OpenCV-writer | ffmpegcv-cpu-w     | gpu-w  |
| ----------- | ------------- | ---------------- | ---- | ------------- | ---------------- | ---- |
| h264        | √             | √                | √    | ×             | √                | √    |
| h265 (hevc) | not sure      | √                | √    | ×             | √                | √    |
| mjpeg       | √             | √                | ×    | √             | √                | ×    |
| mpeg        | √             | √                | ×    | √             | √                | ×    |
| others      | not sure      | ffmpeg -decoders | ×    | not sure      | ffmpeg -encoders | ×    |

## Benchmark
*On the way...（maybe never）*


## Video Reader
---
The ffmpegcv is just similar to opencv in api.
```python
# open cv
import cv2
cap = cv2.VideoCapture(file)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    pass

# ffmpegcv
import ffmpegcv
cap = ffmpegcv.VideoCapture(file)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    pass
cap.release()

# alternative
cap = ffmpegcv.VideoCapture(file)
nframe = len(cap)
for frame in cap:
    pass
cap.release()

# more pythonic, recommand
with ffmpegcv.VideoCapture(file) as cap:
    nframe = len(cap)
    for iframe, frame in enumerate(cap):
        if iframe>100: break
        pass
```

Use GPU to accelerate decoding. It depends on the video codes.
h264_nvcuvid, hevc_nvcuvid ....
```python
cap_cpu = ffmpegcv.VideoCapture(file)
cap_gpu = ffmpegcv.VideoCapture(file, codec='h264_cuvid') #NVIDIA GPU0
cap_gpu0 = ffmpegcv.VideoCaptureNV(file)         #NVIDIA GPU0
cap_gpu1 = ffmpegcv.VideoCaptureNV(file, gpu=1)  #NVIDIA GPU1
cap_qsv = ffmpegcv.VideoCaptureQSV(file)         #Intel QSV, experimental
```

Use `rgb24` instead of `bgr24`. The `gray` version would be more efficient.
```python
cap = ffmpegcv.VideoCapture(file, pix_fmt='rgb24') #rgb24, bgr24, gray
ret, frame = cap.read()
plt.imshow(frame)
```

### ROI Operations
You can crop, resize and pad the video. These ROI operation is `ffmpegcv-GPU` > `ffmpegcv-CPU` >> `opencv`.

**Crop** video, which will be much faster than read the whole canvas.
```python
cap = ffmpegcv.VideoCapture(file, crop_xywh=(0, 0, 640, 480))
```

**Resize** the video to the given size.
```python
cap = ffmpegcv.VideoCapture(file, resize=(640, 480))
```

**Resize** and keep the aspect ratio with black border **padding**.
```python
cap = ffmpegcv.VideoCapture(file, resize=(640, 480), resize_keepratio=True)
```

**Crop** and then **resize** the video.
```python
cap = ffmpegcv.VideoCapture(file, crop_xywh=(0, 0, 640, 480), resize=(512, 512))
```

## Video Writer
---
```python
# cv2
out = cv2.VideoWriter('outpy.avi',
                       cv2.VideoWriter_fourcc('M','J','P','G'), 
                       10, 
                       (w, h))
out.write(frame1)
out.write(frame2)
out.release()

# ffmpegcv, default codec is 'h264' in cpu 'h265' in gpu.
# frameSize is decided by the size of the first frame
out = ffmpegcv.VideoWriter('outpy.mp4', None, 10)
out.write(frame1)
out.write(frame2)
out.release()

# more pythonic
with ffmpegcv.VideoWriter('outpy.mp4', None, 10) as out:
    out.write(frame1)
    out.write(frame2)
```


Use GPU to accelerate encoding. Such as h264_nvenc, hevc_nvenc.
```python
out_cpu = ffmpegcv.VideoWriter('outpy.mp4', None, 10)
out_gpu0 = ffmpegcv.VideoWriterNV('outpy.mp4', 'h264', 10)        #NVIDIA GPU0
out_gpu1 = ffmpegcv.VideoWriterNV('outpy.mp4', 'hevc', 10, gpu=1) #NVIDIA GPU1
out_qsv  = ffmpegcv.VideoWriterQSV('outpy.mp4', 'h264', 10)        #Intel QSV, experimental
```

Input image is rgb24 instead of bgr24
```python
out = ffmpegcv.VideoWriter('outpy.mp4', None, 10, pix_fmt='rgb24')
out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

## Video Reader and Writer
---
```python
import ffmpegcv
vfile_in = 'A.mp4'
vfile_out = 'A_h264.mp4'
vidin = ffmpegcv.VideoCapture(vfile_in)
vidout = ffmpegcv.VideoWriter(vfile_out, None, vidin.fps)

with vidin, vidout:
    for frame in vidin:
        vidout.write(frame)
```

## Camera Reader
---
**Experimental feature**. The ffmpegcv offers Camera reader. Which is consistent with VideoFiler reader. 

- The `VideoCaptureCAM` aims to support ROI operations. The Opencv will be general fascinating than ffmpegcv in camera read. **I recommand the opencv in most camera reading case**.
- The ffmpegcv can use name to retrieve the camera device. Use `ffmpegcv.VideoCaptureCAM("Integrated Camera")` is readable than `cv2.VideoCaptureCAM(0)`.
- The `VideoCaptureCAM` will be laggy and dropping frames if your post-process takes long time. The VideoCaptureCAM will buffer the recent frames.
- The `VideoCaptureCAM` is continously working on background even if you didn't read it. **Please release it in time**.
- Works perfect in Windows, not-perfect in Linux and macOS.

```python
import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

# ffmpegcv, in Windows&Linux
import ffmpegcv
cap = ffmpegcv.VideoCaptureCAM(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

# ffmpegcv use by camera name, in Windows&Linux
cap = ffmpegcv.VideoCaptureCAM("Integrated Camera")

# ffmpegcv use camera path if multiple cameras conflict
cap = ffmpegcv.VideoCaptureCAM('@device_pnp_\\\\?\\usb#vid_2304&'
    'pid_oot#media#0001#{65e8773d-8f56-11d0-a3b9-00a0c9223196}'
    '\\global')

# ffmpegcv use camera with ROI operations
cap = ffmpegcv.VideoCaptureCAM("Integrated Camera", crop_xywh=(0, 0, 640, 480), resize=(512, 512), resize_keepratio=True)


```

**List all camera devices**
```python
from ffmpegcv.ffmpeg_reader_camera import query_camera_devices

devices = query_camera_devices()
print(devices)
```
>{0: ('Integrated Camera', '@device_pnp_\\\\?\\usb#vid_2304&pid_oot#media#0001#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global'),  
1: ('OBS Virtual Camera', '@device_sw_{860BB310-5D01-11D0-BD3B-00A0C911CE86}\\{A3FCE0F5-3493-419F-958A-ABA1250EC20B}')}


**Set the camera resolution, fps, vcodec/pixel-format**

```python
from ffmpegcv.ffmpeg_reader_camera import query_camera_options

options = query_camera_options(0)  # or query_camera_options("Integrated Camera") 
print(options)
cap = ffmpegcv.VideoCaptureCAM(0, **options[-1])
```
>[{'camcodec': 'mjpeg', 'campix_fmt': None, 'camsize_wh': (1280, 720), 'camfps': 60.0002}, {'camcodec': 'mjpeg', 'campix_fmt': None, 'camsize_wh': (640, 480), 'camfps': 60.0002}, {'camcodec': 'mjpeg', 'campix_fmt': None, 'camsize_wh': (1920, 1080), 'camfps': 60.0002}, {'camcodec': None, 'campix_fmt': 'yuyv422', 'camsize_wh': (1280, 720), 'camfps': 10}, {'camcodec': None, 'campix_fmt': 'yuyv422', 'camsize_wh': (640, 480), 'camfps': 30}, {'camcodec': None, 'campix_fmt': 'yuyv422', 'camsize_wh': (1920, 1080), 'camfps': 5}]

**Known issues**
1. The VideoCaptureCAM didn't give a smooth experience in macOS. You must specify all the camera parameters. And the query_camera_options woun't give any suggestion. That's because the `ffmpeg` cannot list device options using mac native `avfoundation`. 
```python
# The macOS requires full argument.
cap = ffmpegcv.VideoCaptureCAM('FaceTime HD Camera', camsize_wh=(1280,720), camfps=30, campix_fmt='nv12')
```

2. The VideoCaptureCAM cann't list the FPS in linux. Because the `ffmpeg` cound't query the device's FPS using linux native `v4l2` module. However, it's just OK to let the FPS empty.


## Stream Reader
**Experimental feature**. The ffmpegcv offers Stream reader. Which is consistent with VideoFiler reader, and more similiar to the camera.
Becareful when using it.

- Support `RTSP`, `RTP`, `RTMP`, `HTTP`, `HTTPS` streams.
- The `VideoCaptureStream` will be laggy and dropping frames if your post-process takes long time. The VideoCaptureCAM will buffer the recent frames.
- The `VideoCaptureStream` is continously working on background even if you didn't read it. **Please release it in time**.
- It's still experimental. Recommand you to use opencv.


```python
# opencv
import cv2
stream_url = 'http://devimages.apple.com.edgekey.net/streaming/examples/bipbop_4x3/gear2/prog_index.m3u8'
cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print('Cannot open the stream')
    exit(-1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    pass

# ffmpegcv
import ffmpegcv
cap = ffmpegcv.VideoCaptureStream(stream_url)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    pass
```

## Noblock
A proxy to automatic prepare frames in backgroud, which does not block when reading&writing current frame (multiprocessing). This make your python program more efficient in CPU usage. Up to 2x boost.

> ffmpegcv.VideoCapture(*args) -> ffmpegcv.noblock(ffmpegcv.VideoCapture, *args)
> 
> ffmpegcv.VideoWriter(*args) -> ffmpegcv.noblock(ffmpegcv.VideoWriter, *args)
```python
#Proxy any VideoCapture&VideoWriter args and kargs
vid_noblock = ffmpegcv.noblock(ffmpegcv.VideoCapture, vfile, pix_fmt='rbg24')

# this is fast
def cpu_tense(): time.sleep(0.01)
for _ in tqdm.trange(1000):
    ret, img = vid_noblock.read() #current img is already buffered, take no time
    cpu_tense()                   #meanwhile, the next img is buffering in background

# this is slow
vid = ffmpegcv.VideoCapture(vfile, pix_fmt='rbg24')
for _ in tqdm.trange(1000):
    ret, img = vid.read()         #this read will block cpu, take time
    cpu_tense()
```