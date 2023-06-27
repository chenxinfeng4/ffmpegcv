# FFMPEGCV 读写视频，替代 OPENCV.
![Python versions](https://img.shields.io/badge/Python-3.6%2B-blue.svg)
[![PyPI version](https://img.shields.io/pypi/v/ffmpegcv.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/ffmpegcv/)
[![PyPI downloads](https://img.shields.io/pypi/dm/ffmpegcv.svg)](https://pypistats.org/packages/ffmpegcv)
![Code size](https://shields.io/github/languages/code-size/chenxinfeng4/ffmpegcv
)
![Last Commit](https://shields.io/github/last-commit/chenxinfeng4/ffmpegcv)

[English Version](./README.md) | 中文版本

ffmpegcv提供了基于ffmpeg的视频读取器和视频编写器，比cv2更快和更强大。

- ffmpegcv与open-cv具有**兼容**的API。
- ffmpegcv可以使用**GPU加速**编码和解码。
- ffmpegcv支持比open-cv更多的**视频编码器**。
- ffmpegcv原生支持**RGB**/BGR/灰度像素格式。
- ffmpegcv支持ROI（感兴趣区域）操作，可以对ROI进行**裁剪**、**调整大小**和**填充**。
总之，ffmpegcv与opencv的API非常相似。但它具有更多的编码器，并且不需要安装opencv。

## 功能：
- `VideoWriter`：写入视频文件。
- `VideoCapture`：读取视频文件。
- `VideoCaptureNV`：使用NVIDIA GPU读取视频文件。
- `VideoCaptureCAM`：读取摄像头。
- `VideoCaptureStream`：读取RTP/RTSP/RTMP/HTTP流。
- `noblock`：在后台读取视频文件（更快）,使用多进程。

## 安装
在使用ffmpegcv之前，您需要下载`ffmpeg`。
```
 #1A. LINUX: sudo apt install ffmpeg
 #1B. MAC: brew install ffmpeg
 #1C. WINDOWS: 下载ffmpeg并添加至环境变量的路径中
 
 conda install ffmpeg  #1D. CONDA
 pip install ffmpegcv
 ```

## 何时选择 `ffmpegcv` 而不是 `opencv`：
- 安装`opencv`比较困难。ffmpegcv仅需要`numpy`和`FFmpeg`，可以在Mac/Windows/Linux平台上工作。
- `opencv`包含太多的图像处理工具箱，而您只是想使用带GPU支持的简单视频/摄像头输入输出操作。
- `opencv`不支持`h264`/`h265`和其他视频编码器。
- 您想对视频/摄像头的感兴趣区域（ROI）进行**裁剪**、**调整大小**和**填充**操作。


## 基本示例
通过CPU读取视频，并通过GPU重写视频。
```python
vidin = ffmpegcv.VideoCapture(vfile_in)
vidout = ffmpegcv.VideoWriterNV(vfile_out, 'h264', vidin.fps)

with vidin, vidout:
    for frame in vidin:
        cv2.imshow('image', frame)
        vidout.write(frame)
```

读取摄像头。
```python
# 通过设备ID
cap = ffmpegcv.VideoCaptureCAM(0)
# 通过设备名称
cap = ffmpegcv.VideoCaptureCAM("Integrated Camera")
```

## GPU加速
- 仅支持NVIDIA显卡。
- 在**Windows**上完美支持。只需通过conda install安装ffmpeg即可支持NVIDIA加速。
- 在**Linux**上有些困难。原始的ffmpeg不支持NVIDIA加速，您需要自己重新编译ffmpeg。详见[链接](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/)
- 在**Google Colab**笔记本上无需繁琐操作（无需重新编译ffmpeg）即可运行。
- 在**MacOS**上无法进行NVIDIA加速，因为ffmpeg根本就不支持NVIDIA。

> 在CPU数量充足的条件下，GPU读取速度可能比CPU读取速度稍慢。在使用感兴趣区域（ROI）操作（裁剪、调整大小、填充）时，GPU优势更凸显。

## 编解码器

| 编解码器      | OpenCV读取器 | ffmpegcv-CPU读取器 | GPU读取器  | OpenCV写入器 | ffmpegcv-CPU写入器 | GPU写入器  |
| ----------- | ------------- | ---------------- | ---- | ------------- | ---------------- | ---- |
| h264        | √             | √                | √    | ×             | √                | √    |
| h265 (hevc) | 不确定        | √                | √    | ×             | √                | √    |
| mjpeg       | √             | √                | ×    | √             | √                | ×    |
| mpeg        | √             | √                | ×    | √             | √                | ×    |
| 其他      | 不确定        | ffmpeg -decoders | ×    | 不确定       | ffmpeg -encoders | ×    |

## 基准测试
*正在进行中...(遥遥无期)*


## 视频读取器
---
ffmpegcv与opencv在API上非常类似。
```python
# OpenCV
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

# 另一种写法
cap = ffmpegcv.VideoCapture(file)
nframe = len(cap)
for frame in cap:
    pass
cap.release()

# 更加Pythonic的写法，推荐使用
with ffmpegcv.VideoCapture(file) as cap:
    nframe = len(cap)
    for iframe, frame in enumerate(cap):
        if iframe>100: break
        pass
```

使用GPU加速解码。具体取决于视频编码格式。
h264_nvcuvid, hevc_nvcuvid ....
```python
cap_cpu = ffmpegcv.VideoCapture(file)
cap_gpu = ffmpegcv.VideoCapture(file, codec='h264_cuvid') # NVIDIA GPU0
cap_gpu0 = ffmpegcv.VideoCaptureNV(file)                # NVIDIA GPU0
cap_gpu1 = ffmpegcv.VideoCaptureNV(file, gpu=1)         # NVIDIA GPU1
```

使用`rgb24`代替`bgr24`。`gray`版本会更高效。
```python
cap = ffmpegcv.VideoCapture(file, pix_fmt='rgb24') # rgb24, bgr24, gray
ret, frame = cap.read()
plt.imshow(frame)
```

### 感兴趣区域（ROI）操作
您可以对视频进行裁剪、调整大小和填充。这些ROI操作中，`ffmpegcv-GPU` > `ffmpegcv-CPU` >> `opencv` 在性能上。

**裁剪**视频，比读取整个画布要快得多。
```python
cap = ffmpegcv.VideoCapture(file, crop_xywh=(0, 0, 640, 480))
```

将视频调整为给定大小的**大小**。
```python
cap = ffmpegcv.VideoCapture(file, resize=(640, 480))
```

**调整大小**并保持宽高比，使用黑色边框进行**填充**。
```python
cap = ffmpegcv.VideoCapture(file, resize=(640, 480), resize_keepratio=True)
```

对视频进行**裁剪**，然后进行**调整大小**。
```python
cap = ffmpegcv.VideoCapture(file, crop_xywh=(0, 0, 640, 480), resize=(512, 512))
```

## 视频写入器
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

# ffmpegcv，默认的编码器为'h264'在CPU上，'h265'在GPU上。
# 帧大小由第一帧决定
out = ffmpegcv.VideoWriter('outpy.mp4', None, 10)
out.write(frame1)
out.write(frame2)
out.release()

# 更加Pythonic的写法
with ffmpegcv.VideoWriter('outpy.mp4', None, 10) as out:
    out.write(frame1)
    out.write(frame2)
```

使用GPU加速编码。例如h264_nvenc，hevc_nvenc。
```python
out_cpu = ffmpegcv.VideoWriter('outpy.mp4', None, 10)
out_gpu0 = ffmpegcv.VideoWriterNV('outpy.mp4', 'h264', 10)        # NVIDIA GPU0
out_gpu1 = ffmpegcv.VideoWriterNV('outpy.mp4', 'hevc', 10, gpu=1) # NVIDIA GPU1
```

输入图像使用rgb24而不是bgr24。
```python
out = ffmpegcv.VideoWriter('outpy.mp4', None, 10, pix_fmt='rgb24')
out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

## 视频读取器和写入器
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

## 相机读取器
---
**实验性功能**。ffmpegcv提供了相机读取器。与VideoCapture读取器一致。

- VideoCaptureCAM旨在支持感兴趣区域（ROI）操作。在相机读取方面，Opencv比ffmpegcv更具吸引力。**对于大多数相机读取情况，我推荐使用Opencv**。
- ffmpegcv可以使用名称检索相机设备，使用`ffmpegcv.VideoCaptureCAM("Integrated Camera")`比使用`cv2.VideoCaptureCAM(0)`更易读。
- 如果后处理时间过长，VideoCaptureCAM将会出现卡顿和丢帧。VideoCaptureCAM会缓冲最近的帧。
- 即使没有读取视频帧，VideoCaptureCAM也会在后台不断工作。**请及时释放资源**。
- 在Windows上表现良好，在Linux和macOS上表现不完美。

```python
import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

# ffmpegcv，在Windows和Linux上
import ffmpegcv
cap = ffmpegcv.VideoCaptureCAM(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

# ffmpegcv 使用相机名称，在Windows和Linux上
cap = ffmpegcv.VideoCaptureCAM("Integrated Camera")

# ffmpegcv 使用相机路径（避免多个相机冲突）
cap = ffmpegcv.VideoCaptureCAM('@device_pnp_\\\\?\\usb#vid_2304&'
    'pid_oot#media#0001#{65e8773d-8f56-11d0-a3b9-00a0c9223196}'
    '\\global')

# ffmpegcv 使用具有ROI操作的相机
cap = ffmpegcv.VideoCaptureCAM("Integrated Camera", crop_xywh=(0, 0, 640, 480), resize=(512, 512), resize_keepratio=True)


```

**列出所有相机设备**
```python
from ffmpegcv.ffmpeg_reader_camera import query_camera_devices

devices = query_camera_devices()
print(devices)
```
>{0: ('Integrated Camera', '@device_pnp_\\\\?\\usb#vid_2304&pid_oot#media#0001#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global'),  
1: ('OBS Virtual Camera', '@device_sw_{860BB310-5D01-11D0-BD3B-00A0C911CE86}\\{A3FCE0F5-3493-419F-958A-ABA1250EC20B}')}


**设置相机的分辨率、帧率、视频编码/像素格式**

```python
from ffmpegcv.ffmpeg_reader_camera import query_camera_options

options = query_camera_options(0)  # 或者 query_camera_options("Integrated Camera") 
print(options)
cap = ffmpegcv.VideoCaptureCAM(0, **options[-1])
```
>[{'camcodec': 'mjpeg', 'campix_fmt': None, 'camsize_wh': (1280, 720), 'camfps': 60.0002}, {'camcodec': 'mjpeg', 'campix_fmt': None, 'camsize_wh': (640, 480), 'camfps': 60.0002}, {'camcodec': 'mjpeg', 'campix_fmt': None, 'camsize_wh': (1920, 1080), 'camfps': 60.0002}, {'camcodec': None, 'campix_fmt': 'yuyv422', 'camsize_wh': (1280, 720), 'camfps': 10}, {'camcodec': None, 'campix_fmt': 'yuyv422', 'camsize_wh': (640, 480), 'camfps': 30}, {'camcodec': None, 'campix_fmt': 'yuyv422', 'camsize_wh': (1920, 1080), 'camfps': 5}]

**已知问题**
1. VideoCaptureCAM在macOS上的体验不太流畅。你必须指定所有相机参数。而且query_camera_options不会给出任何建议。这是因为`ffmpeg`无法使用mac本机的`avfoundation`列出设备选项。
```python
# macOS需要提供完整参数。
cap = ffmpegcv.VideoCaptureCAM('FaceTime HD Camera', camsize_wh=(1280,720), camfps=30, campix_fmt='nv12')
```

2. 在Linux上VideoCaptureCAM无法列出FPS，因为`ffmpeg`无法使用Linux本机的`v4l2`模块查询设备的FPS。不过，让FPS为空也没问题。

## 流读取器
**实验性功能**。ffmpegcv提供了流读取器，与VideoFile读取器一致，更类似于相机。

- 支持`RTSP`、`RTP`、`RTMP`、`HTTP`、`HTTPS`流。
- 如果后处理时间过长，VideoCaptureStream会出现卡顿和丢帧。VideoCaptureCAM会缓冲最近的帧。
- 即使没有读取视频帧，VideoCaptureStream也会在后台不断工作。**请及时释放资源**。
- 这仍然是实验性功能。建议您使用opencv。

```python
# opencv
import cv2
stream_url = 'http://devimages.apple.com.edgekey.net/streaming/examples/bipbop_4x3/gear2/prog_index.m3u8'
cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print('无法打开流')
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

## FFmpegReaderNoblock
更快的读写取视频。利用多进程在后台自动准备帧，这样在读写当前帧时不会阻塞。这使得您的Python程序在CPU使用方面更高效。带来最大翻倍效率提升。

> ffmpegcv.VideoCapture(*args) -> ffmpegcv.noblock(ffmpegcv.VideoCapture, *args)
>
> ffmpegcv.VideoWriter(*args) -> ffmpegcv.noblock(ffmpegcv.VideoWriter, *args)

```python
# 代理任何 VideoCapture&VideoWriter 的参数和kargs
vid_noblock = ffmpegcv.noblock(ffmpegcv.VideoCapture, vfile, pix_fmt='rbg24')

# 这很快
def cpu_tense(): time.sleep(0.01)
for _ in tqdm.trange(1000):
    ret, img = vid_noblock.read() #当前图像已经被缓冲，不会占用时间
    cpu_tense()                   #同时，下一帧在后台缓冲

# 这很慢
vid = ffmpegcv.VideoCapture(vfile, pix_fmt='rbg24')
for _ in tqdm.trange(2000):
    ret, img = vid.read()         #此读取将阻塞CPU，占用时间
    cpu_tense()
```
