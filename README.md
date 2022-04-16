# FFMPEGCV is an alternative to OPENCV for video read and write.
The ffmpegcv provide faster and powerful VideoCapture and VideoWriter to cv2 in python.

- The ffmpegcv is api **compatible** to open-cv 
- The ffmpegcv can use **GPU accelerate** encoding and decoding. 
- The ffmpegcv support much more video **codecs** v.s. open-cv.
- The ffmpegcv support **RGB** & BGR format as you like.
- The ffmpegcv can **resize video** to specific size with/without **padding**.

In all, ffmpegcv is just similar to opencv api. But is faster and with more codecs.

## Video Reader
---
The ffmpegcv is just similar to opencv in api.
```python
# open cv
cap = cv2.VideoCapture(file)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    pass

# ffmpegcv
cap = ffmpegcv.VideoCapture(file)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    pass

# alternative, recommand
cap = ffmpegcv.VideoCapture(file)
nframe = len(cap)
for frame in cap:
    pass
```

Use GPU to accelerate decoding. It depends on the video codes.
h264_nvcuvid, hevc_nvcuvid ....
```python
cap_cpu = ffmpegcv.VideoCapture(file, codec='h264')
cap_gpu = ffmpegcv.VideoCapture(file, codec='h264_cuvid')
```

Use rgb24 instead of bgr24
```python
cap = ffmpegcv.VideoCapture(file, pix_fmt='rgb24')
ret, frame = cap.read()
plt.imshow(frame)
```

Resize the video to the given size
```python
cap = ffmpegcv.VideoCapture(file, resize=(640, 480))
```

Resize and keep the aspect ratio with black border padding.
```python
cap = ffmpegcv.VideoCapture(file, resize=(640, 480), resize_keepratio=True)
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

# ffmpegcv
out = ffmpegcv.VideoWriter('outpy.avi', None, 10, (w, h))
out.write(frame1)
out.write(frame2)
out.release()
```

Use GPU to accelerate encoding. Such as h264_nvenc, hevc_nvenc.
```python
out_cpu = ffmpegcv.VideoWriter('outpy.avi', None, 10, (w, h))
out_gpu = ffmpegcv.VideoWriter('outpy.avi', 'h264_nvenc', 10, (w, h))
```

Use rgb24 instead of bgr24
```python
out = ffmpegcv.VideoWriter('outpy.avi', None, 10, (w, h), pix_fmt='rgb24')
out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

## Video Reader and Writer
---
```python
vfile_in = 'A.mp4'
vfile_out = 'A_h264.mp4'
vidin = ffmpegcv.VideoCapture(vfile_in)
w, h = vidin.width, vidin.height
vidout = ffmpegcv.VideoWriter(vfile_out, 'h264_nvenc', vidin.fps, (w, h))

for frame in vidin:
    vidout.write(frame)

vidin.release()
vidout.release()
```
