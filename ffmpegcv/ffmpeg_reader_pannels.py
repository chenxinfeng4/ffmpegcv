import os
import numpy as np
from ffmpegcv.ffmpeg_reader import FFmpegReader, get_outnumpyshape

from ffmpegcv.video_info import (
    get_info,
    run_async
)


class FFmpegReaderPannels(FFmpegReader):
    @staticmethod
    def VideoReader(
        filename:str,
        crop_xywh_l:list,
        codec,
        pix_fmt='bgr24',
        resize=None
    ):
        assert os.path.exists(filename) and os.path.isfile(
            filename
        ), f"{filename} not exists"
        assert pix_fmt in ["rgb24", "bgr24", "yuv420p", "nv12", "gray"]
        vid = FFmpegReaderPannels()
        crop_xywh_l = np.array(crop_xywh_l)
        vid.crop_xywh_l = crop_xywh_l
        videoinfo = get_info(filename)
        vid.origin_width = videoinfo.width
        vid.origin_height = videoinfo.height
        vid.fps = videoinfo.fps
        vid.count = videoinfo.count
        vid.duration = videoinfo.duration
        vid.pix_fmt = pix_fmt
        vid.codec = codec if codec else videoinfo.codec

        vid.crop_width_l = crop_xywh_l[:,2]
        vid.crop_height_l = crop_xywh_l[:,3]
        vid.size_l = crop_xywh_l[:,2:][:,::-1]
        vid.npannel = len(crop_xywh_l)
        vid.out_numpy_shape_l = [get_outnumpyshape(s[::-1], pix_fmt) for s in vid.size_l]
        if len(set(vid.crop_width_l)) == len(set(vid.crop_height_l)) == 1:
            vid.is_pannel_similar = True
            vid.crop_width = vid.crop_width_l[0]
            vid.crop_height = vid.crop_height_l[0]
            vid.size = vid.size_l[0]
            vid.out_numpy_shape = (vid.npannel, *vid.out_numpy_shape_l[0])
        else:
            vid.is_pannel_similar = False
            vid.crop_width = vid.crop_height = vid.size = None
            vid.out_numpy_shape = (np.sum(np.prod(s) for s in vid.out_numpy_shape_l),)

        VINSRCs =''.join(f'[VSRC{i}]' for i in range(vid.npannel))
        pix_fmtopt = ',extractplanes=y' if pix_fmt=='gray' else ''
        CROPs = ';'.join(f'[VSRC{i}]crop={w}:{h}:{x}:{y}{pix_fmtopt}[VPANEL{i}]'
                          for i, (x,y,w,h) in enumerate(vid.crop_xywh_l))
        filteropt = f' -filter_complex "split={vid.npannel}{VINSRCs};{CROPs}"'
        outmaps = ''.join(f' -map [VPANEL{i}] -pix_fmt {pix_fmt} -r {vid.fps} -f rawvideo pipe:'
                          for i in range(vid.npannel))

        vid.ffmpeg_cmd = (
            f"ffmpeg -loglevel warning "
            f' -r {vid.fps} -i "{filename}" '
            f" {filteropt} {outmaps}"
        )
        return vid
    
    def read(self):
        if self.waitInit:
            self.process = run_async(self.ffmpeg_cmd)
            self.waitInit = False
            
        in_bytes = self.process.stdout.read(np.prod(self.out_numpy_shape))
        if not in_bytes:
            self.release()
            return False, None
        self.iframe += 1
        img0 = np.frombuffer(in_bytes, np.uint8)
        if self.is_pannel_similar:
            img = img0.reshape(self.out_numpy_shape)
        else:
            img = []
            for out_numpy_shape in self.out_numpy_shape_l:
                nbuff = np.prod(out_numpy_shape)
                img.append(img0[:nbuff].reshape(out_numpy_shape))
                img0 = img0[nbuff:]

        return True, img
