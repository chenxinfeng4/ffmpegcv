import pycuda.driver as cuda
from pycuda.driver import PointerHolderBase
from pycuda.compiler import SourceModule
from pycuda import gpuarray
from ffmpegcv.ffmpeg_reader import FFmpegReader, FFmpegReaderNV
import numpy as np
from typing import Tuple


cuda.init()

mod_code = ("""
__device__ void yuv_to_rgb(unsigned char &y, unsigned char &u, unsigned char &v, 
                            float &r, float &g, float &b)
{
    // https://fourcc.org/fccyvrgb.php
    float Y_val = (float)1.164 * ((float)y - 16.0);
    float U_val = (float)u - 128.0;
    float V_val = (float)v - 128.0;
    r = Y_val + 1.596 * V_val;
    r = max(0.0, min(255.0, r)); // clamp(r, 0.0, 255.0);
    g = Y_val - 0.813 * V_val - 0.391 * U_val;
    g = max(0.0, min(255.0, g));
    b = Y_val + 2.018 * U_val;
    b = max(0.0, min(255.0, b));
}

__global__ void yuv420p_CHW_fp32(unsigned char *YUV420p, float *RGB24, int *width_, int *height_)
{
    int width = *width_; int height = *height_;
    // Get the thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if we are within the bounds of the image
    if (x >= width || y >= height)
        return;

    // Get the Y, U, and V values for this pixel
    auto w_h = width * height;
    auto yW = y * width;
    auto out_ind = yW + x;
    unsigned char *Y = YUV420p;
    unsigned char *U = YUV420p + w_h;
    unsigned char *V = U + w_h/4;
    int delta = (y/2)*(width/2)+x/2;

    yuv_to_rgb(Y[out_ind], U[delta], V[delta],
            RGB24[out_ind], RGB24[out_ind + w_h], RGB24[out_ind + w_h*2]);
}

__global__ void yuv420p_HWC_fp32(unsigned char *YUV420p, float *RGB24, int *width_, int *height_)
{
    int width = *width_; int height = *height_;
    // Get the thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if we are within the bounds of the image
    if (x >= width || y >= height)
        return;

    // Get the Y, U, and V values for this pixel
    auto w_h = width * height;
    auto yW = y * width;
    auto out_ind = yW + x;
    unsigned char *Y = YUV420p;
    unsigned char *U = YUV420p + w_h;
    unsigned char *V = U + w_h/4;
    int delta = (y/2)*(width/2)+x/2;
    auto ind = (yW + x)*3;

    yuv_to_rgb(Y[out_ind], U[delta], V[delta],
            RGB24[ind], RGB24[ind+1], RGB24[ind+2]);
}

__global__ void NV12_CHW_fp32(unsigned char *NV12, float *RGB24, int *width_, int *height_)
{
    int width = *width_; int height = *height_;
    // Get the thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if we are within the bounds of the image
    if (x >= width || y >= height)
        return;

    // Get the Y, U, and V values for this pixel
    auto w_h = width * height;
    auto yW = y * width;
    auto out_ind = yW + x;
    unsigned char *Y = NV12 + out_ind;
    unsigned char *UV = NV12 + w_h + (y / 2) * width + (x / 2) * 2;
    yuv_to_rgb(Y[0], UV[0], UV[1],
            RGB24[out_ind], RGB24[out_ind + w_h], RGB24[out_ind + w_h*2]);
}

__global__ void NV12_HWC_fp32(unsigned char *NV12, float *RGB24, int *width_, int *height_)
{
    int width = *width_; int height = *height_;
    // Get the thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if we are within the bounds of the image
    if (x >= width || y >= height)
        return;

    // Get the Y, U, and V values for this pixel
    auto w_h = width * height;
    auto yW = y * width;
    auto out_ind = yW + x;
    unsigned char *Y = NV12 + out_ind;
    unsigned char *UV = NV12 + w_h + (y / 2) * width + (x / 2) * 2;
    auto ind = (yW + x)*3;
    yuv_to_rgb(Y[0], UV[0], UV[1],
            RGB24[ind], RGB24[ind+1], RGB24[ind+2]);
}
"""
)


def load_cuda_module():
    mod = SourceModule(mod_code)
    converter = {('yuv420p', 'chw'): mod.get_function('yuv420p_CHW_fp32'),
                ('yuv420p', 'hwc'): mod.get_function('yuv420p_HWC_fp32'),
                ('nv12', 'chw'): mod.get_function('NV12_CHW_fp32'),
                ('nv12', 'hwc'): mod.get_function('NV12_HWC_fp32')}
    return converter


class Holder(PointerHolderBase):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor
        self.gpudata = tensor.data_ptr()

    def get_pointer(self):
        return self.tensor.data_ptr()

    def __index__(self):
        return self.gpudata


def tensor_to_gpuarray(tensor) -> gpuarray.GPUArray:
    '''Convert a :class:`torch.Tensor` to a :class:`pycuda.gpuarray.GPUArray`. The underlying
    storage will be shared, so that modifications to the array will reflect in the tensor object.
    Parameters
    ----------
    tensor  :   torch.Tensor
    Returns
    -------
    pycuda.gpuarray.GPUArray
    Raises
    ------
    ValueError
        If the ``tensor`` does not live on the gpu
    '''
    return gpuarray.GPUArray(tensor.shape, dtype=np.float32, gpudata=Holder(tensor))


class PycudaContext:
    def __init__(self, gpu=0):
        self.ctx = cuda.Device(gpu).make_context()

    def __enter__(self):
        if self.ctx is not None:
            self.ctx.push()
        return self

    def __exit__(self, *args, **kwargs):
        if self.ctx is not None:
            self.ctx.pop()

    def __del__(self):
        if self.ctx is not None:
            self.ctx.pop()


class FFmpegReaderCUDA(FFmpegReader):
    def __init__(self, vid:FFmpegReader, gpu=0, tensor_format='hwc'):
        assert vid.pix_fmt in ['yuv420p', 'nv12'], 'Set pix_fmt to yuv420p or nv12. Auto convert to rgb in cuda.'
        assert tensor_format in ['hwc', 'chw'], 'tensor_format must be hwc or chw'
        if isinstance(vid, FFmpegReaderNV) and vid.pix_fmt != 'nv12':
            print('--Tips: please use VideoCaptureNV(..., pix_fmt="NV12") for better performance.')
        elif not isinstance(vid, FFmpegReaderNV) and vid.pix_fmt == 'nv12':
            print('--Tips: please use VideoCapture(..., pix_fmt="yuv420p") for better performance.')

        # work like normal FFmpegReaderObj
        props_name = ['width', 'height', 'fps', 'count', 'codec', 'ffmpeg_cmd',
                      'size', 'out_numpy_shape', 'iframe', 
                      'duration', 'origin_width', 'origin_height']
        for name in props_name:
            setattr(self, name, getattr(vid, name, None))
        self.ctx = PycudaContext(gpu)
        self.pix_fmt = 'rgb24'
        self.vid = vid
        self.out_numpy_shape = (vid.height, vid.width, 3) if tensor_format == 'hwc' else (3, vid.height, vid.width)
        self.torch_device = f'cuda:{gpu}'
        self.block_size = (16, 16, 1)
        self.grid_size = ((self.width + self.block_size[0] - 1) // self.block_size[0],
                          (self.height + self.block_size[1] - 1) // self.block_size[1])
        self.process = None
        with self.ctx:
            self.converter = load_cuda_module()[(vid.pix_fmt, tensor_format)]

    def read(self, out_MAT:gpuarray.GPUArray=None) -> Tuple[bool, gpuarray.GPUArray]:
        self.waitInit = False
        ret, frame_yuv420p = self.vid.read()
        if not ret:
            return False, None
        
        with self.ctx:
            if out_MAT is None:
                out_MAT = gpuarray.empty(self.out_numpy_shape, dtype=np.float32)
            self.converter(cuda.In(frame_yuv420p), out_MAT, 
                        cuda.In(np.int32(self.width)), cuda.In(np.int32(self.height)),
                        block=self.block_size, grid=self.grid_size)
            return True, out_MAT
    
    def read_cudamem(self, out_MAT:cuda.DeviceAllocation=None) -> Tuple[bool, cuda.DeviceAllocation]:
        self.waitInit = False
        ret, frame_yuv420p = self.vid.read()
        if not ret:
            return False, None
            
        with self.ctx:
            if out_MAT is None:
                out_MAT = cuda.mem_alloc(int(np.prod(self.out_numpy_shape) * 
                                            np.dtype(np.float32).itemsize))
            self.converter(cuda.In(frame_yuv420p), out_MAT, 
                        cuda.In(np.int32(self.width)), cuda.In(np.int32(self.height)),
                        block=self.block_size, grid=self.grid_size)
            return True, out_MAT
    
    def read_torch(self, out_MAT=None):
        import torch
        self.waitInit = False
        ret, frame_yuv420p = self.vid.read()
        if not ret:
            return False, None
        
        with self.ctx:
            if out_MAT is None:
                out_MAT = torch.empty(self.out_numpy_shape, dtype=torch.float32, device=self.torch_device)
            tensor_proxy = tensor_to_gpuarray(out_MAT)
            self.converter(cuda.In(frame_yuv420p), tensor_proxy.gpudata, 
                        cuda.In(np.int32(self.width)), cuda.In(np.int32(self.height)),
                        block=self.block_size, grid=self.grid_size)
            return True, out_MAT

    def release(self):
        self.vid.release()
        super().release()
