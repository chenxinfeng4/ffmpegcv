import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.driver import PointerHolderBase
from pycuda.compiler import SourceModule
from pycuda import gpuarray
from ffmpegcv.ffmpeg_reader import FFmpegReader, FFmpegReaderNV
import numpy as np


mod = SourceModule("""
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
    unsigned char Y_val = Y[out_ind];
    float U_val = U[(y / 2) * (width / 2) + x / 2] - 128.0;
    float V_val = V[(y / 2) * (width / 2) + x / 2] - 128.0;

    // Convert the YUV values to RGB values
    float R_val = Y_val + 1.403 * V_val;
    float G_val = Y_val - 0.344 * U_val - 0.714 * V_val;
    float B_val = Y_val + 1.770 * U_val;

    // Clamp the RGB values to the range [0, 255]
    RGB24[out_ind]         = R_val;
    RGB24[out_ind + w_h]   = G_val;
    RGB24[out_ind + w_h*2] = B_val;
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
    unsigned char Y_val = Y[out_ind];
    float U_val = U[(y / 2) * (width / 2) + x / 2] - 128.0;
    float V_val = V[(y / 2) * (width / 2) + x / 2] - 128.0;

    // Convert the YUV values to RGB values
    float R_val = Y_val + 1.403 * V_val;
    float G_val = Y_val - 0.344 * U_val - 0.714 * V_val;
    float B_val = Y_val + 1.770 * U_val;

    // Clamp the RGB values to the range [0, 255]
    auto ind = (yW + x)*3;
    RGB24[ind + 0] = R_val;
    RGB24[ind + 1] = G_val;
    RGB24[ind + 2] = B_val;
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
    unsigned char Y_val = *Y;
    float U_val = UV[0] - 128.0;
    float V_val = UV[1] - 128.0;

    // Convert the YUV values to RGB values
    float R_val = Y_val + 1.403 * V_val;
    float G_val = Y_val - 0.344 * U_val - 0.714 * V_val;
    float B_val = Y_val + 1.770 * U_val;

    // Clamp the RGB values to the range [0, 255]
    RGB24[out_ind]         = R_val;
    RGB24[out_ind + w_h]   = G_val;
    RGB24[out_ind + w_h*2] = B_val;
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
    unsigned char Y_val = *Y;
    float U_val = UV[0] - 128.0;
    float V_val = UV[1] - 128.0;

    // Convert the YUV values to RGB values
    float R_val = Y_val + 1.403 * V_val;
    float G_val = Y_val - 0.344 * U_val - 0.714 * V_val;
    float B_val = Y_val + 1.770 * U_val;

    // Clamp the RGB values to the range [0, 255]
    auto ind = (yW + x)*3;
    RGB24[ind + 0] = R_val;
    RGB24[ind + 1] = G_val;
    RGB24[ind + 2] = B_val;
}

"""
)


converter = {('yuv420p', 'chw'): mod.get_function('yuv420p_CHW_fp32'),
             ('yuv420p', 'hwc'): mod.get_function('yuv420p_HWC_fp32'),
             ('nv12', 'chw'): mod.get_function('NV12_CHW_fp32'),
             ('nv12', 'hwc'): mod.get_function('NV12_HWC_fp32')}


class Holder(PointerHolderBase):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor
        self.gpudata = tensor.data_ptr()

    def get_pointer(self):
        return self.tensor.data_ptr()

    def __index__(self):
        return self.gpudata


def tensor_to_gpuarray(tensor):
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


class FFmpegReaderCUDA(FFmpegReader):
    def __init__(self, vid:FFmpegReader, gpu=0, tensor_format='hwc'):
        assert gpu==0, 'Only support gpu=0 for now.'
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
        self.pix_fmt = 'rgb24'
        self.vid = vid
        self.out_numpy_shape = (vid.height, vid.width, 3) if tensor_format == 'hwc' else (3, vid.height, vid.width)
        self.torch_device = f'cuda:{gpu}'
        self.converter = converter[(vid.pix_fmt, tensor_format)]
        self.block_size = (16, 16, 1)
        self.grid_size = ((self.width + self.block_size[0] - 1) // self.block_size[0],
                          (self.height + self.block_size[1] - 1) // self.block_size[1])
        self.process = None

    def read(self, out_MAT=None):
        self.waitInit = True
        ret, frame_yuv420p = self.vid.read()
        if not ret:
            return False, None
        
        if out_MAT is None:
            out_MAT = gpuarray.empty(self.out_numpy_shape, dtype=np.float32)
        self.converter(cuda.In(frame_yuv420p), out_MAT, 
                       cuda.In(np.int32(self.width)), cuda.In(np.int32(self.height)),
                       block=self.block_size, grid=self.grid_size)
        return True, out_MAT
    
    def read_cudamem(self, out_MAT=None):
        self.waitInit = True
        ret, frame_yuv420p = self.vid.read()
        if not ret:
            return False, None
        
        if out_MAT is None:
            out_MAT = cuda.mem_alloc(int(np.prod(self.out_numpy_shape) * 
                                         np.dtype(np.float32).itemsize))
        self.converter(cuda.In(frame_yuv420p), out_MAT, 
                       cuda.In(np.int32(self.width)), cuda.In(np.int32(self.height)),
                       block=self.block_size, grid=self.grid_size)
        return True, out_MAT
    
    def read_torch(self):
        import torch
        self.waitInit = True
        ret, frame_yuv420p = self.vid.read()
        if not ret:
            return False, None
        
        tensor = torch.empty(self.out_numpy_shape, dtype=torch.float32, device=self.torch_device)
        tensor_proxy = tensor_to_gpuarray(tensor)
        self.converter(cuda.In(frame_yuv420p), tensor_proxy.gpudata, 
                       cuda.In(np.int32(self.width)), cuda.In(np.int32(self.height)),
                       block=self.block_size, grid=self.grid_size)
        return True, tensor

    def release(self):
        self.vid.release()
        return super().release()
