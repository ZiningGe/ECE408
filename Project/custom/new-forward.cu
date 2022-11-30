#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 20
#define CONSTANT_MEM_SIZE 6000

__constant__ float weight[CONSTANT_MEM_SIZE];
__constant__ int height_out, width_out, w_size;

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_size = ceil(Width_out * 1.0 / TILE_WIDTH);
    int H_size = ceil(Height_out * 1.0 / TILE_WIDTH);

    int b = blockIdx.z;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;
    for (int c = 0; c < Channel; c++){
        for (int p = 0; p < K; p++){
            for (int q = 0; q < K; q++){
                acc += in_4d(b, c, h+p, w+q) * mask_4d(m, c, p, q);
            }
        }
    }
    if (h < Height_out && w < Width_out)
        out_4d(b, m, h, w) = acc;

    

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void conv_forward_kernel_constant(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * height_out * width_out) + (i2) * (height_out * width_out) + (i1) * (width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) weight[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    // int W_size = ceil(Width_out * 1.0 / TILE_WIDTH);

    int b = blockIdx.z;
    int m = blockIdx.x;
    int h = (blockIdx.y / w_size) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % w_size) * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;
    for (int c = 0; c < Channel; c++){
        for (int p = 0; p < K; p++){
            for (int q = 0; q < K; q++){
                acc += in_4d(b, c, h+p, w+q) * mask_4d(m, c, p, q);
            }
        }
    }
    if (h < height_out && w < width_out)
        out_4d(b, m, h, w) = acc;

    

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}


__global__ void conv_forward_kernel_tiled(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{   
    extern __shared__ float tiled_input[];


    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_size = ceil(Width_out * 1.0 / TILE_WIDTH);
    int H_size = ceil(Height_out * 1.0 / TILE_WIDTH);

    int b = blockIdx.z;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
    int block_size = TILE_WIDTH + K - 1;

    for (int i = 0; i < Channel; i++){
        if ((h < Height) && (w < Width)){
            tiled_input[i * block_size * block_size + threadIdx.y * block_size + threadIdx.x] = in_4d(b, i, h, w);
        }
    }
    __syncthreads();
        
    if (threadIdx.y < TILE_WIDTH && threadIdx.x < TILE_WIDTH){
        float acc = 0.0f;
        for (int c = 0; c < Channel; c++){
            for (int p = 0; p < K; p++){
                for (int q = 0; q < K; q++){
                    // acc += in_4d(b, c, h+p, w+q) * mask_4d(m, c, p, q);
                    acc += tiled_input[(c)* block_size * block_size + (threadIdx.y + p)* block_size + (threadIdx.x+q)] * mask_4d(m, c, p, q);
                }
            }
        }
    
        __syncthreads();
        if (h < Height_out && w < Width_out)
            out_4d(b, m, h, w) = acc;
    }
    

    

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void conv_forward_kernel_atomics(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_size = ceil(Width_out * 1.0 / TILE_WIDTH);
    int H_size = ceil(Height_out * 1.0 / TILE_WIDTH);

    int b = blockIdx.z;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_size) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_size) * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;
    
    for (int p = 0; p < K; p++){
        for (int q = 0; q < K; q++){
            acc += in_4d(b, threadIdx.z, h+p, w+q) * mask_4d(m, threadIdx.z, p, q);
        }
    }
    
    if (h < Height_out && w < Width_out)
        atomicAdd(&out_4d(b, m, h, w), acc);

    

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, 
    float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int W_size = ceil(Width_out * 1.0 / TILE_WIDTH);


    cudaMalloc(device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc(device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    cudaMalloc(device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(weight, host_mask, Map_out * Channel * K * K * sizeof(float));
    cudaMemcpyToSymbol(height_out, &Height_out, sizeof(int));
    cudaMemcpyToSymbol(width_out, &Width_out, sizeof(int));
    cudaMemcpyToSymbol(w_size, &W_size, sizeof(int));


    

    // // streaming
    // cudaStream_t stream0, stream1, stream2, stream3, stream4;
    // cudaStreamCreate(&stream0);
    // cudaStreamCreate(&stream1);
    // cudaStreamCreate(&stream2);
    // cudaStreamCreate(&stream3);
    // cudaStreamCreate(&stream4);

    // int SegSize = 10;
    // int input_SegSize = Channel * Height * Width;
    // int output_SegSize = Map_out * Height_out * Width_out;
    // const int H_out = Height - K + 1;
    // const int W_out = Width - K + 1;
    // int W_size = ceil(W_out * 1.0 / TILE_WIDTH);
    // int H_size = ceil(H_out * 1.0 / TILE_WIDTH);
    // int Y = H_size * W_size;

    // for (int i = 0; i < Batch; i += 5*SegSize){
    //     cudaMemcpyAsync(*device_input_ptr + (i+ 0 * SegSize) * input_SegSize, host_input + (i+ 0 * SegSize) * input_SegSize, SegSize * input_SegSize * sizeof(float), cudaMemcpyHostToDevice, stream0);
    //     cudaMemcpyAsync(*device_input_ptr + (i+ 1 * SegSize) * input_SegSize, host_input + (i+ 1 * SegSize) * input_SegSize, SegSize * input_SegSize * sizeof(float), cudaMemcpyHostToDevice, stream1);
    //     cudaMemcpyAsync(*device_input_ptr + (i+ 2 * SegSize) * input_SegSize, host_input + (i+ 2 * SegSize) * input_SegSize, SegSize * input_SegSize * sizeof(float), cudaMemcpyHostToDevice, stream2);
    //     cudaMemcpyAsync(*device_input_ptr + (i+ 3 * SegSize) * input_SegSize, host_input + (i+ 3 * SegSize) * input_SegSize, SegSize * input_SegSize * sizeof(float), cudaMemcpyHostToDevice, stream3);
    //     cudaMemcpyAsync(*device_input_ptr + (i+ 4 * SegSize) * input_SegSize, host_input + (i+ 4 * SegSize) * input_SegSize, SegSize * input_SegSize * sizeof(float), cudaMemcpyHostToDevice, stream4);

    //     dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    //     dim3 gridDim(Map_out, Y, SegSize);

    //     conv_forward_kernel<<<gridDim, blockDim, 0, stream0>>>(*device_output_ptr + (i+ 0 * SegSize) * output_SegSize, *device_input_ptr + (i+ 0 * SegSize) * input_SegSize, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    //     conv_forward_kernel<<<gridDim, blockDim, 0, stream1>>>(*device_output_ptr + (i+ 1 * SegSize) * output_SegSize, *device_input_ptr + (i+ 1 * SegSize) * input_SegSize, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    //     conv_forward_kernel<<<gridDim, blockDim, 0, stream2>>>(*device_output_ptr + (i+ 2 * SegSize) * output_SegSize, *device_input_ptr + (i+ 2 * SegSize) * input_SegSize, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    //     conv_forward_kernel<<<gridDim, blockDim, 0, stream3>>>(*device_output_ptr + (i+ 3 * SegSize) * output_SegSize, *device_input_ptr + (i+ 3 * SegSize) * input_SegSize, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    //     conv_forward_kernel<<<gridDim, blockDim, 0, stream4>>>(*device_output_ptr + (i+ 4 * SegSize) * output_SegSize, *device_input_ptr + (i+ 4 * SegSize) * input_SegSize, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        

    //     cudaMemcpyAsync((float *)host_output + (i+ 0 * SegSize) * output_SegSize, *device_output_ptr + (i+ 0 * SegSize) * output_SegSize, SegSize * output_SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream0);
    //     cudaMemcpyAsync((float *)host_output + (i+ 1 * SegSize) * output_SegSize, *device_output_ptr + (i+ 1 * SegSize) * output_SegSize, SegSize * output_SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    //     cudaMemcpyAsync((float *)host_output + (i+ 2 * SegSize) * output_SegSize, *device_output_ptr + (i+ 2 * SegSize) * output_SegSize, SegSize * output_SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    //     cudaMemcpyAsync((float *)host_output + (i+ 3 * SegSize) * output_SegSize, *device_output_ptr + (i+ 3 * SegSize) * output_SegSize, SegSize * output_SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream3);
    //     cudaMemcpyAsync((float *)host_output + (i+ 4 * SegSize) * output_SegSize, *device_output_ptr + (i+ 4 * SegSize) * output_SegSize, SegSize * output_SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream4);

    // }

    

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // // original version
    // const int H_out = Height - K + 1;
    // const int W_out = Width - K + 1;
    // int W_size = ceil(W_out * 1.0 / TILE_WIDTH);
    // int H_size = ceil(H_out * 1.0 / TILE_WIDTH);
    // int Y = H_size * W_size;

    // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 gridDim(Map_out, Y, Batch);

    // conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);


    //weight in constant memory
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;
    int W_size = ceil(W_out * 1.0 / TILE_WIDTH);
    int H_size = ceil(H_out * 1.0 / TILE_WIDTH);
    int Y = H_size * W_size;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(Map_out, Y, Batch);

    conv_forward_kernel_constant<<<gridDim, blockDim>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);


    // // tiled share memory
    // const int H_out = Height - K + 1;
    // const int W_out = Width - K + 1;
    // int W_size = ceil(W_out * 1.0 / TILE_WIDTH);
    // int H_size = ceil(H_out * 1.0 / TILE_WIDTH);
    // int Y = H_size * W_size;

    // dim3 blockDim(TILE_WIDTH + K - 1, TILE_WIDTH + K - 1, 1);
    // dim3 gridDim(Map_out, Y, Batch);

    // conv_forward_kernel_tiled<<<gridDim, blockDim, Channel*(TILE_WIDTH + K - 1)*(TILE_WIDTH + K - 1)*sizeof(float)>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    

    // // input channel reduction atomics
    // const int H_out = Height - K + 1;
    // const int W_out = Width - K + 1;
    // int W_size = ceil(W_out * 1.0 / TILE_WIDTH);
    // int H_size = ceil(H_out * 1.0 / TILE_WIDTH);
    // int Y = H_size * W_size;

    // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, Channel);
    // dim3 gridDim(Map_out, Y, Batch);

    // conv_forward_kernel_atomics<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);


}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMemcpy(host_output, device_output, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
