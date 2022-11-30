// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


__global__ void add(float *input, float *aux, int len){
  if ((blockIdx.x + 1) * blockDim.x + threadIdx.x < len)
    input[(blockIdx.x + 1) * blockDim.x + threadIdx.x] += aux[blockIdx.x];
}


__global__ void scan(float *input, float *output, float *aux, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float TILE[BLOCK_SIZE * 2];

  // load from global memory
  if (2*(blockIdx.x * blockDim.x + threadIdx.x) < len){
    TILE[2*threadIdx.x] = input[2*(blockIdx.x * blockDim.x + threadIdx.x)];
  }
  else{
    TILE[2*threadIdx.x] = 0.0;
  }

  if (2*(blockIdx.x * blockDim.x + threadIdx.x) + 1 < len){
    TILE[2*threadIdx.x + 1] = input[2*(blockIdx.x * blockDim.x + threadIdx.x) + 1];
  }
  else{
    TILE[2*threadIdx.x + 1] = 0.0;
  }

  int stride = 1;
  while(stride < 2*BLOCK_SIZE){
    __syncthreads();
    int index = (threadIdx.x + 1)*stride*2 - 1;
    if (index < 2*BLOCK_SIZE && (index - stride) >= 0){
      TILE[index] += TILE[index-stride];
    }
    stride = stride*2;
  }

  stride = BLOCK_SIZE/2;
  while(stride > 0){
    __syncthreads();
    int index = (threadIdx.x + 1)*stride*2 - 1;
    if ((index + stride) < 2*BLOCK_SIZE){
      TILE[index+stride] += TILE[index];
    }
    stride = stride/2;
  }

  __syncthreads();

  if (2*(blockIdx.x * blockDim.x + threadIdx.x) < len)
    output[2*(blockIdx.x * blockDim.x + threadIdx.x)] = TILE[2*threadIdx.x];

  if (2*(blockIdx.x * blockDim.x + threadIdx.x) + 1 < len)
    output[2*(blockIdx.x * blockDim.x + threadIdx.x) + 1] = TILE[2*threadIdx.x + 1];
  
  aux[blockIdx.x] = TILE[2*BLOCK_SIZE-1];
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *AuxInput;
  float *AuxOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);
  int block_number = (numElements - 1) / (BLOCK_SIZE << 1) + 1;

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&AuxInput,  block_number * sizeof(float)));
  wbCheck(cudaMalloc((void **)&AuxOutput, block_number * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(AuxOutput, 0, block_number * sizeof(float)));

  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid1(block_number, 1, 1);
  dim3 dimBlock1(BLOCK_SIZE, 1, 1);
  dim3 dimGrid2((block_number - 1) / (BLOCK_SIZE << 1) + 1, 1, 1);
  dim3 dimBlock2(BLOCK_SIZE, 1, 1);
  dim3 dimGrid3(block_number - 1, 1, 1);
  dim3 dimBlock3(2*BLOCK_SIZE, 1, 1);
 
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid1, dimBlock1>>>(deviceInput, deviceOutput, AuxInput, numElements);
  cudaDeviceSynchronize();
  scan<<<dimGrid2, dimBlock2>>>(AuxInput, AuxOutput, deviceInput, block_number);
  cudaDeviceSynchronize();
  add<<<dimGrid3, dimBlock3>>>(deviceOutput, AuxOutput, numElements);
  cudaDeviceSynchronize();


  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(AuxInput);
  cudaFree(AuxOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
