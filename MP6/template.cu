// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define TILE_SIZE 32
#define BLOCK_SIZE 256

//@@ insert code here
__global__ void convert_float_to_char(float *input, unsigned char *output, int imageWidth, int imageHeight, int imageChannels){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((col < imageWidth) && (row < imageHeight)){
    for (int i = 0; i < imageChannels; i++){
      output[row * imageWidth * imageChannels + col * imageChannels + i] = (unsigned char)(255 * input[row * imageWidth * imageChannels + col * imageChannels + i]);
    }
  }
}

__global__ void convert_RGB_to_Gray(unsigned char *input, unsigned char *output, int imageWidth, int imageHeight, int imageChannels){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((col < imageWidth) && (row < imageHeight)){
    float r = input[row * imageWidth * imageChannels + col * imageChannels];
    float g = input[row * imageWidth * imageChannels + col * imageChannels + 1];
    float b = input[row * imageWidth * imageChannels + col * imageChannels + 2];

    output[row * imageWidth + col] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
    // printf("%f", (0.21*r + 0.71*g + 0.07*b));
  }
}

__global__ void histogram_kernel(unsigned char *input, int imageWidth, int imageHeight, unsigned int * histogram){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("hello");
  if (col < HISTOGRAM_LENGTH && row == 1){
    histogram[col] = 0;
  }

  if ((col < imageWidth) && (row < imageHeight)){
    atomicAdd(&histogram[input[row * imageWidth + col]], 1);
  }


}

__global__ void scan(unsigned int *input, float *output, int len, int imageWidth, int imageHeight) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float TILE[BLOCK_SIZE * 2];
  
  
  // if (blockIdx.x * blockDim.x + threadIdx.x < len){
  //   input[blockIdx.x * blockDim.x + threadIdx.x] /= (imageWidth* imageHeight * 1.0);
  // }
  // __syncthreads();

  // load from global memory
  if (2*(blockIdx.x * blockDim.x + threadIdx.x) < len){
    TILE[2*threadIdx.x] = input[2*(blockIdx.x * blockDim.x + threadIdx.x)]/(imageWidth* imageHeight * 1.0);
  }
  else{
    TILE[2*threadIdx.x] = 0.0;
  }

  if (2*(blockIdx.x * blockDim.x + threadIdx.x) + 1 < len){
    TILE[2*threadIdx.x + 1] = input[2*(blockIdx.x * blockDim.x + threadIdx.x) + 1]/(imageWidth* imageHeight * 1.0);
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

  // printf("%d was %u\n", threadIdx.x, input[threadIdx.x]);

}

// unsigned char correct_color(unsigned char val, unsigned int *scan_histogram){
//   return min(max(255*(scan_histogram[val]- scan_histogram[0])/(1.0 - scan_histogram[0]), 0.0), 255.0);
// }


// unsigned char clamp(float x, float start, float end){
//   return min(max(x, start), end);
// }


__global__ void histogram_equalization(unsigned char *input, unsigned char *output, int imageWidth, int imageHeight, int imageChannels, float *scan_histogram){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((col < imageWidth) && (row < imageHeight)){
    for (int i = 0; i < imageChannels; i++){
      unsigned char val = input[row * imageWidth * imageChannels + col * imageChannels + i];
      output[row * imageWidth * imageChannels + col * imageChannels + i] = (unsigned char)min(max(255*(scan_histogram[val]- scan_histogram[0])/(1.0 - scan_histogram[0]), 0.0), 255.0);
    }
  }
}

__global__ void convert_char_to_float(unsigned char *input, float *output, int imageWidth, int imageHeight, int imageChannels){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((col < imageWidth) && (row < imageHeight)){
    for (int i = 0; i < imageChannels; i++){
      output[row * imageWidth * imageChannels + col * imageChannels + i] = (float)(input[row * imageWidth * imageChannels + col * imageChannels + i]/255.0);
      // printf("%f", output[row * imageWidth * imageChannels + col * imageChannels + i]);
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  float *deviceInput;
  float *deviceOutput;
  unsigned char *deviceUchar;
  unsigned char *deviceGreyScale;
  unsigned int *histogram;
  float *scan_histogram;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  // printf("%d, %d, %d\n", imageWidth, imageHeight, imageChannels);

  //@@ insert code here
  cudaMalloc(&deviceInput, imageHeight*imageWidth*imageChannels*sizeof(float));
  cudaMalloc(&deviceOutput, imageHeight*imageWidth*imageChannels*sizeof(float));
  cudaMalloc(&deviceUchar, imageHeight*imageWidth*imageChannels*sizeof(unsigned char));
  cudaMalloc(&deviceGreyScale, imageHeight*imageWidth*sizeof(unsigned char));
  cudaMalloc(&histogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc(&scan_histogram, HISTOGRAM_LENGTH * sizeof(float));


  cudaMemcpy(deviceInput, hostInputImageData, imageHeight*imageWidth*imageChannels*sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid(ceil(imageWidth * 1.0 / TILE_SIZE), ceil(imageHeight * 1.0 / TILE_SIZE), 1);
  dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

  dim3 dimGrid1(1, 1, 1);
  dim3 dimBlock1(BLOCK_SIZE, 1, 1);

  convert_float_to_char<<<dimGrid, dimBlock>>>(deviceInput, deviceUchar, imageWidth, imageHeight, imageChannels);
  convert_RGB_to_Gray<<<dimGrid, dimBlock>>>(deviceUchar, deviceGreyScale, imageWidth, imageHeight, imageChannels);
  histogram_kernel<<<dimGrid, dimBlock>>>(deviceGreyScale, imageWidth, imageHeight, histogram);

  scan<<<dimGrid1, dimBlock1>>>(histogram, scan_histogram, HISTOGRAM_LENGTH, imageWidth, imageHeight);
  histogram_equalization<<<dimGrid, dimBlock>>>(deviceUchar, deviceUchar, imageWidth, imageHeight, imageChannels, scan_histogram);
  convert_char_to_float<<<dimGrid, dimBlock>>>(deviceUchar, deviceOutput, imageWidth, imageHeight, imageChannels);


  cudaMemcpy(hostOutputImageData, deviceOutput, imageHeight*imageWidth*imageChannels*sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceUchar);
  cudaFree(deviceGreyScale);
  cudaFree(histogram);
  cudaFree(scan_histogram);

  return 0;
}
