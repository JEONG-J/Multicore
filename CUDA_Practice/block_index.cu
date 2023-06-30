#include <stdio.h>

__global__ void exec_conf(void) {
     int ix =  threadIdx.x + blockIdx.x * blockDim.x;
#if __CUDA_ARCH__ >= 200
     printf("gridDim = (%d,%d,%d), blockDim = (%d,%d,%d)\n",
            gridDim.x,gridDim.y,gridDim.z,
            blockDim.x,blockDim.y,blockDim.z);

    printf("blockIdx = (%d,%d,%d), threadIdx = (%d,%d,%d), arrayIdx %d\n",
            blockIdx.x,blockIdx.y,blockIdx.z,
            threadIdx.x,threadIdx.y,threadIdx.z, ix);
#endif
}

int main (void) {
    exec_conf<<<2,3>>>();
    cudaDeviceSynchronize();
    return 0;
}
