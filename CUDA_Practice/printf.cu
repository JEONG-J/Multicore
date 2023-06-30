#include <stdio.h>

__global__ void hello_world(void) {
#if __CUDA_ARCH__ >= 200
	printf("Hello World!\n");
#endif
}

int main (void) {
    hello_world<<<1,5>>>();
    cudaDeviceSynchronize();
    return 0;
}
