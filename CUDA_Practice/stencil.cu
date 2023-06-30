#include <stdio.h>

// we assume N is divisible by BLOCK_SIZE
#define N 16  
#define BLOCK_SIZE 8
#define RADIUS 3

__global__ void stencil_1d(int *in, int *out) 
{
        __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
        int gindex = threadIdx.x + blockIdx.x * blockDim.x;
        int lindex = threadIdx.x + RADIUS;

        temp[lindex] = in[gindex];
        if (threadIdx.x < RADIUS) {
                if (gindex-RADIUS>=0) temp[lindex - RADIUS] = in[gindex - RADIUS];
                else temp[lindex-RADIUS]=0; 
                if (gindex+BLOCK_SIZE<N) temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
                else temp[lindex+BLOCK_SIZE]=0;
        }
        __syncthreads();

        int result = 0;
        for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
                result += temp[lindex + offset];
        out[gindex] = result;
}

__global__ void stencil_simple(int *in, int *out) 
{
        int gindex = threadIdx.x + blockIdx.x * blockDim.x;

        int result = 0;
        for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
          if ((gindex+offset>=0) && (gindex+offset<N))
            result += in[gindex + offset];
        out[gindex] = result;
}

int main()
{
        int* a, *b;
        int* dev_a, *dev_b;
        int i;

        a= (int*)malloc(sizeof(int)*N);
        b= (int*)malloc(sizeof(int)*N);

        cudaMalloc((void**)&dev_a,sizeof(int)*N);
        cudaMalloc((void**)&dev_b,sizeof(int)*N);

        for (i=0;i<N;i++) a[i]=i;

        cudaMemcpy(dev_a, a, sizeof(int)*N, cudaMemcpyHostToDevice);
        stencil_1d<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dev_a, dev_b);
        cudaMemcpy(b,dev_b,sizeof(int)*N,cudaMemcpyDeviceToHost);


        for (i=0;i<N;i++) {
                printf("a[%d]=%d, b[%d]=%d\n",i,a[i],i,b[i]);
        }

	free(a); free(b);
	cudaFree(dev_a); cudaFree(dev_b);
        return 0;

}