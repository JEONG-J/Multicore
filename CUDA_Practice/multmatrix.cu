#include <stdio.h>

#define WIDTH 16

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Matrix multiplication kernel ? thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
    // 2D Thread ID
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float Pvalue = 0;
 
    for (int k = 0; k < M.width; ++k)
    { 
         float Melement = M.elements[ty * M.width + k];
         float Nelement = N.elements[k * N.width + tx];
         Pvalue += Melement * Nelement;
    } 
    // Write the matrix to device memory;
    // each thread writes one element
    P.elements[ty * P.width + tx] = Pvalue;
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
    Matrix Mdevice = M;
    int size = M.width * M.height * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix M) {
    cudaFree(M.elements);
}

void FreeMatrix(Matrix M) {
    free(M.elements);
}

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.width * Mhost.height * sizeof(float);
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
	cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
    int size = Mdevice.width * Mdevice.height * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
	cudaMemcpyDeviceToHost);
}

// Matrix multiplication on the device
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P); // Clear memory
    
     // Setup the execution configuration
    dim3 dimBlock(WIDTH, WIDTH);
    dim3 dimGrid(1, 1);

    // Launch the device computation threads!
    MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd);

    // Read P from the device
    CopyFromDeviceMatrix(P, Pd); 

    // Free device matrices
    FreeDeviceMatrix(Md);
    FreeDeviceMatrix(Nd);
    FreeDeviceMatrix(Pd);
} 

Matrix AllocateMatrix(int height, int width)
{
    Matrix M;
    M.width = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;

    M.elements = (float*) malloc(size*sizeof(float));

    for(unsigned int i = 0; i < M.height * M.width; i++)
    {
        M.elements[i] = (rand() / (float)RAND_MAX);
        if(rand() % 2)
            M.elements[i] = - M.elements[i];
    }
    return M;
}


void PrintMatrix(float* ma, int X, int Y)
{
	int i,j;
	for (j=0;j<Y;j++) {
		for (i=0;i<X;i++) {
			printf("%4f ",ma[i+j*X]);
		}
		printf("\n");
	}
}


int main(void) 
{
	int i,j;
    // Allocate and initialize the matrices
    Matrix  M  = AllocateMatrix(WIDTH, WIDTH);
    Matrix  N  = AllocateMatrix(WIDTH, WIDTH);
    Matrix  P  = AllocateMatrix(WIDTH, WIDTH);

    // M * N on the device
    MatrixMulOnDevice(M, N, P);


	PrintMatrix(M.elements,M.width,M.height);
	printf("\n");
	PrintMatrix(N.elements,N.width,N.height);
	printf("\n");
	PrintMatrix(P.elements,P.width,P.height);

    // Free matrices
    FreeMatrix(M);
    FreeMatrix(N);
    FreeMatrix(P);

    return 0;
}

