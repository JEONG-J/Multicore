#include <stdio.h>
#include <sys/time.h>
#include <chrono>

using namespace std::chrono;

#define BLOCK_SIZE 32
#define WIDTH 1027

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Matrix multiplication kernel ? thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	// 2D Thread ID
	int tx = threadIdx.x + blockIdx.x*blockDim.x;
	int ty = threadIdx.y + blockIdx.y*blockDim.y;

	if (tx<WIDTH && ty<WIDTH) {

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
    //dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid((WIDTH+BLOCK_SIZE-1)/BLOCK_SIZE, (WIDTH+BLOCK_SIZE-1)/BLOCK_SIZE);
	//dim3 dimGrid(1, 1);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);

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
        //M.elements[i] = (rand() / (float)RAND_MAX);
		M.elements[i] = 1.0;
        //if(rand() % 2) M.elements[i] = - M.elements[i];
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

float mat_sum(Matrix& m)
{
  float result=0.0;
  int i;
  for (i=0;i<m.width*m.height;i++) {
      result+=m.elements[i];
  }
  return result;
}


// Matrix multiplication on the (CPU) host in double precision
// for simplicity, we will assume that all dimensions are equal

void MatrixMulOnHost(const Matrix M, const Matrix N, Matrix P)
{   
    for (int i = 0; i < M.height; ++i)
        for (int j = 0; j < N.width; ++j) {
            double sum = 0;
            for (int k = 0; k < M.width; ++k) {
                double a = M.elements[i * M.width + k];
                double b = N.elements[k * N.width + j];
                sum += a * b;
            }
            P.elements[i * N.width + j] = sum;
        }
}



int main(void) 
{
    // Allocate and initialize the matrices
    Matrix  M  = AllocateMatrix(WIDTH, WIDTH);
    Matrix  N  = AllocateMatrix(WIDTH, WIDTH);
    Matrix  P  = AllocateMatrix(WIDTH, WIDTH);

    auto start = high_resolution_clock::now();

    MatrixMulOnHost(M, N, P);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    printf("HOST(single thread) %dX%d matrix multiplication (sum=%f) execution time: %dms\n",WIDTH, WIDTH, mat_sum(P), (int)duration.count());
    

    // M * N on the device

    start = high_resolution_clock::now();
    MatrixMulOnDevice(M, N, P);
    cudaDeviceSynchronize();
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    printf("DEVICE(CUDA) %dX%d matrix multiplication (sum=%f) execution time: %dms\n",WIDTH,WIDTH,mat_sum(P),(int)duration.count());

    FreeMatrix(M);
    FreeMatrix(N);
    FreeMatrix(P);

    return 0;
}


