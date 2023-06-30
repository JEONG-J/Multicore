#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define SPHERES 20
#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f
#define DIM 2048

struct Sphere {
    float r, b, g;
    float radius;
    float x, y, z;
};

// 레이가 구체에 충돌하는지 확인하고, 충돌하면 그 위치의 깊이(z값)과 표면의 단위 벡터를 반환하는 함수
__device__ float hit(Sphere s, float ox, float oy, float* n) {
    // 수평(x,y) 거리 계산
    float dx = ox - s.x;
    float dy = oy - s.y;
    // 레이가 구체에 충돌하는지 확인
    if (dx * dx + dy * dy < s.radius * s.radius) {
        // 충돌한 경우, z 값을 계산하고, 단위 벡터를 반환
        float dz = sqrtf(s.radius * s.radius - dx * dx - dy * dy);
        *n = dz / sqrtf(s.radius * s.radius);
        return dz + s.z;
    }
    // 충돌하지 않은 경우, -INF 반환
    return -INF;
}

__global__ void kernel(Sphere* s, unsigned char* ptr) {
    // 각 스레드에 대해 연산할 픽셀의 위치를 계산
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // 픽셀의 오프셋 계산
    int offset = x + y * blockDim.x * gridDim.x;
    // 픽셀의 중심 위치 계산
    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = hit(s[i], ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    }

    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}

void ppm_write(unsigned char* bitmap, int xdim, int ydim, FILE* fp) {
    int i, x, y;
    fprintf(fp, "P3\n");
    fprintf(fp, "%d %d\n", xdim, ydim);
    fprintf(fp, "255\n");
    for (y = 0; y < ydim; y++) {
        for (x = 0; x < xdim; x++) {
            i = x + y * xdim;
            fprintf(fp, "%d %d %d ", bitmap[4 * i], bitmap[4 * i + 1], bitmap[4 * i + 2]);
        }
        fprintf(fp, "\n");
    }
}

int main(int argc, char* argv[]) {
    int x, y;
    unsigned char* bitmap;

    srand(time(NULL));  // 랜덤 시드를 초기화

    Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);  // 구체 구조체를 위한 메모리 할당
    Sphere* dev_temp_s;  // 디바이스(즉, GPU)에서 사용될 구체 구조체 포인터
    cudaMalloc((void**)&dev_temp_s, sizeof(Sphere) * SPHERES);  // 디바이스에서 사용할 메모리를 할당

    // 모든 구체에 대해 랜덤한 속성을 설정
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(2000.0f) - 1000;
        temp_s[i].y = rnd(2000.0f) - 1000;
        temp_s[i].z = rnd(2000.0f) - 1000;
        temp_s[i].radius = rnd(200.0f) + 40;
    }

    bitmap = (unsigned char*)malloc(sizeof(unsigned char) * DIM * DIM * 4);  // 비트맵 이미지를 위한 메모리 할당
    unsigned char* dev_bitmap;  // 디바이스에서 사용될 비트맵 이미지 포인터
    cudaMalloc((void**)&dev_bitmap, sizeof(unsigned char) * DIM * DIM * 4);  // 디바이스에서 사용할 메모리를 할당

    cudaMemcpy(dev_temp_s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice);  // 호스트(즉, CPU)에서 디바이스로 구체 정보를 복사

    dim3 grids(DIM / 16, DIM / 16);  // 그리드의 크기 설정
    dim3 threads(16, 16);  // 블록당 스레드의 수 설정

    cudaEvent_t start, stop;  // 이벤트 변수 선언 (시간 측정을 위해 사용됨)
    cudaEventCreate(&start);  // 시작 이벤트 생성
    cudaEventCreate(&stop);  // 종료 이벤트 생성
    cudaEventRecord(start, 0);  // 레이 트레이싱 시작 시간 기록

    kernel << <grids, threads >> > (dev_temp_s, dev_bitmap);  // CUDA 커널 호출

    cudaEventRecord(stop, 0);  // 레이 트레이싱 종료 시간 기록
    cudaEventSynchronize(stop);  // 모든 CUDA 연산이 종료될 때까지 대기
    float elapsedTime;  // 경과 시간을 저장할 변수
    cudaEventElapsedTime(&elapsedTime, start, stop);  // 시작 이벤트와 종료 이벤트 사이의 시간 측정
    printf("CUDA ray tracing: %.3f sec\n", elapsedTime / 1000.0f);  // 측정된 시간을 출력

    cudaEventDestroy(start);  // 이벤트 파괴
    cudaEventDestroy(stop);  // 이벤트 파괴

    cudaMemcpy(bitmap, dev_bitmap, sizeof(unsigned char) * DIM * DIM * 4, cudaMemcpyDeviceToHost);  // 디바이스에서 호스트로 비트맵 이미지 복사

    FILE* fp = fopen("result_CUDA.ppm", "w");  // 결과 이미지를 저장할 파일 열기
    ppm_write(bitmap, DIM, DIM, fp);  // ppm 형식으로 파일에 쓰기

    fclose(fp);  // 파일 닫기
    free(bitmap);  // 할당한 메모리 해제
    free(temp_s);  // 할당한 메모리 해제
    cudaFree(dev_temp_s);  // 디바이스에서 할당한 메모리 해제
    cudaFree(dev_bitmap);  // 디바이스에서 할당한 메모리 해제

    return 0;
}
