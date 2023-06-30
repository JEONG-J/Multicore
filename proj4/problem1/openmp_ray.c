#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define SPHERES 20 // 생성할 구의 수
#define rnd(x) (x * rand() / RAND_MAX) // 난수를 생성하기 위한 함수
#define INF 2e10f // 무한대를 나타내는 상수
#define DIM 2048 // 이미지의 가로, 세로 픽셀 수

// 구를 나타내는 구조체
struct Sphere {
    float r, b, g; // 구의 색상 (RGB)
    float radius; // 구의 반지름
    float x, y, z; // 구의 위치
};

// 구와의 충돌 여부와, 충돌지점의 거리를 반환하는 함수
float hit(struct Sphere s, float ox, float oy, float *n) {
    float dx = ox - s.x;
    float dy = oy - s.y;
    if (dx*dx + dy*dy < s.radius*s.radius) { // 거리가 구의 반지름보다 작다면, 즉 구와 충돌한다면
        float dz = sqrtf(s.radius*s.radius - dx*dx - dy*dy);
        *n = dz / sqrtf(s.radius * s.radius); // 충돌 지점의 위치를 계산
        return dz + s.z; // 충돌 지점의 z 좌표 반환
    }
    return -INF; // 충돌하지 않는다면 -INF 반환
}

// 각 픽셀에 대한 색상을 계산하는 함수
void kernel(int x, int y, struct Sphere* s, unsigned char* ptr) {
    int offset = x + y * DIM; // 이미지 데이터에서의 현재 픽셀 위치
    float ox = (x - DIM / 2); // 광선의 x좌표
    float oy = (y - DIM / 2); // 광선의 y좌표

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for(int i=0; i<SPHERES; i++) {
        float n;
        float t = hit(s[i], ox, oy, &n); // 광선과 구와의 충돌 여부 확인
        if (t > maxz) { // 만약 가장 가까운 구와 충돌했다면
            float fscale = n;
            r = s[i].r * fscale; // 색상의 r 요소 계산
            g = s[i].g * fscale; // 색상의 g 요소 계산
            b = s[i].b * fscale; // 색상의 b 요소 계산
            maxz = t;
        }
    }

    ptr[offset*4 + 0] = (int)(r * 255); // 픽셀의 r 값 저장
    ptr[offset*4 + 1] = (int)(g * 255); // 픽셀의 g 값 저장
    ptr[offset*4 + 2] = (int)(b * 255); // 픽셀의 b 값 저장
    ptr[offset*4 + 3] = 255; // 픽셀의 투명도 값 저장
}

// PPM 형식의 이미지 파일을 작성하는 함수
void ppm_write(unsigned char* bitmap, int xdim, int ydim, FILE* fp) {
    int i, x, y;
    fprintf(fp, "P3\n");
    fprintf(fp, "%d %d\n", xdim, ydim); // 이미지의 가로, 세로 크기를 파일에 기록
    fprintf(fp, "255\n"); // 색상의 최대값을 파일에 기록
    for (y=0; y<ydim; y++) {
        for (x=0; x<xdim; x++) {
            i = x + y * xdim;
            fprintf(fp, "%d %d %d ", bitmap[4*i], bitmap[4*i+1], bitmap[4*i+2]); // 각 픽셀의 RGB값을 파일에 기록
        }
        fprintf(fp, "\n");
    }
}

int main(int argc, char* argv[]) {
    int no_threads; // 사용할 쓰레드의 개수
    int x, y;
    unsigned char* bitmap; // 이미지 데이터를 저장할 포인터

    srand(time(NULL)); // 난수 생성기 초기화

    // 명령줄 인자로 쓰레드의 개수가 제공되지 않았다면 프로그램 종료
    if (argc != 2) {
        printf("> openmp_ray.exe [number of threads]\n");
        exit(0);
    }

    // 명령줄 인자로 받은 쓰레드의 개수를 정수로 변환
    no_threads = atoi(argv[1]);

    // 구의 정보를 저장할 메모리를 할당하고, 각 구의 속성을 랜덤하게 설정
    struct Sphere* temp_s = (struct Sphere*)malloc(sizeof(struct Sphere) * SPHERES);
    for (int i=0; i<SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(2000.0f) - 1000;
        temp_s[i].y = rnd(2000.0f) - 1000;
        temp_s[i].z = rnd(2000.0f) - 1000;
        temp_s[i].radius = rnd(200.0f) + 40;
    }

    // 이미지 데이터를 저장할 메모리를 할당
    bitmap = (unsigned char*)malloc(sizeof(unsigned char) * DIM * DIM * 4);
    
    double start_time, end_time; // 측정을 위한 변수
    start_time = omp_get_wtime(); // 시간 측정 시작

    omp_set_num_threads(no_threads); // OpenMP에 사용할 쓰레드의 개수 설정
    #pragma omp parallel for private(y) // 병렬 처리 시작
    for (x = 0; x < DIM; x++)
        for (y = 0; y < DIM; y++)
            kernel(x, y, temp_s, bitmap); // 각 픽셀의 색상 계산

    end_time = omp_get_wtime(); // 시간 측정 종료
    printf("> OpenMP (%d threads) ray tracing: %.3f sec\n", no_threads, end_time - start_time); // 결과 출력

    // PPM 형식의 이미지 파일 작성
    FILE* fp = fopen("result.ppm", "w");
    ppm_write(bitmap, DIM, DIM, fp);
    fclose(fp);

    // 할당했던 메모리 해제
    free(bitmap);
    free(temp_s);

    return 0;
}
