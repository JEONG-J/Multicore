#include <thrust/device_vector.h> // 효율적인 GPU 연산을 위한 thrust 라이브러리를 불러온다
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <cmath>
#include <iostream>

// 연산을 수행하는 함수 정의
struct pi_functor
{
    double step;
    pi_functor(double step) : step(step) {} // 구조체 생성자.

    __host__ __device__
        double operator()(const int& i) const // 원주율을 계산하는 식을 정의
    {
        double x = (i + 0.5) * step;
        return 4.0 / (1.0 + x * x);
    }
};

// 원주율을 계산하기 위해 위에서 정의한 함수를 적용하여 합계 구하기
double compute_pi(thrust::device_vector<int>::iterator first,
    thrust::device_vector<int>::iterator last,
    double step)
{
    return step * thrust::transform_reduce(first, last, pi_functor(step), 0.0, thrust::plus<double>());
}

int main()
{
    const long num_steps = 1000000000; // 총 단계 수
    double step = 1.0 / (double)num_steps; // 단계 크기

    // 숫자를 저장할 디바이스 벡터를 선언하고 이에 0부터 num_steps-1까지의 숫자를 생성
    thrust::device_vector<int> d_sequence(num_steps);
    thrust::sequence(d_sequence.begin(), d_sequence.end());

    // CUDA 이벤트를 생성하여 연산 시간을 측정
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // 위에서 선언한 함수를 호출하여 원주율을 계산
    double pi = compute_pi(d_sequence.begin(), d_sequence.end(), step);

    // 계산이 끝났으므로 CUDA 이벤트를 기록하고 시간을 측정
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float timeDiff;
    cudaEventElapsedTime(&timeDiff, start, stop);

    // 계산에 걸린 시간과 계산 결과를 출력
    std::cout << "Execution Time : " << timeDiff / 1000.0 << " sec" << std::endl;
    std::cout << "pi = " << pi << std::endl;

    // 생성했던 CUDA 이벤트를 제거
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
