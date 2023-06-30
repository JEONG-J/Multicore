#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <execution>

bool isPrime(int x)
{
    int i;
    if (x <= 1) return false;
    for (i = 2; i < x; i++) {
        if (x % i == 0) return false;
    }
    return true;
}

int getPrimeNo(int n)
{
    int i;
    int prime_count = 0;

    for (i = 2; i <= n; i++) {
        if (isPrime(i)) prime_count++;
    }
    std::cout << "No. of Primes (" << n << ") =" << prime_count << "\n";
    return prime_count;
}

int main()
{
    int i;
    std::vector<int> A = { 100000, 100100, 100200, 100300, 100400, 100500, 100600, 100700, 100800, 100900, 
                           101000, 101100, 101200, 101300, 101400, 101500, 101600, 101700, 101800, 101900 };
    std::vector<int> B;
    B.resize(A.size());

    auto start_time = std::chrono::high_resolution_clock::now();
    std::transform(A.begin(), A.end(), B.begin(), getPrimeNo);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_diff = end_time - start_time;
    std::cout << "transform time:" << time_diff / std::chrono::milliseconds(1) << "ms\n";
    for (i = 0; i < B.size(); i++) std::cout << B[i] << "\n";

    start_time = std::chrono::high_resolution_clock::now();
    std::transform(std::execution::par, A.begin(), A.end(), B.begin(), getPrimeNo);
    end_time = std::chrono::high_resolution_clock::now();
    time_diff = end_time - start_time;
    std::cout << "parallel transform time:" << time_diff / std::chrono::milliseconds(1) << "ms\n";
    for (i = 0; i < B.size(); i++) std::cout << B[i] << "\n";

    system("pause");
    return 0;
}