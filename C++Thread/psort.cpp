#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <execution>

int main()
{
	std::vector<int> vec(100000000);

	// fill vector with random numbers
	std::srand(unsigned(std::time(nullptr)));
	std::generate(vec.begin(), vec.end(), std::rand);
	auto start_time = std::chrono::high_resolution_clock::now();

	std::sort(vec.begin(), vec.end());

	auto end_time = std::chrono::high_resolution_clock::now();
	auto time_diff = end_time - start_time;
	std::cout << "sorting time: " <<
		time_diff / std::chrono::milliseconds(1) << "ms to run.\n";


	std::srand(unsigned(std::time(nullptr)));
	std::generate(vec.begin(), vec.end(), std::rand);
	start_time = std::chrono::high_resolution_clock::now();

	std::sort(std::execution::par, vec.begin(), vec.end());

	end_time = std::chrono::high_resolution_clock::now();
	time_diff = end_time - start_time;
	std::cout << "parallel sorting time: " <<
		time_diff / std::chrono::milliseconds(1) << "ms to run.\n";


	system("pause");
	return 0;
}