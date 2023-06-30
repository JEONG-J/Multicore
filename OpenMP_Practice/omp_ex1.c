#include <omp.h>
#include <stdio.h>

void main()
{	
	printf("######### before openmp block#### \n");
#pragma omp parallel num_threads(7)
	{
		printf("Hello World!\n");
	}
	printf("######### after openmp block#### \n");
}