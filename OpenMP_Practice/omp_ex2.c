#include <omp.h>
#include <stdio.h>

void main()
{	
	omp_set_num_threads(7);
#pragma omp parallel
	{
		printf("Hello World! Thread#: %d , Total Thread#:%d, numprocs:%d\n", omp_get_thread_num(),omp_get_num_threads(), omp_get_num_procs());
	}
}