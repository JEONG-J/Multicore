#include <omp.h>
#include <stdio.h>

#define NUM_THREADS 4
#define END_NUM 1000

int main ()
{ 
	int i;
	int sum=0;
	double start_time, end_time;
	omp_set_num_threads(NUM_THREADS);
	start_time = omp_get_wtime( );

#pragma omp parallel for
		for (i = 1; i <= END_NUM; i++) {
			#pragma omp critical
			sum+=i;
		}

	end_time = omp_get_wtime( );
	printf("sum = 1+2+..+%d = %d\n",END_NUM,sum);
	printf("time elapsed: %lfs\n",end_time-start_time);

	return 1;
}