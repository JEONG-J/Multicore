#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

long num_steps = 10000000;
double step;

int main(int argc, char* argv[]){
    if (argc != 4){
        printf("Usage: %s scheduling_type# chunk size #_of_thread\n", argv[0]);
        return -1;
    }

    int scheduling_type_num = atoi(argv[1]);
	int chunk_size = atoi(argv[2]);
    int number_threads = atoi(argv[3]);

    long i;
    double x, pi, sum = 0.0;
    double start_time, end_time;

    omp_set_num_threads(number_threads);

    start_time = omp_get_wtime();
    step = 1.0/(double) num_steps;

    switch(scheduling_type_num){
        case 1:
            #pragma omp parallel for reduction(+:sum) private(x) schedule(static, chunk_size)
            for(i=0; i<num_steps; i++){
                x = (i+0.5)*step;
                sum = sum + 4.0/(1.0+x*x);
            }
            break;
        case 2:
            #pragma omp parallel for reduction(+:sum) private(x) schedule(dynamic, chunk_size)
            for(i=0; i<num_steps; i++){
                x = (i+0.5)*step;
                sum = sum + 4.0/(1.0+x*x);
            }
            break;
        case 3:
            #pragma omp parallel for reduction(+:sum) private(x) schedule(guided, chunk_size)
            for(i=0; i<num_steps; i++){
                x = (i+0.5)*step;
                sum = sum + 4.0/(1.0+x*x);
            }
            break;
    }
    
    pi = step * sum;
    end_time = omp_get_wtime();
    double result_time = end_time - start_time;

    printf("Execution Time : %lfms\n", result_time);
    printf("pi=%.24lf\n",pi);
}
