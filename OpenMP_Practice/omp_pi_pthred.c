//using pthread

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define NUM_THREADS 4
#define num_steps 100000

double pi[NUM_THREADS];

void *run(void *threadid) {
	int* t_ptr=(int*)threadid;
    double step;
	double x, sum = 0.0;
	int my_id, i;
	int i_start = 0, i_end = 0;
	my_id = *t_ptr;

	i_start = my_id * (num_steps / NUM_THREADS);
	i_end = i_start + (num_steps / NUM_THREADS);
	step = 1.0 / (double)num_steps;

	for (i = i_start; i < i_end; i++) {
		x = (i + 0.5)*step;
		sum = sum + 4.0 / (1.0 + x*x);
	}

	printf("Myid%d, sum=%.8lf\n", my_id, sum*step);
	pi[my_id] = sum*step;
	pthread_exit(NULL);
}

int main(int argc, char *argv[]) {

	pthread_t threads[NUM_THREADS];

	int t, pro_i, status;
	int ta[NUM_THREADS];

	for (t = 0; t < NUM_THREADS; t++){
		ta[t]=t;
		pro_i = pthread_create(&threads[t], NULL, run, (void *)&ta[t]);

		if (pro_i) {
			printf("ERROR code is %d\n", pro_i);
			exit(-1);
		}
	}

    int i;
    for (i=0;i<NUM_THREADS;i++) 
	pthread_join(threads[i], (void **)&status);

    double pi_sum=0;
    for (i=0;i<NUM_THREADS;i++) pi_sum = pi_sum + pi[i];
    
    printf("integration result=%.8lf\n", pi_sum);
    pthread_exit(NULL);
}