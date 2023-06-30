package Problem1_block;

public class pc_static_block {
	
	//사용할 전체 쓰레드의 넘버와 소수를 구할 최대 범위 지정
    private static int NUM_END = 200000;
    private static int[] NUM_THREADS = {1, 2, 4, 6, 8, 10, 12, 14, 16, 32};

    public static void main(String[] args) {
    	// 쓰레드를 반복하면서 소수 계산을 실행
        for (int num_Threads : NUM_THREADS) {
        	int counter = 0;
        	int[] eachPrimeCnt = new int[num_Threads];
            long startTime = System.currentTimeMillis();
            
            //작업을 처리할 쓰레드 객체를 저장
            Thread[] threads = new Thread[num_Threads];
            // 주어진 범위에서 소수를 계산하고, 소수의 개수를 저장
            PrimeCounter[] primeCnt = new PrimeCounter[num_Threads];
            
           //각각 쓰레드와 PrimeCnt를 초기화하고, 쓰레드를 시작하여 주어진 범위 내의 소수를 병렬로 계산 작업한다.
           //block decomposition 방식을 사용하여 전체 범위를 쓰레드 수의 블록으로 나눈다.
           //PrimeCnt를 실행할 쓰레드를 생성 후 할당하여 쓰레드를 시작한다.
            int blockSize = NUM_END / num_Threads;
            for (int i = 0; i < num_Threads; i++) {
                int start = i * blockSize;
                int end = (i == num_Threads - 1) ? NUM_END : (i + 1) * blockSize;
                primeCnt[i] = new PrimeCounter(start, end);
                threads[i] = new Thread(primeCnt[i]);
                threads[i].start();
            }


            //PrimeCnt에 가지고 있는 소수의 개수를 returnPrimeCnt 메소드를 통해 반환한다.
            //반환된 소수의 개수를 counter에 누적해서 계속 더한다.
            //join을 통해 해당 쓰레드가 작업을 수행하는 동안 접근하지 못하도록 한다.
            for (int i = 0; i < num_Threads; i++) {
                try {
                    threads[i].join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                eachPrimeCnt[i] = primeCnt[i].returnPrimeCnt();
                counter += primeCnt[i].returnPrimeCnt();
            }

            long endTime = System.currentTimeMillis();
            long timeDiff = endTime - startTime;
            System.out.println("Entire Threads : " + num_Threads);
            System.out.println();
            
            //(1) execution time of each thread
            System.out.println("(1) [execution time of each thread]");

            for(int i = 0;i<num_Threads;i++) {
                System.out.println("    Thread-"+ (i + 1) + " Execution Time : " + primeCnt[i].returnThreadTime() + "ms");
            }
            System.out.println();
            
            //(2) program execution time
            System.out.println("(2) [program execution time]");
            System.out.println("    Program Execution Time : " + timeDiff + "ms");
            System.out.println();
            
          //(3) prime number
          //쓰레드가 수행하는 수의 범위도 같이 보여
            System.out.println("(3) [the number of prime numbers]");
            for(int i = 0; i < num_Threads; i++) {
                System.out.println("    Thread-" + (i + 1) + " Range: " + primeCnt[i].returnStart() + " to " + (primeCnt[i].returnEnd() - 1) + ", Prime Count: " + eachPrimeCnt[i]);
            }
            
            System.out.println();
            
            System.out.println("1..." + (NUM_END-1) + " prime# counter = " + counter);
            
            System.out.println();
            System.out.println("------------------------------");
            System.out.println();
        }
    }


    // 소수의 개수를 구하기 위한 멀티 쓰레딩 작업을 수행한다.
    private static class PrimeCounter implements Runnable {
        private int start;
        private int end;
        private int count;
        private long threadTime;


        public PrimeCounter(int start, int end) {
            this.start = start;
            this.end = end;
            this.count = 0;
        }

        public int returnPrimeCnt() {
            return count;
        }

        public long returnThreadTime() {
            return threadTime;
        }
        
        public int returnStart() {
            return start;
        }

        public int returnEnd() {
            return end;
        }


        public void run() {
            long startThreadTime = System.currentTimeMillis();
            for (int i = start; i < end; i++) {
                if (isPrime(i)) {
                	count ++;
                }
            }
            long endThreadTime = System.currentTimeMillis();
            threadTime = endThreadTime - startThreadTime;
        }
        private static boolean isPrime(int x) {
            if (x <= 1) {
                return false;
            }

            // 소수는  재곱근 이상의 두 정수의 곱으로 나타낼 수 없다. 그래서 판별식을 제곱근 이하 모든 정수로 나누어 떨어지는지 확인하도록 수정했다.
            for (int i = 2; i *i<= x; i++) {
                if (x % i == 0) {
                    return false;
                }
            }
            return true;
        }
    }
}
