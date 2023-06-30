package Problem1_cyclic;

public class pc_stack_cyclic {
	

	private static final int NUM_END = 200000;
    private static final int[] NUM_THREADS = {1, 2, 4, 6, 8, 10, 12, 14, 16, 32,100,500};

    public static void main(String[] args) {

        for (int num_Threads : NUM_THREADS) {
        	int[] eachPrimeCnt = new int[num_Threads];
        	int counter = 0;
            long startTime = System.currentTimeMillis();


            Thread[] threads = new Thread[num_Threads];
      
            PrimeCounter[] primeCnt = new PrimeCounter[num_Threads];
            for (int i = 0; i < num_Threads; i++) {
                primeCnt[i] = new PrimeCounter(i, num_Threads);
                threads[i] = new Thread(primeCnt[i]);
                threads[i].start();
            }
   
            for (int i = 0; i < num_Threads; i++) {
                try {
                    threads[i].join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                eachPrimeCnt[i] = primeCnt[i].returnPrimeCnt();
                counter += eachPrimeCnt[i];
            }

            long endTime = System.currentTimeMillis();
            long timeDiff = endTime - startTime;
           
            System.out.println("Entire Threads : " + num_Threads);
            System.out.println();
            
   
            System.out.println("(1) [execution time of each thread]");

            for(int i = 0;i<num_Threads;i++) {
            	System.out.println("    Thread-"+ (i + 1) + " Execution Time : " + primeCnt[i].returnThreadtime() + "ms");
            }
            System.out.println();
            
        
            System.out.println("(2) [program execution time]");
            System.out.println("    Program Execution Time : " + timeDiff + "ms");
            System.out.println();
            
      
            System.out.println("(3) [the number of prime numbers]");
            for(int i = 0; i < num_Threads; i++) {
                System.out.print("    Thread-" + (i + 1) + " Ranges: ");
                int current = primeCnt[i].returnStart() * 10 + 1;
                while (current < 50) {
                    int endPoint = Math.min(current + 10, 50);
                    System.out.print("{" + current + "~" + (endPoint - 1) + "}");
                    if (current + primeCnt[i].returnStep() * 10 < 50) {
                        System.out.print(", ");
                    }
                    current += primeCnt[i].returnStep() * 10;
                }
                System.out.print("......");
                System.out.println(" Prime Count: " + eachPrimeCnt[i]);
            }
            
            System.out.println(); 
            System.out.println("1..." + (NUM_END-1) + " prime# counter = " + counter);
            
            System.out.println();
            System.out.println("------------------------------");
            System.out.println();
        }
    }


    private static class PrimeCounter implements Runnable {
        private int start;
        private int	step;
        private int count;
        private long eachThreadtime;

        public PrimeCounter(int start, int step) {
            this.start = start;
            this.step = step;
            this.count = 0;
        }

        public int returnPrimeCnt() {
            return count;
        }
        
        public long returnThreadtime() {
        	return eachThreadtime;
        }
        
        public int returnStart() {
            return start;
        }

        public int returnStep() {
            return step;
        }

       
        public void run() {
            long startThreadtime = System.currentTimeMillis();
            int current = start * 10 + 1;
            while (current < NUM_END) {
                int endPoint = Math.min(current + 10, NUM_END);
                for (int i = current; i < endPoint; i++) {
                    if (isPrime(i)) {
                    	count ++;
                    }
                }
                current += step * 10;
            }
            long endThreadtime = System.currentTimeMillis();
            eachThreadtime = endThreadtime - startThreadtime;
        }



        private static boolean isPrime(int x) {
            if (x <= 1) {
                return false;
            }

      
            for (int i = 2; i *i<= x; i++) {
                if (x % i == 0) {
                    return false;
                }
            }
            return true;
        }
    }
}