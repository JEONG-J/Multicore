package Problem1_dynamic;

public class pc_dynamic {
     private static final int NUM_END = 200000;
        private static final int[] NUM_THREADS = {1, 2, 4, 6, 8, 10, 12, 14, 16, 32};

        public static void main(String[] args) {
            for (int num_Threads : NUM_THREADS) {
                int[] eachPrimeCnt = new int[num_Threads];
                int counter = 0;
                long startTime = System.currentTimeMillis();

                Thread[] threads = new Thread[num_Threads];
                PrimeCounter[] primeCnt = new PrimeCounter[num_Threads];

                ThreadManager threadManager = new ThreadManager();

                for (int i = 0; i < num_Threads; i++) {
                    primeCnt[i] = new PrimeCounter(threadManager);
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
                
                //(1) execution time of each thread
                System.out.println("(1) [execution time of each thread]");

                for(int i = 0;i<num_Threads;i++) {
                    System.out.println("    Thread-"+ (i + 1) + " Execution Time : " + primeCnt[i].returnThreadtime() + "ms");
                }
                System.out.println();
                
                //(2) program execution time
                System.out.println("(2) [program execution time]");
                System.out.println("    Program Execution Time : " + timeDiff + "ms");
                System.out.println();
                
                //(3) prime number
                System.out.println("(3) [the number of prime numbers]");
                for(int i = 0; i<num_Threads; i++) {
                     System.out.println("    Thread-" + (i + 1) + " Prime Count : " + eachPrimeCnt[i]);
                }
                
                System.out.println(); 
                System.out.println("1..." + (NUM_END-1) + " prime# counter = " + counter);
                
                System.out.println();
                System.out.println("------------------------------");
                System.out.println();
            }
        }

        private static class ThreadManager {
            private int currentTask;
            private int taskSize = 10;

            // 첫 번째 작업 위치를 초기화한다.
            public ThreadManager() {
                currentTask = 1;
            }


            // 다중 쓰레드 환경에서 안전하게 작업을 가져올 수 있도록한다.
            // 현재 작업의 시작 위치 반환 -> 처리 완료시, size만큼 증가시켜 다음 작업 시작 위치로 이동
            public synchronized int returnNextTask() {
                int task = currentTask;
                currentTask += taskSize;
                return task;
            }
        }

        private static class PrimeCounter implements Runnable {
            private ThreadManager threadManager;
            private int count;
            private long eachThreadtime;

            public PrimeCounter(ThreadManager taskManager) {
                this.threadManager = taskManager;
                this.count = 0;
            }

            public int returnPrimeCnt() {
                return count;
            }

            public long returnThreadtime() {
                return eachThreadtime;
            }

            public void run() {
                long startThreadtime = System.currentTimeMillis();

                while (true) {
                    int current = threadManager.returnNextTask();
                    if (current >= NUM_END) {
                        break;
                    }
                    int endPoint = Math.min(current + 10, NUM_END);
                    for (int i = current; i < endPoint; i++) {
                        if (isPrime(i)) {
                        	count++;
                        }
                    }
                }

                long endThreadtime = System.currentTimeMillis();
                eachThreadtime = endThreadtime - startThreadtime;
            }

            private static boolean isPrime(int x) {
                if (x <= 1) {
                    return false;
                }

                for (int i = 2; i * i <= x; i++) {
                    if (x % i == 0) {
                        return false;
                    }
                }
                return true;
            }
        }
    }
