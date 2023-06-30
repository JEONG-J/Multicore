package Prob3_4;

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class ex4 {
    public static void main(String[] args) {
        int numberOfThreads = 5;
        CyclicBarrier barrier = new CyclicBarrier(numberOfThreads, () -> System.out.println("All threads have arrived"));

        for (int i = 0; i < numberOfThreads; i++) {
            new Thread(new Worker(barrier, i)).start();
        }
    }
}

class Worker implements Runnable {
    private final CyclicBarrier barrier;
    private final int id;

    Worker(CyclicBarrier barrier, int id) {
        this.barrier = barrier;
        this.id = id;
    }

    @Override
    public void run() {
        try {
            System.out.println("Thread " + id + " start!");
            Thread.sleep((long) (Math.random() * 3000));

            System.out.println("Thread " + id + " finish, Arrive at the barrier");
            barrier.await();

            System.out.println("Thread " + id + " -> Pass");
        } catch (InterruptedException | BrokenBarrierException e) {
            e.printStackTrace();
        }
    }
}
