package Prob3_3;

import java.util.concurrent.atomic.AtomicInteger;

public class ex3 {
    public static void main(String[] args) {
        Counter counter = new Counter();

        Thread increment = new Thread(new Increment(counter));
        Thread decrement = new Thread(new Decrement(counter));

        increment.start();
        decrement.start();

        try {
            increment.join();
            decrement.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println("-----------");
        System.out.println("최종 카운터 값: " + counter.getValue());
    }
}

class Counter {
    private AtomicInteger value = new AtomicInteger(0);

    public void increment() {
        value.addAndGet(1);
    }

    public void decrement() {
        value.getAndAdd(-1);
    }

    public void setValue(int newValue) {
        value.set(newValue);
    }

    public int getValue() {
        return value.get();
    }
}

class Increment implements Runnable {
    private final Counter counter;

    Increment(Counter counter) {
        this.counter = counter;
    }

    @Override
    public void run() {
        for (int i = 0; i < 10; i++) {
            counter.increment();
            System.out.println("증가: " + counter.getValue());
        }
    }
}

class Decrement implements Runnable {
    private final Counter counter;

    Decrement(Counter counter) {
        this.counter = counter;
    }

    @Override
    public void run() {
        for (int i = 0; i < 10; i++) {
            counter.decrement();
            System.out.println("감소: " + counter.getValue());
        }
    }
}
