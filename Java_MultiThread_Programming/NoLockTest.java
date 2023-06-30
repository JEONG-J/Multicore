import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class NoLockTest {

    public static void main(String[] args) {
        CounterLock c_lock = new CounterLock();
	int inc_num = 10001234;
	int dec_num = 10000000;

	long start = System.currentTimeMillis();
	Thread p = new Thread (new Producer(c_lock, inc_num));
	p.start();

	Thread c = new Thread (new Consumer(c_lock, dec_num));
	c.start();
        try {
	   p.join();
        } catch (InterruptedException e) {}

        try {
	    c.join();
        } catch (InterruptedException e) {}
	long finish = System.currentTimeMillis();
        System.out.println(inc_num+" inc() calls, "+dec_num+" dec() calls = " + c_lock.getCount());
        System.out.println("No-Lock time: "+(finish-start)+"ms");
     }
}

class Producer implements Runnable{
    
    private CounterLock myCounter;
    int num;
    
    public Producer(CounterLock x, int Num) {
	this.num=Num;
        this.myCounter = x;
    }

    @Override
    public void run() {
        for (int i = 0; i < num; i++) {
		myCounter.inc();
        }
    }
}

class Consumer implements Runnable{
    
    private CounterLock myCounter;
    int num;
    
    public Consumer(CounterLock x, int Num) {
	this.num=Num;
	this.myCounter = x;
    }

    @Override
    public void run() {
        for (int i = 0; i < num; i++) {
		myCounter.dec();
        }
    }
}

class CounterLock {

    private long count = 0;

    public void inc() {
            this.count++;
    }

    public void dec() {
            this.count--;
    }

    public long getCount() {
         return this.count;
    }
}
