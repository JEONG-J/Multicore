import java.util.concurrent.*;

public class SemaphoreTest {

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
	System.out.println("With-Lock Time: "+(finish-start)+"ms");
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

    private Semaphore sema = new Semaphore(1);

    public void inc() {
        try {
            sema.acquire();
            this.count++;
        } catch(InterruptedException e) {
        } finally {
            sema.release();
        }
    }

    public void dec() {
        try {
            sema.acquire();
            this.count--;
        } catch(InterruptedException e) {
        } finally {
            sema.release();
        }
    }

    public long getCount() {
        try {
            sema.acquire();
            return this.count;
        } catch(InterruptedException e) {
            return -1;
        } finally {
            sema.release();
        }
    }
}
