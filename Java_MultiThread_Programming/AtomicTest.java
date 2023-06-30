import java.util.concurrent.atomic.*;

public class AtomicTest {

    public static void main(String[] args) {
        AtomicInteger atomic_int = new AtomicInteger(0);
	int inc_num = 10001234;
	int dec_num = 10000000;

	long start = System.currentTimeMillis();
	Thread p = new Thread (new Producer(atomic_int, inc_num));
	p.start();
	Thread c = new Thread (new Consumer(atomic_int, dec_num));
	c.start();
        try {
	   p.join();
        } catch (InterruptedException e) {}

        try {
	    c.join();
        } catch (InterruptedException e) {}
	long finish = System.currentTimeMillis();
	System.out.println(inc_num+" inc() calls, "+dec_num+" dec() calls = " + atomic_int.get());
	System.out.println("With-Lock Time: "+(finish-start)+"ms");
     }
}

class Producer implements Runnable{
    private AtomicInteger myAtomicCounter;
    int num;
    
    public Producer(AtomicInteger x, int Num) {
	this.num=Num;
        this.myAtomicCounter = x;
    }

    @Override
    public void run() {
        for (int i = 0; i < num; i++) {
		myAtomicCounter.incrementAndGet(); // myAtomicCounter++
        }
    }
}

class Consumer implements Runnable{
    
    private AtomicInteger myAtomicCounter;
    int num;
    
    public Consumer(AtomicInteger x, int Num) {
	this.num=Num;
	this.myAtomicCounter = x;
    }

    @Override
    public void run() {
        for (int i = 0; i < num; i++) {
		myAtomicCounter.decrementAndGet(); // myAtomicCounter--
        }
    }
}
