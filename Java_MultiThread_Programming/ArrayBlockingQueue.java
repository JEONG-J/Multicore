import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class ArrayBlockingQueueExample {

    public static void main(String[] args) {
        // Let's create a blocking queue that can hold at most 5 elements.
        BlockingQueue queue = new ArrayBlockingQueue<String>(3);
        
        // The two threads will access the same queue, in order
        // to test its blocking capabilities.
        Thread producer = new Thread(new Producer(queue));
        Thread consumer = new Thread(new Consumer(queue));
        
        producer.start();
        consumer.start();
    }
}

class Producer implements Runnable{
    
    private BlockingQueue queue;
    
    public Producer(BlockingQueue queue) {
        this.queue = queue;
    }

    @Override
    public void run() {
        
        // We are adding elements using offer() in order to check if
        // it actually managed to insert them.
        for (int i = 0; i < 1000; i++) {
            System.out.println("Producer " +i);
            try {
                 System.out.println("Trying to put: (String " + i +").");  
                 queue.put("String ("+i+")");
                 System.out.println("Put (String " + i +") Done. Queue Size:"+queue.size());
            
            
                Thread.sleep((int)(Math.random()*2000));
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}

class Consumer implements Runnable{
    
    private BlockingQueue queue;
    
    public Consumer(BlockingQueue queue) {
        this.queue = queue;
    }

    @Override
    public void run() {
        String str;
        
        // As long as there are empty positions in our array,
        // we want to check what's going on.
        //while (queue.remainingCapacity() > 0) {
        for (int i = 0; i < 1000; i++) {
            System.out.println("Consumer " +i);
            try {
                System.out.println("Trying to take()");  
                str=(String)queue.take();
                System.out.println("take: ("+str+ ") Done. Queue Size:"+queue.size());
                Thread.sleep((int)(Math.random()*2000));
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}