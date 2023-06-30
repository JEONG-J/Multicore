#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

int inc_num=10001234;
int dec_num=10000000;

class CountLock {
	int count;
public:
	CountLock() : count(0) {}
	int getCount() 
	{
		return count;
	}
	
	void inc()
	{
		count++;
	}

	void dec()
	{
		count--;
	}
};
 
class Producer
{
	CountLock& c_lock;
public:
	Producer(CountLock& clock): c_lock(clock) {
	}

	void run() {
		for (int i=0;i<inc_num;i++) c_lock.inc();
	}
};

class Consumer
{
	CountLock& c_lock;
public:
	Consumer(CountLock& clock): c_lock(clock) {
	}

	void run() {
		for (int i=0;i<dec_num;i++) c_lock.dec();
	}
};
 
int main()  
{
    CountLock count_lock;
    Producer p(count_lock);
    Consumer c(count_lock);
    std::thread threadP(&Producer::run,&p);
    std::thread threadC(&Consumer::run,&c);
    threadP.join();    
    threadC.join();    
    std::cout<<"after main join count:"<<count_lock.getCount()<<std::endl;
    return 0;
}