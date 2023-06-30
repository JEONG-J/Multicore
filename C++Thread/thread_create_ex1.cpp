#include <iostream>
#include <thread>

long OddSum=0;
long EvenSum=0;

void findEven(long start, long end)
{
        for (long i=start;i<=end;i++) if ((i&1)==0) EvenSum += i;
}

void findOdd(long start, long end)
{
        for (long i=start;i<=end;i++) if ((i&1)==1) OddSum += i;
}


// Functor (Function Object)
class FindOddFunctor {
        public:
                void operator()(int start, int end) {
                        for (int i=start;i<=end;i++) if ((i&1)==1) OddSum += i;
                }
};

// class member function
class FindOddClass {
	public:
		void myrun(int start, int end) {
                        for (int i=start;i<=end;i++) if ((i&1)==1) OddSum += i;
		}
};


int main()
{
        long start = 0, end = 1000;

        std::thread t1(findEven, start, end);

        //std::thread t2(findOdd, start, end);   // (method 1) create thread using function pointer

	FindOddFunctor findoddfunctor;
        std::thread t2(findoddfunctor, start, end); // (method 2) create thread using functor

	//FindOddClass oddObj;
        //std::thread t2(&FindOddClass::myrun, &oddObj, start, end); // (method 3) create thread using member function of an object

        t1.join(); // wait until thread t1 is finished.
        t2.join(); // wait until thread t2 is finished.

        std::cout << "OddSum: " << OddSum << std::endl;
        std::cout << "EvenSum: " << EvenSum << std::endl;

        return 0;
}
