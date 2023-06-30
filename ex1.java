package Prob3;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class ex1 {
	public static void main(String[] args) {
		BlockingQueue<Integer> account = new ArrayBlockingQueue<>(1);
		account.add(0);
		
		Thread deposit = new Thread(new Deposit(account));
		Thread withdraw = new Thread(new Withdraw(account));
		
		deposit.start();
		withdraw.start();
		
	}
}


class Deposit implements Runnable{
	private BlockingQueue<Integer> account;
	
	Deposit(BlockingQueue<Integer> account){
		this.account = account;
	}
	
	@Override
	public void run() {
		for(int i=0;i<=5;i++) {
			try {
				int currentBalance = account.take();
				int newBalance = currentBalance + i * 1000;
				System.out.println("입금 : " + (i * 1000) + "원");
				System.out.println("현재 잔액: " +newBalance + "원");
	            System.out.println("------------------");
				account.put(newBalance);
				Thread.sleep(2000);
			}catch(InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
}


class Withdraw implements Runnable{
	private BlockingQueue<Integer> account;
	
	Withdraw(BlockingQueue<Integer> account){
		this.account = account;
	}
	
	@Override
	public void run() {
		for(int i = 1;i<=5;i++) {
			try {
				int currentBalance = account.take();
				int amountToWithdraw = i * 500;
				System.out.println("출금 : " + amountToWithdraw + "원");
				if(currentBalance >= amountToWithdraw) {
					int newBalance = currentBalance - amountToWithdraw;
					System.out.println("남은금액 : " + newBalance + "원");
					System.out.println("------------------");
					account.put(newBalance);
				} else {
					System.out.println("잔액부족");
					System.out.println("------------------");
					account.put(currentBalance);
				}
				Thread.sleep(2000);
			}catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
}