package Prob3_2;

import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class ex2 {
    public static void main(String[] args) {
        BankAccount account = new BankAccount();

        Thread deposit = new Thread(new Deposit(account));
        Thread withdraw = new Thread(new Withdraw(account));

        deposit.start();
        withdraw.start();
    }
}

class BankAccount {
    private int balance = 0;
    private ReadWriteLock lock = new ReentrantReadWriteLock();

    public void deposit(int amount) {
        lock.writeLock().lock();
        try {
            balance += amount;
            System.out.println("입금: " + amount + "원");
            System.out.println("현재 잔액: " + balance + "원");
            System.out.println("------------------");
        } finally {
            lock.writeLock().unlock();
        }
    }

    public void withdraw(int amount) {
        lock.writeLock().lock();
        try {
            int currentBalance = getBalance();
            System.out.println("출금: " + amount + "원");
            if (currentBalance >= amount) {
                balance -= amount;
                System.out.println("남은 금액: " + (currentBalance - amount) + "원");
                System.out.println("------------------");
            } else {
                System.out.println("잔액부족");
                System.out.println("------------------");
            }
        } finally {
            lock.writeLock().unlock();
        }
    }

    public int getBalance() {
        lock.readLock().lock();
        try {
            return balance;
        } finally {
            lock.readLock().unlock();
        }
    }
}

class Deposit implements Runnable {
    private final BankAccount account;

    Deposit(BankAccount account) {
        this.account = account;
    }

    @Override
    public void run() {
        for (int i = 0; i <= 10; i++) {
            try {
                account.deposit(i * 1000);
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}

class Withdraw implements Runnable {
    private final BankAccount account;

    Withdraw(BankAccount account) {
        this.account = account;
    }

    @Override
    public void run() {
        for (int i = 1; i <= 10; i++) {
            try {
                account.withdraw(i * 500);
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
