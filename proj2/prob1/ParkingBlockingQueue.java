package Prob1;

import java.util.concurrent.ArrayBlockingQueue;


class ParkingGarage {
  private ArrayBlockingQueue<Integer> parking;
  
  public ParkingGarage(int places) {
	  parking = new ArrayBlockingQueue<>(places);
	  for(int i = 0; i < places; i++) {
		  parking.offer(i);
	  }
  }
  
  public void enter() {
	  try {
		  parking.take();
	  } catch (InterruptedException e) {
		  e.printStackTrace();
	  }
  }
  
  public void leave() {
	  parking.offer(1);
  }
  
  public int getPlaces() {
	  return parking.size();
  }
}

class Car extends Thread {
  private ParkingGarage parkingGarage;
  
  public Car(String name, ParkingGarage p) {
    super(name);
    this.parkingGarage = p;
    start();
  }

  private void tryingEnter()
  {
      System.out.println(getName()+": trying to enter");
  }


  private void justEntered()
  {
      System.out.println(getName()+": just entered");
  }

  private void aboutToLeave()
  {
      System.out.println(getName()+":                                     about to leave");
  }

  private void Left()
  {
      System.out.println(getName()+":                                     have been left");
  }

  public void run() {
    while (true) {
      try {
        sleep((int)(Math.random() * 10000)); // drive before parking
      } catch (InterruptedException e) {}
      tryingEnter();
      parkingGarage.enter();
      justEntered();
      try {
        sleep((int)(Math.random() * 20000)); // stay within the parking garage
      } catch (InterruptedException e) {}
      aboutToLeave();
      parkingGarage.leave();
      Left();

    }
  }
}


public class ParkingBlockingQueue {
  public static void main(String[] args){
    ParkingGarage parkingGarage = new ParkingGarage(7);
    for (int i=1; i<= 10; i++) {
      Car c = new Car("Car "+i, parkingGarage);
    }
  }
}
