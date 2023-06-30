import java.util.concurrent.*;

public class ConcurrentHashMapExample {
   public static void main(String args[]) {

      /* This is how to declare HashMap */
      ConcurrentMap<Integer, String> hmap = new ConcurrentHashMap<Integer, String>();

      /*Adding elements to HashMap*/
      hmap.put(12, "Chaitanya");
      hmap.put(2, "Rahul");
      hmap.put(7, "Singh");
      hmap.put(49, "Ajeet");
      hmap.put(3, "Anuj");

      /* Get values based on key*/
      String var= hmap.get(2);
      System.out.println("Value at index 2 is: "+var);
      String var2= hmap.get(7);
      System.out.println("Value at index 7 is: "+var2);      

      /* Remove values based on key*/
      hmap.remove(3);

      System.out.println(hmap);
   }
}