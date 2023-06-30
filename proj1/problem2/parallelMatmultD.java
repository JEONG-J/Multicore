package Problem2;

import java.util.*;
import java.lang.*;

public class parallelMatmultD {
	private static Scanner sc = new Scanner(System.in);
	// 쓰레드가 수행할 작업을 나타내는 객체들을 저장한다.
	private static MatrixMultiplier[] matManager;

    public static void main(String[] args) {
    	
    	//쓰레드 개수 지정
        int thread_no = (args.length == 1) ? Integer.parseInt(args[0]) : 1;

        int[][] a = readMatrix();
        int[][] b = readMatrix();
        
        // 행렬 a,b의 곱을 계산 한다. -> 입력된 쓰레드 개수를 사용하여 병렬 처리한다
        // 계산하는 데 걸리는 시간을 측정한다
        long startTime = System.currentTimeMillis();
        int[][] c = multMatrix(a, b, thread_no);
        long endTime = System.currentTimeMillis();
        

        //(3) sum of all elements in the resulting matrix
        printMatrix(c);
        
        //(2) execution time when using all threads
        System.out.printf("[thread_no]:%2d , [Time]:%4d ms\n", thread_no, endTime - startTime);
        System.out.println();
        
        // (1)the execution time print of each threed
        for (int i = 0; i < thread_no; i++) {
            System.out.printf("Thread-%d Execution Time: %d ms\n", i + 1, matManager[i].returnTime());
        }
    }

    public static int[][] readMatrix() {
        int rows = sc.nextInt();
        int cols = sc.nextInt();
        int[][] result = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = sc.nextInt();
            }
        }
        return result;
    }

    public static void printMatrix(int[][] mat) {
        int rows = mat.length;
        int columns = mat[0].length;
        int sum = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                System.out.printf("%4d ", mat[i][j]);
                sum += mat[i][j];
            }
            System.out.println();
        }
        System.out.println();
        System.out.println("Matrix Sum = " + sum + "\n");
    }
    
    
    //인수를 받아 곱한 결과를 반환한다.
    //곱셈은 병렬 처리되며 입력으로 받은 thread_no에 따라 사용할 쓰레드의 수가 결정된다.
    public static int[][] multMatrix(int[][] a, int[][] b, int thread_no) {
        int rows = a.length;
        int cols = b[0].length;
        
        //첫 번째 행의 수와 두 번째 열의 수가 같은지를 확인하고 다르면 빈행렬 반환한다.
        if (rows == 0 || a[0].length != b.length) {
            return new int[0][0];
        }
        
        int[][] result = new int[rows][cols];
        matManager = new MatrixMultiplier[thread_no]; // Make the workers array a class-level variable
        Thread[] threads = new Thread[thread_no];

        
        
        for (int i = 0; i < thread_no; i++) {
        	// 처리할 행 범위를 지정하여 maxtrixMulti 객체를 생성하고 배열에 저장한다.
        	matManager[i] = new MatrixMultiplier(a, b, result, i * rows / thread_no, (i + 1) * rows / thread_no);
            threads[i] = new Thread(matManager[i]);
            threads[i].start();
        }
        
        // 모든 쓰레드가 완료할 때까지 기다린다.
        for (int i = 0; i < thread_no; i++) {
            try {
                threads[i].join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        return result;
    }

    
    // 각 쓰레드는 행렬 곱셈 작업의 일부를 독립적으로 수행하며, 작업이 완료되면 result 행렬에 결과를 저장한다
    // 전체 작업이 병렬로 수행된다.
    private static class MatrixMultiplier implements Runnable {
        private int[][] a, b, result;
        private int rowStart, rowEnd;
        private long executionTime;

        public MatrixMultiplier(int[][] a, int[][] b, int[][] result, int rowStart, int rowEnd) {
            this.a = a;
            this.b = b;
            this.result = result;
            this.rowStart = rowStart;
            this.rowEnd = rowEnd;
        }
        
        // 쓰레드는 행렬 곱셉을 실행한다.
        public void run() {
            long startTime = System.currentTimeMillis();
            for (int i = rowStart; i < rowEnd; i++) {
                for (int j = 0; j < b[0].length; j++) {
                    for (int k = 0; k < a[0].length; k++) {
                    	synchronized(this) {
                        this.result[i][j] += a[i][k] * b[k][j];
                    }
                    	}
                    	
                }
            }
            long endTime = System.currentTimeMillis();
            executionTime = endTime - startTime;
        }

        public long returnTime() {
            return executionTime;
        }
    }
}

