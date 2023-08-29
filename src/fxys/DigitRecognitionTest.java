package fxys;

import fxys.NeuralNetwork.NeuralNetwork;

import java.io.File;
import java.util.*;

public class DigitRecognition {

    private static final File training_data_file = new File("mnist_train.csv");
    private static final File test_data_file = new File("mnist_test.csv");
    private static final List<int[]> training_data = new ArrayList<>();
    private static final List<int[]> test_data = new ArrayList<>();
    private static NeuralNetwork nn;

    public static void main(String[] args) {
        try {
                extract(training_data_file, training_data);
                extract(test_data_file, test_data);
        } catch (Exception e) {
            e.printStackTrace();
        }
        train();
        test();
    }

    private static void extract(File input, List<int[]> output) throws Exception {
        Scanner data = new Scanner(input);
        data.nextLine();
        while (data.hasNextLine()) {
            String[] temp = data.nextLine().split(",");
            int[] arr = new int[temp.length];
            for(int i = 0; i < temp.length; i++) {
                arr[i] = Integer.parseInt(temp[i]);
            }
            output.add(arr);
        }
        data.close();
    }

    private static void train() {
        nn = new NeuralNetwork(new int[] {784, 100, 10}, 0.01, 0.5, NeuralNetwork.Activation.Leaky_ReLu, NeuralNetwork.Activation.SoftMax, NeuralNetwork.Cost.CrossEntropy, true);
        for(int i = 0; i < 60000; i++) {
            int label = training_data.get(i)[0];
            int[] temp = Arrays.copyOfRange(training_data.get(i), 1, training_data.get(i).length);
            double[] data = new double[temp.length];
            for(int j = 0; j < temp.length; j++) {
                data[j] = (double)temp[j] / 255;
            }
            nn.train(data, target(label));
            System.out.print("\r" + i + " | " + nn.cost());
        }
    }

    private static void test() {
        int correct = 0;
        for(int i = 0; i < 10000; i++) {
            int label = test_data.get(i)[0];
            int[] temp = Arrays.copyOfRange(test_data.get(i), 1, test_data.get(i).length);
            double[] data = new double[temp.length];
            for(int j = 0; j < temp.length; j++) {
                data[j] = (double)temp[j] / 255;
            }
            int result = nn.test(data, target(label));
            if(result == label) correct++;
        }
        System.out.println("\n Test: " + correct + " | " + (double)correct / 100 + "%");
    }

    private static double[] target(int label) {
        double[] target = new double[10];
        Arrays.fill(target, 0);
        for(int i = 0; i < target.length; i++) {
            if(i == label) target[i] = 1;
        }
        return target;
    }
}
