package fxys.NeuralNetwork;

import java.util.*;
import java.util.stream.IntStream;

public class NeuralNetwork {
    public List<Layer> layers = new ArrayList<>();
    double[][][] weights;
    double[][][] momentumWeights;
    final Random r = new Random();
    double learning_rate;
    double momentum;
    Activation activation;
    Activation activation_output;
    Cost cost_function;
    boolean gaussian;
    public double[] output;
    public double[] target;

    public NeuralNetwork(int[] layers, double learning_rate, double momentum, Activation activation, Activation activation_output, Cost cost_function, boolean use_Gaussian) {
        for (int i : layers) this.layers.add(new Layer(i, r));
        this.learning_rate = learning_rate;
        this.momentum = momentum;
        this.activation = activation;
        this.activation_output = activation_output;
        this.cost_function = cost_function;
        this.gaussian = use_Gaussian;
        this.weights = new double[layers.length][1000][1000];
        this.momentumWeights = new double[layers.length][1000][1000];
        init();
    }

    private void init() {
        for(int i = 0; i < layers.size() - 1; i++) {
            for(int j = 0; j < layers.get(i).size; j++) {
                for(int k = 0; k < layers.get(i + 1).size; k++) {
                    weights[i + 1][j][k] = random();
                    momentumWeights[i + 1][j][k] = 0;
                }
            }
        }
    }

    private void forward() {
        for(int i = 1; i < layers.size(); i++) {
            double[] temp = new double[layers.get(i).size];
            for(int j = 0; j < layers.get(i).size; j++) {
                double sum = layers.get(i).getBias(j);
                for(int k = 0; k < layers.get(i - 1).size; k++) {
                    sum += layers.get(i - 1).getValue(k) * weights[i][k][j];
                }
                if(activation == Activation.SoftMax || (activation_output == Activation.SoftMax && i + 1 == layers.size())) temp[j] = sum;
                else layers.get(i).setValue(j, activation(sum, i + 1 == layers.size()));
            }
            if(activation == Activation.SoftMax || (activation_output == Activation.SoftMax && i + 1 == layers.size())) {
                double sum = Arrays.stream(temp).map(Math::exp).sum();
                for (int k = 0; k < layers.get(i).size; k++) {
                    layers.get(i).setValue(k, activation(Math.exp(temp[k]) / sum, i + 1 == layers.size()));
                }
            }
        }
        output = layers.get(layers.size() - 1).nodes.clone();
    }

    private void backpropagation() {
        double[] delta = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            delta[i] = (target[i] - output[i]) * activationDerivative(output[i], true);
        }

        for (int j = 0; j < layers.get(layers.size() - 1).size; j++) {
            for (int k = 0; k < layers.get(layers.size() - 2).size; k++) {
                double gradient = delta[j] * layers.get(layers.size() - 2).getValue(k);
                momentumWeights[layers.size() - 1][k][j] = momentum * momentumWeights[layers.size() - 1][k][j] + learning_rate * gradient;
                weights[layers.size() - 1][k][j] += momentumWeights[layers.size() - 1][k][j];
            }
            layers.get(layers.size() - 1).setBias(j, layers.get(layers.size() - 1).getBias(j) + learning_rate * delta[j]);
        }

        for (int i = layers.size() - 2; i > 0; i--) {
            double[] deltah = new double[layers.get(i).size];
            for (int j = 0; j < layers.get(i).size; j++) {
                double errorSum = 0;
                for (int k = 0; k < layers.get(i + 1).size; k++) {
                    errorSum += delta[k] * weights[i + 1][j][k];
                }
                deltah[j] = errorSum * activationDerivative(layers.get(i).getValue(j), false);
            }

            for (int j = 0; j < layers.get(i).size; j++) {
                for (int k = 0; k < layers.get(i - 1).size; k++) {
                    double gradient = deltah[j] * layers.get(i - 1).getValue(k);
                    momentumWeights[i][k][j] = momentum * momentumWeights[i][k][j] + learning_rate * gradient;
                    weights[i][k][j] += momentumWeights[i][k][j];
                }
                layers.get(i).setBias(j, layers.get(i).getBias(j) + learning_rate * deltah[j]);
            }

            delta = deltah;
        }
    }

    private double activationDerivative(double x, boolean output) {
        return switch (output ? activation_output : activation) {
            case Sigmoid -> x * (1 - x);
            case ReLu -> x > 0 ? 1 : 0;
            case Leaky_ReLu -> x > 0 ? 1 : 0.01;
            case TanH -> 1 - x * x;
            case SiLu -> x + (1 / (1 + Math.exp(-x))) * (1 - x);
            case SoftMax -> 1;
        };
    }

    private double random() {
        return gaussian ? 2 * r.nextGaussian() - 1 : 2 * r.nextDouble() - 1D;
    }

    public void train(double[] input, double[] target) {
        layers.get(0).setValue(input);
        this.target = target;
        forward();
        backpropagation();
    }

    public int test(double[] input, double[] target) {
        layers.get(0).setValue(input);
        this.target = target;
        forward();
        double[] sort = output.clone();
        Arrays.sort(sort);
        return Arrays.stream(output).boxed().toList().indexOf(sort[output.length - 1]);
    }

    private double activation(double x, boolean output) {
        return switch (output ? activation_output : activation) {
            case Sigmoid -> 1 / (1 + Math.exp(-x));
            case ReLu -> Math.max(0, x);
            case Leaky_ReLu -> Math.max(0.01 * x, x);
            case TanH -> Math.tanh(x);
            case SiLu -> x / (1 + Math.exp(-x));
            case SoftMax -> x;
        };
    }

    public double cost() {
        return switch (cost_function) {
            case MSE -> IntStream.range(0, output.length).mapToDouble(i -> {
                double d = target[i] - output[i];
                return 0.5 * d * d;
            }).sum();
            case CrossEntropy -> IntStream.range(0, output.length).mapToDouble(i -> -target[i] * Math.log(output[i]) - (1 - target[i]) * Math.log(1 - output[i])).sum();
        };
    }

    public enum Activation {
        Sigmoid,
        ReLu,
        Leaky_ReLu,
        SoftMax,
        TanH,
        SiLu
    }

    public enum Cost {
        MSE,
        CrossEntropy
    }

    public static class Layer {
        int size;
        double[] nodes;
        double[] biases;
        Random r;

        public Layer(int size, Random r) {
            this.size = size;
            this.nodes = new double[size];
            this.biases = new double[size];
            this.r = r;
            for(int i = 0; i < size; i++) {
                biases[i] = 2 * r.nextDouble() - 1;
            }
        }

        public double getValue(int i) {
            return this.nodes[i];
        }

        public void setValue(int i, double value) {
            this.nodes[i] = value;
        }

        public void setValue(double[] nodes) {
            this.nodes = nodes;
        }

        public double getBias(int i) {
            return this.biases[i];
        }

        public void setBias(int i, double value) {
            this.biases[i] = value;
        }
    }
}