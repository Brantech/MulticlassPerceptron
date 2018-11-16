/*
 * University of Central Florida
 * CAP4630 - Fall 2018
 * Author(s): Brandon Gotay
 */
package multiclass;

import java.util.ArrayList;
import java.util.List;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class MulticlassPerceptron implements Classifier {

    private final String file;
    private final int epochs;

    private List<List<Double>> weights = new ArrayList<>();
    private List<List<Double>> values = new ArrayList<>();
    private List<Integer> mapping = new ArrayList<>();

    private int actualEpochs, weightUpdates;

    public MulticlassPerceptron(String[] options) {
        file = options[0];
        epochs = Integer.parseInt(options[1]);
    }

    /**
     * Initializes the weights, values, and mapping, ArrayLists
     *
     * @param instances Weka data instances
     */
    private void initValues(Instances instances) {
        System.out.println("University of Central Florida\nCAP4630 Artificial Intelligence - Fall 2018" +
                                   "\nMulti-Class Perceptron Classifier\nAuthor(s): Brandon Gotay\n");

        // Initialize the weights ArrayList and prefill them with 0's
        for(int i = 0; i < instances.numClasses(); i++) {
            weights.add(new ArrayList<>());
            for(int j = 0; j < instances.numAttributes(); j++) {
                weights.get(i).add(0.0);
            }
        }

        // Initialize the values ArrayList
        for(int i = 0; i < instances.numInstances(); i++) {
            values.add(new ArrayList<>());
        }

        // Add the values in the data set to the lists
        for(int i = 0; i < instances.numAttributes() - 1; i++) {
            if(instances.classIndex() == i) {
                continue;
            }

            double[] vals = instances.attributeToDoubleArray(i);
            mapping.add(i);

            for(int j = 0; j < values.size(); j++) {
                values.get(j).add(vals[j]);
            }
        }

        // Add the bias factor
        for(int i = 0; i < instances.numInstances(); i++) {
            values.get(i).add(1.0);
        }
    }

    /**
     * Uses the weights to make a prediction for the given values
     *
     * @param vals Values to make a prediction for
     * @return Class prediction
     */
    private int predict(List<Double> vals) {
        double max = 0;
        int classGuess = 0;

        // Get the index of the weight ArrayList with the highest dot product with the values
        for(int i = 0; i < weights.size(); i++) {
            double dotProduct = 0;
            for(int j = 0; j < weights.get(i).size(); j++) {
                dotProduct += weights.get(i).get(j) * vals.get(j);
            }

            if(dotProduct > max) {
                max = dotProduct;
                classGuess = i;
            }
        }

        return classGuess;
    }

    /**
     * Updates the weights with the passed in values
     *
     * @param vals Values to add or subtract
     * @param increaseIndex Weight array to add to
     * @param decreaseIndex Weight array to subtract from
     */
    private void updateWeights(List<Double> vals, int increaseIndex, int decreaseIndex) {
        for(int i = 0; i < weights.get(0).size(); i++) {
            double old = weights.get(increaseIndex).get(i);
            weights.get(increaseIndex).set(i, old + vals.get(i));

            old = weights.get(decreaseIndex).get(i);
            weights.get(decreaseIndex).set(i, old - vals.get(i));
        }
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        initValues(instances);
        weightUpdates = 0;

        // Trains the model for the number of epochs or until the predictions are all correct
        for(int i = 0; i < epochs; i++) {
            StringBuilder training = new StringBuilder("Epoch\t" + (i + 1) + " [ ");
            StringBuilder guesses = new StringBuilder();
            int updates = 0;

            // Makes a prediction for every line of data
            for(int j = 0; j < values.size(); j++) {
                int guess = predict(values.get(j));
                if(guess != instances.get(j).classValue()) {
                    guesses.append("0");
                    updates++;
                    updateWeights(values.get(j), (int) instances.get(j).classValue(), guess);
                } else {
                    guesses.append("1");
                }
            }

            training.append(updates);
            training.append("] ");
            training.append(guesses);
            System.out.println(training);

            if(updates == 0) {
                actualEpochs = i + 1;
                break;
            }
            weightUpdates += updates;
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] ret = new double[instance.numClasses()];

        // Gets the attributes of instance and puts their values in an ArrayList
        List<Double> vals = new ArrayList<>();
        for(int i : mapping) {
            vals.add(instance.value(i));
        }
        vals.add(1.0);

        ret[predict(vals)] = 1;
        return ret;
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Source file: ").append(file).append("\n");
        sb.append("Training epoch limit\t: ").append(epochs).append("\n");
        sb.append("Actual training epochs : ").append(actualEpochs).append("\n");
        sb.append("Total # weight updates : ").append(weightUpdates).append("\n\n");
        sb.append("Final weights:\n\n");

        for(int i = 0; i < weights.size(); i++) {
            sb.append("Class ").append(i).append(" weights:\t");
            for(double j : weights.get(i)) {
                sb.append(String.format("%.3f", j)).append(" ");
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}
