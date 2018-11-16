package multiclass;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class MulticlassPerceptron implements Classifier {

    private final int epochs;
    private double[][] weights;
    private double[][] values;

    public MulticlassPerceptron(String[] options) {
        epochs = Integer.parseInt(options[1]);
    }

    private void initValues(Instances instances) {
        weights = new double[instances.numAttributes()][instances.numClasses()];
        values = new double[instances.numInstances()][instances.numAttributes()];

        for(int i = 0; i < values.length; i++) {
            values[i][0] = 1;
        }

        for(int i = 0; i < instances.numAttributes() - 1; i++) {
            double[] vals = instances.attributeToDoubleArray(i);

            for(int j = 0; j < values.length; j++) {
                values[j][i + 1] = vals[j];
            }
        }
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {

        double[] weights = new double[instances.numAttributes()];
        initValues(instances);


    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }
}
