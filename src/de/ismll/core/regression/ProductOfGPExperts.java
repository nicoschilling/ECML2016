package de.ismll.core.regression;

import de.ismll.core.DenseInstance;
import de.ismll.core.Instance;
import de.ismll.core.InstanceUtils;
import de.ismll.core.Instances;
import de.ismll.core.SparseInstance;

public class ProductOfGPExperts 
{

	private boolean isBCM = false;
	private boolean useDifferentialEntropyBetas = false;
	
	public int numberOfExperts;
	
	public double[] betas;
	
	public GaussianProcessRegression[] experts;
	
	Instances[] splitTrainData;
	Instances trainData;
	
	public int[] numberOfTrainInstancesPerExpert;
	
	public boolean normalizeInstances = false;
	
	
//	public ProductOfGPExperts(int numberOfExperts, Instances trainData, double beta)
//	{
//		this.numberOfExperts = numberOfExperts;
//		this.betas = new double[this.numberOfExperts];
//		
//		this.experts = new GaussianProcessRegression[this.numberOfExperts];
//		this.splitTrainData = new Instances[this.numberOfExperts];
//		for (int expert = 0; expert < this.experts.length; expert++) {
//			this.experts[expert] = new GaussianProcessRegression();
//			this.experts[expert].setLearnKernelParameters(true);
//			this.splitTrainData[expert] = new Instances(trainData.numValues());
//			this.betas[expert] = beta;
//		}
//		
//		int numOfMinimumInstances = trainData.numInstances()/this.numberOfExperts;
//		trainData.shuffle(Random.nextInt(100000));
//		int instanceIdx = 0;
//		for (int expert = 0; expert < this.experts.length-1; expert++) {
//			for (int idx = 0; idx < numOfMinimumInstances; idx++) {
//				this.splitTrainData[expert].add(trainData.instance(idx+instanceIdx));
//			}
//			instanceIdx += numOfMinimumInstances;
//		}
//		for (int idx = instanceIdx; idx < trainData.numInstances(); idx++) {
//			this.splitTrainData[this.experts.length-1].add(trainData.instance(idx));
//		}
//		
//		for (int expert = 0; expert < this.experts.length; expert++) {
//			this.experts[expert].train(this.splitTrainData[expert]);
//		}
//		
//		
//	}
	
	
	
	public ProductOfGPExperts(int numberOfExperts, Instances[] trainData, double beta, boolean normalizeInstances,
			boolean isBCM, boolean useDifferentialEntropyBetas)
	{
		
		this.normalizeInstances = normalizeInstances;
		int numValues = trainData[0].numValues();
		
		this.numberOfTrainInstancesPerExpert = new int[trainData.length];
		this.splitTrainData = new Instances[trainData.length];
		for (int expert = 0; expert < this.splitTrainData.length; expert++) {
			this.splitTrainData[expert] = new Instances(numValues);
			this.numberOfTrainInstancesPerExpert[expert] = trainData[expert].numInstances();
			if (this.normalizeInstances) {
//				System.out.println("buh");
				double[] estimateMeanAndSd = this.estimateMeanAndSd(trainData[expert]);
				for (Instance instance : trainData[expert]) {
					this.splitTrainData[expert].add(this.getNormalizedInstance(estimateMeanAndSd[0], estimateMeanAndSd[1], instance));
				}
			}
			else {
				this.splitTrainData[expert].addAll(trainData[expert]);
			}
		}
		
		this.numberOfExperts = numberOfExperts;
		
		this.betas = new double[this.numberOfExperts];
		
		this.experts = new GaussianProcessRegression[this.numberOfExperts];
		for (int expert = 0; expert < this.experts.length; expert++) {
			this.experts[expert] = new GaussianProcessRegression();
			this.experts[expert].setLearnKernelParameters(true);
			this.experts[expert].train(this.splitTrainData[expert]);
			this.betas[expert] = beta;
		}
		
	}
	
	

	public double predict(Instance instance)
	{
		double ret = 0;
		double mean = 0;
		double var = 0;
		double precision = 0;
		double[] predicted = null;
		double sumOfPrecisions = 0;
		double sumOfBetas = 0;
		
		for (int expert = 0; expert < this.experts.length; expert++) {
			predicted = this.experts[expert].predictWithUncertainty(instance);
			mean = predicted[0];
			var = predicted[1]*predicted[1];
			if (this.useDifferentialEntropyBetas) {
				this.betas[expert] = -0.5*Math.log(var);
			}
			precision = 1d/var;
			ret += this.betas[expert] * precision * mean;
			sumOfPrecisions += this.betas[expert] * precision;
			sumOfBetas += this.betas[expert];
		}
		if (this.isBCM) {
			// return mean divided by rbcm sum of precisions
			double divisor = sumOfPrecisions + (1-sumOfBetas);
			ret = ret/divisor;
		}
		else {
			// just standard poe
			ret = ret/sumOfPrecisions;
		}
		
		return ret;
	}

	public double[] predict(Instances instances)
	{
		int i = 0;
		double[] ret = new double[instances.numInstances()];
		for (Instance instance : instances) {
			ret[i] = this.predict(instance);
			i++;
		}
		return ret;
	}
	
	public double[] predictWithUncertainty(Instance instance) {
		double ret = 0;
		double mean = 0;
		double var = 0;
		double precision = 0;
		double[] predicted = null;
		double sumOfPrecisions = 0;
		double sumOfBetas = 0;
		for (int expert = 0; expert < this.experts.length; expert++) {
			predicted = this.experts[expert].predictWithUncertainty(instance);
			mean = predicted[0];
			var = predicted[1]*predicted[1];
			if (this.useDifferentialEntropyBetas) {
				this.betas[expert] = -0.5*Math.log(var);
			}
			precision = 1d/var;
			ret += this.betas[expert] * precision * mean;
			sumOfPrecisions += this.betas[expert] * precision;
			sumOfBetas += this.betas[expert];
		}
		double divisor = 0;
		if (this.isBCM) {
			// return mean divided by rbcm sum of precisions
			divisor = sumOfPrecisions + (1-sumOfBetas);
			ret = ret/divisor;
		}
		else {
			// just standard poe
			divisor = sumOfPrecisions;
			ret = ret/sumOfPrecisions;
		}
		return new double[] {ret, 1d/(Math.sqrt(divisor))};
	}
	
	public double[] estimateMeanAndSd(Instances instances)
	{
		if (instances.numInstances() == 1) {
			return new double[] {instances.instance(0).target() , 0 };
		}
		double[] meanSd = new double[2];
		for(int j = 0; j < instances.numInstances(); j++)
		{
			meanSd[0] += instances.instance(j).target();
		}
		meanSd[0] /= instances.numInstances();
		for(int j = 0; j < instances.numInstances(); j++)
		{
			meanSd[1] += Math.pow(instances.instance(j).target() - meanSd[0], 2);
		}
		meanSd[1] = Math.sqrt(meanSd[1] / (instances.numInstances()-1));

		if(meanSd[1] == 0)
			meanSd[1] = 1;

		return meanSd;
	}
	
	public double normalize(double d, double mean, double sd)
	{
		return (d - mean) / (sd == 0 ? 1 : sd);
	}
	
	/**
	 * z-normalize for a given mean and standard deviation
	 */
	public Instance getNormalizedInstance(double mean, double sd, Instance instance)
	{
		if(instance instanceof DenseInstance)
		{
			return InstanceUtils.createDenseInstance(this.normalize(instance.target(), mean, sd), instance.getValues());
		}
		else if(instance instanceof SparseInstance)
		{
			return InstanceUtils.createSparseInstance(this.normalize(instance.target(), mean, sd), instance.getValues(), instance.getKeys());
		}
		else
			throw new IllegalArgumentException("Unsupported instance");
	}

}
