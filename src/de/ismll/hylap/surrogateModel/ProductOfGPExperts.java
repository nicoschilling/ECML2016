package de.ismll.hylap.surrogateModel;

import de.ismll.core.DenseInstance;
import de.ismll.core.Instance;
import de.ismll.core.InstanceUtils;
import de.ismll.core.Instances;
import de.ismll.core.SparseInstance;
import de.ismll.core.regression.GaussianProcessRegression;

public class ProductOfGPExperts implements SurrogateModel, OnlineLearnable
{

	private boolean isBCM = false;
	private boolean useDifferentialEntropyBetas = false;
	private boolean normalizeInstances = false;

	public static final int SINGLE_EXPERT = 0;
	public static final int ALL_EXPERTS = 1;

	public double[] targetMeanAndSd = new double[2];

	public double[] betas;
	public GaussianProcessRegression[] experts;

	Instances[] splitTrainData;
	Instances trainData;

	public int[] numberOfTrainInstancesPerExpert;
	public int numberOfExperts;

	public Instances targetInstancesSeen;
	public Instances targetInstancesScaled;

	int mode = 1;
	boolean firstInstance = true;
	int numValues = 0;

	GaussianProcessRegression targetGP;
	public double targetBeta = 0.5;

	public ProductOfGPExperts(Instances[] trainData, int mode, boolean normalizeInstances, boolean isBCM, boolean useDifferentialEntropyBetas)
	{
		// Set booleans accordingly:
		this.normalizeInstances = normalizeInstances;
		this.isBCM = isBCM;
		this.useDifferentialEntropyBetas = useDifferentialEntropyBetas;
		this.mode = mode;

		this.targetGP = new GaussianProcessRegression();
		this.numValues = trainData[0].numValues();
		this.numberOfExperts = trainData.length;

		// Compute betas depending on single or all experts
		double beta = 1;

		if(isBCM)
		{
			beta = 1;
		}
		else
		{
			if(mode == SINGLE_EXPERT)
			{
				beta = 1d / (2 * trainData.length);
			}
			else if(mode == ALL_EXPERTS)
			{
				beta = 1d / (trainData.length);
			}
		}

		// initialize objects of instances seen and scaled for scaling test points labels
		this.targetInstancesSeen = new Instances(this.numValues);
		this.targetInstancesScaled = new Instances(this.numValues);

		// normalize train Data eventually and copy it to a new Instances object
		this.numberOfTrainInstancesPerExpert = new int[trainData.length];
		this.splitTrainData = new Instances[trainData.length];
		for(int expert = 0; expert < this.splitTrainData.length; expert++)
		{
			this.splitTrainData[expert] = new Instances(numValues);
			this.numberOfTrainInstancesPerExpert[expert] = trainData[expert].numInstances();
			if(this.normalizeInstances)
			{
				// System.out.println("buh");
				double[] estimateMeanAndSd = this.estimateMeanAndSd(trainData[expert]);
				for(Instance instance : trainData[expert])
				{
					this.splitTrainData[expert].add(this.getNormalizedInstance(estimateMeanAndSd[0], estimateMeanAndSd[1], instance));
				}
			}
			else
			{
				this.splitTrainData[expert].addAll(trainData[expert]);
			}
		}

		// Initialisiere alle GP experten
		this.betas = new double[this.numberOfExperts];
		this.experts = new GaussianProcessRegression[this.numberOfExperts];
		for(int expert = 0; expert < this.experts.length; expert++)
		{
			this.experts[expert] = new GaussianProcessRegression();
			this.experts[expert].setLearnKernelParameters(true);
			this.experts[expert].train(this.splitTrainData[expert]);
			this.betas[expert] = beta;
		}

	}

	@Override
	public void onlineUpdate(Instance instance)
	{
		if(this.normalizeInstances)
		{
			// Add the new instance and normalize all seen instances:
			this.targetInstancesSeen.add(instance);
			this.targetInstancesScaled = new Instances(this.numValues);
			this.targetMeanAndSd = this.estimateMeanAndSd(this.targetInstancesSeen);
			Instance scaledInstance = this.getNormalizedInstance(this.targetMeanAndSd[0], this.targetMeanAndSd[1], instance);
			for(Instance targetInstance : this.targetInstancesSeen)
			{
				this.targetInstancesScaled.add(this.getNormalizedInstance(this.targetMeanAndSd[0], this.targetMeanAndSd[1], targetInstance));
			}

			if(this.mode == ALL_EXPERTS)
			{
				// Change the labels of the new instances for each expert
				for(int expert = 0; expert < this.experts.length; expert++)
				{
					int startIdx = this.numberOfTrainInstancesPerExpert[expert];
					for(int instanceIdx = 0; instanceIdx < this.targetInstancesSeen.numInstances() - 1; instanceIdx++)
					{
						this.experts[expert].instances.instance(startIdx + instanceIdx).setTarget(
								this.targetInstancesScaled.instance(instanceIdx).target());
					}
					this.experts[expert].onlineUpdate(scaledInstance);
				}
			}
			else if(this.mode == SINGLE_EXPERT)
			{
				// Change the labels of the new instances for each expert
//				for(int expert = 0; expert < this.experts.length; expert++)
//				{
//					int startIdx = this.numberOfTrainInstancesPerExpert[expert];
//					for(int instanceIdx = 0; instanceIdx < this.targetInstancesSeen.numInstances() - 1; instanceIdx++)
//					{
//						this.experts[expert].instances.instance(startIdx + instanceIdx).setTarget(
//								this.targetInstancesScaled.instance(instanceIdx).target());
//					}
//					this.experts[expert].onlineUpdate(scaledInstance);
//				}
				// If it is the first Instance just add the scaled Target instance to the GP Training Data and call train()
				if(this.firstInstance)
				{
					Instances instances = new Instances(this.numValues);
					instances.add(scaledInstance);
					targetGP.train(instances);
					this.firstInstance = false;
				}
				else
				// not the first instance, so update the instances in the target GP, and then online update with the scaled Instance
				{
					for(int instanceIdx = 0; instanceIdx < this.targetInstancesSeen.numInstances() - 1; instanceIdx++)
					{
						targetGP.instances.instance(instanceIdx).setTarget(this.targetInstancesScaled.instance(instanceIdx).target());
						// System.out.println("buh");
					}
					targetGP.onlineUpdate(scaledInstance);
				}
			}
			else
			{
				System.err.println("Unknown mode, as it is set to: " + mode);
			}

		}
		else
		{
			if(this.mode == ALL_EXPERTS)
			{
				// Simply update all experts with the new instance:
				for(int expert = 0; expert < this.experts.length; expert++)
				{
					this.experts[expert].onlineUpdate(instance);
				}
			}
			else if(this.mode == SINGLE_EXPERT)
			{
				// if first instance, learn the GP on the one instance using train()
				if(this.firstInstance)
				{
					Instances instances = new Instances(this.numValues);
					instances.add(instance);
					targetGP.train(instances);
					this.firstInstance = false;
				}
				else
				// not first instance so update target GP using online update
				{
					targetGP.onlineUpdate(instance);
				}
				// Update all experts
				for(int expert = 0; expert < this.experts.length; expert++)
				{
					this.experts[expert].onlineUpdate(instance);
				}
			}
			else
			{
				System.err.println("Unknown mode, as it is set to: " + mode);
			}
		}

	}

	@Override
	public void train(Instances instances)
	{
		System.out.println("Ich bin nicht konfiguriert!");
	}

	public double[] predictWithUncertainty(Instance instance)
	{
		double ret = 0;
		double mean = 0;
		double var = 0;
		double precision = 0;
		double[] predicted = null;
		double sumOfPrecisions = 0;
		double sumOfBetas = 0;
		if(this.mode == ALL_EXPERTS)
		{
			for(int expert = 0; expert < this.experts.length; expert++)
			{
				predicted = this.experts[expert].predictWithUncertainty(instance);
				mean = predicted[0];
				var = predicted[1] * predicted[1];
				if(this.useDifferentialEntropyBetas)
				{
					this.betas[expert] = -0.5 * Math.log(var);
				}
				precision = 1d / var;
				ret += this.betas[expert] * precision * mean;
				sumOfPrecisions += this.betas[expert] * precision;
				sumOfBetas += this.betas[expert];
			}
			double divisor = 0;
			if(this.isBCM)
			{
				// return mean divided by rbcm sum of precisions
				divisor = sumOfPrecisions + (1 - sumOfBetas);
				ret = ret / divisor;
			}
			else
			{
				// just standard poe
				divisor = sumOfPrecisions;
				ret = ret / divisor;
			}
//			System.out.println("Target: " + instance.target());
//			System.out.println("Predicted: " + ret);
//			System.out.println("Precision: " +divisor);
//			System.out.println("-----------------------------------------------------------------------");
			return new double[] { ret, 1d / (Math.sqrt(divisor)) };
		}
		else if(this.mode == SINGLE_EXPERT)
		{
			for(int expert = 0; expert < this.experts.length; expert++)
			{
				predicted = this.experts[expert].predictWithUncertainty(instance);
				mean = predicted[0];
				var = predicted[1] * predicted[1];
				if(this.useDifferentialEntropyBetas)
				{
					this.betas[expert] = -0.5 * Math.log(var);
				}
				precision = 1d / var;
				ret += this.betas[expert] * precision * mean;
				sumOfPrecisions += this.betas[expert] * precision;
				sumOfBetas += this.betas[expert];
			}
			if(!this.firstInstance)
			{
				double[] targetRet = this.targetGP.predictWithUncertainty(instance);
				double targetMean = targetRet[0];
				double targetVar = targetRet[1] * targetRet[1];
				double targetPrecision = 1d / targetVar;
				ret += this.targetBeta * targetPrecision * targetMean;

				sumOfBetas += this.targetBeta;
				sumOfPrecisions += this.targetBeta * targetPrecision;
			}
			double divisor = 0;
			if(this.isBCM)
			{
				// return mean divided by rbcm sum of precisions
				divisor = sumOfPrecisions + (1 - sumOfBetas);
				ret = ret / divisor;
			}
			else
			{
				// just standard poe
				divisor = sumOfPrecisions;
				ret = ret / divisor;
			}
			this.firstInstance = true;
			return new double[] { ret, 1d / (Math.sqrt(divisor)) };
		}
		else
		{
			return null;
		}

	}

	@Override
	public double[] predict(Instance instance)
	{
		return this.predictWithUncertainty(instance);
	}

	public double[] estimateMeanAndSd(Instances instances)
	{
		if(instances.numInstances() == 1)
		{
			return new double[] { instances.instance(0).target(), 0 };
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
		meanSd[1] = Math.sqrt(meanSd[1] / (instances.numInstances() - 1));

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
