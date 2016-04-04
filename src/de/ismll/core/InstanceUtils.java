package de.ismll.core;

import java.util.Random;

import de.ismll.hylap.util.IntRange;

public abstract class InstanceUtils
{

	static final int MULTIBOOST_NUM_HYPERPARAMETERS = 2;
	static final int SVM_NUM_HYPERPARAMETERS = 6;

	public static DenseInstance createDenseInstance(double target, double... values)
	{
		return new DenseInstance(target, values);
	}

	public static SparseInstance createSparseInstance(double target, double[] values, int[] keys)
	{
		return new SparseInstance(target, keys, values);
	}

	/**
	 * Z-Normalizes the whole set of instances.
	 * 
	 * @param instances
	 * @return
	 */
	public static Instances zNormalize(Instances instances)
	{
		return zNormalize(instances, new IntRange("0," + (instances.numValues() - 1)));
	}

	/**
	 * Normalizes the columns of all instances defined in intRange.
	 * 
	 * @param instances
	 * @param intRange
	 * @return
	 */
	public static Instances zNormalize(Instances instances, IntRange intRange)
	{
		Instances ret = new Instances(instances.numValues());
		
		
		for (int i = 0; i < instances.numInstances(); i++) {
			ret.add(copyInstance(instances.instance(i)));
		}

		int[] usedIndexes = intRange.getUsedIndexes();

		double[] variances = new double[usedIndexes.length];
		double[] expectedValues = new double[usedIndexes.length];

		for(int idx = 0; idx < usedIndexes.length; idx++)
		{
			int col = usedIndexes[idx];
			for(int i = 0; i < instances.numInstances(); i++)
			{
				expectedValues[idx] += instances.instance(i).getValue(col);
			}
			expectedValues[idx] = (double) (expectedValues[idx] / instances.numInstances());
		}

		for(int idx = 0; idx < usedIndexes.length; idx++)
		{
			int col = usedIndexes[idx];
			for(int i = 0; i < instances.numInstances(); i++)
			{
				double diff = (instances.instance(i).getValue(col) - expectedValues[idx]);
				variances[idx] += diff * diff;
			}
			variances[idx] = (double) (variances[idx] / (instances.numInstances() - 1));
		}

		for(int idx = 0; idx < usedIndexes.length; idx++)
		{
			int col = usedIndexes[idx];
			if(variances[idx] != 0)
			{
				for(int i = 0; i < ret.numInstances(); i++)
				{
					double value = (ret.instance(i).getValue(col) - expectedValues[idx]) / Math.sqrt(variances[idx]);
					ret.instance(i).setValue(value, col);
				}
			}
		}

		return ret;
	}

	/**
	 * Max-normalizes all the instances.
	 * 
	 * @param instances
	 * @return
	 */
	public static Instances maxNormalize(Instances instances)
	{
		return maxNormalize(instances, new IntRange("0," + (instances.numValues() - 1)));
	}

	/**
	 * Max-normalizes all the columns of all instances defined in intRange.
	 * 
	 * @param instances
	 * @param intRange
	 * @return
	 */
	public static Instances maxNormalize(Instances instances, IntRange intRange)
	{
		System.err.println("max Normalization Not implemented!");
		if(instances.instance(0) instanceof SparseInstance)
		{
			System.err.println("Currently only normalization of Dense Instances is implemented!");
		}
		Instances ret = new Instances(instances.numValues());

		return ret;
	}

	public static double dotProduct(Instance i1, Instance i2)
	{
		double result = 0;
		int index1 = 0, index2 = 0;
		int[] keys1 = i1.getKeys(), keys2 = i2.getKeys();
		double[] values1 = i1.getValues(), values2 = i2.getValues();
		while(index1 < keys1.length && index2 < keys2.length)
		{
			if(keys1[index1] < keys2[index2])
				index1++;
			else if(keys1[index1] > keys2[index2])
				index2++;
			else
				result += values1[index1++] * values2[index2++];
		}
		return result;
	}

	public static double euclideanDistance(Instance i1, Instance i2)
	{
		return euclideanDistance(i1, i2, Integer.MAX_VALUE);
	}

	/**
	 * Euclidean distance between two instances using only the first m attributes.
	 */
	public static double euclideanDistance(Instance i1, Instance i2, int m)
	{
		double result = 0;
		int index1 = 0, index2 = 0;
		int[] keys1 = i1.getKeys(), keys2 = i2.getKeys();
		double[] values1 = i1.getValues(), values2 = i2.getValues();
		while((index1 < keys1.length || index2 < keys2.length) && (index1 < keys1.length && keys1[index1] < m || index2 < keys2.length && keys2[index2] < m))
		{
			if(index1 < keys1.length && keys1[index1] < m && (index2 >= keys2.length || keys2[index2] > keys1[index1]))
				result += Math.pow(values1[index1++], 2);
			else if(index2 < keys2.length && keys2[index2] < m && (index1 >= keys1.length || keys1[index1] > keys2[index2]))
				result += Math.pow(values2[index2++], 2);
			else if(index1 < keys1.length && index2 < keys2.length && keys1[index1] < m && keys2[index2] < m )
				result += Math.pow(values1[index1++] - values2[index2++], 2);
		}
		return Math.sqrt(result);
	}

	public static Instance copyInstance(Instance instance)
	{
		if(instance instanceof DenseInstance)
		{
			return createDenseInstance(instance.target(), instance.values);
		}
		else if(instance instanceof SparseInstance)
		{
			return createSparseInstance(instance.target(), instance.values, instance.getKeys());
		}
		else
		{
			System.err.println("Instance type not supported!");
			return null;
		}
	}

	public static Instances copyInstances(Instances instances)
	{
		Instances ret = new Instances(instances.numValues());
		for(int i = 0; i < instances.numInstances(); i++)
		{
			ret.add(InstanceUtils.copyInstance(instances.instance(i)));
		}
		return ret;
	}

	public static Instances copyInstancesAndAddOrdinalFeatures(Instances instances, int nrOfDatasets, int datasetId)
	{
		Instances ret = new Instances(instances.numValues() + nrOfDatasets);
		for(int i = 0; i < instances.numInstances(); i++)
		{
			ret.add(copyInstanceAndAddOrdinalFeatures(instances.instance(i), nrOfDatasets, datasetId));
		}
		return ret;
	}

	public static void shuffleInstancesArray(Instances[] instances, long seed)
	{
		java.util.Random random = new Random(seed);
		for(int i = instances.length - 1; i > 0; i--)
		{
			int index = random.nextInt(i + 1);
			// Simple swap
			Instances a = instances[index];
			instances[index] = instances[i];
			instances[i] = a;
		}
	}

	public static Instance copyInstanceAndAddOrdinalFeatures(Instance instance, int nrOfDatasets, int datasetId)
	{
		double target = instance.target();
		if(instance instanceof DenseInstance)
		{
			double[] values = instance.getValues();
			double[] newValues = new double[values.length + nrOfDatasets];

			for(int i = 0; i < newValues.length; i++)
			{
				if(i < nrOfDatasets)
				{
					if(i != datasetId)
					{
						newValues[i] = 0;
					}
					else
					{
						newValues[i] = 1;
					}
				}
				else
				{
					newValues[i] = values[i - nrOfDatasets];
				}
			}
			return createDenseInstance(target, newValues);
		}
		else if(instance instanceof SparseInstance)
		{
			int[] keys = instance.getKeys();
			double[] values = instance.getValues();
			int[] newKeys = new int[keys.length + 1];
			double[] newValues = new double[values.length + 1];

			newKeys[0] = datasetId;
			newValues[0] = 1;
			for(int i = 1; i < newKeys.length; i++)
			{
				newKeys[i] = keys[i - 1] + nrOfDatasets;
				newValues[i] = values[i - 1];
			}
			return createSparseInstance(target, newValues, newKeys);
		}
		else
		{
			System.err.println("Instance type not supported!");
			return null;
		}
	}

	public static int getNumberOfHyperparametes(String experiment)
	{
		if(experiment.equals("multiboost"))
		{
			return MULTIBOOST_NUM_HYPERPARAMETERS;
		}
		else if(experiment.equals("svm"))
		{
			return SVM_NUM_HYPERPARAMETERS;
		}
		else
		{
			System.err.println("Experiment " + experiment + " not supported!");
			return 0;
		}
	}

	public static Instances combineInstances(Instances[] instances)
	{
		Instances ret = new Instances(instances[0].numValues());
		for(int i = 0; i < instances.length; i++)
		{
			ret.addAll(instances[i]);
		}
		return ret;
	}

	public static Instances[] getColumns(Instances[] instances, String intRangeString)
	{
		Instances[] ret = new Instances[instances.length];
		for(int i = 0; i < ret.length; i++)
		{
			if(instances[i].numInstances() > 0)
				ret[i] = getColumns(instances[i], intRangeString);
			else
				ret[i] = new Instances(instances[0].numValues());
		}
		return ret;
	}

	public static Instances getColumns(Instances instances, String intRangeString)
	{
		return getColumns(instances, new IntRange(intRangeString));
	}

	public static Instances getColumns(Instances instances, IntRange intRange)
	{
		if(instances.instance(0) instanceof SparseInstance)
		{
			System.err.println("Getting Columns of Sparse Instances is currently not supported....");
		}
		int numValues = intRange.getUsedIndexes().length;
		Instances ret = new Instances(numValues);

		for(int i = 0; i < instances.numInstances(); i++)
		{
			double target = instances.instance(i).target();
			double[] values = new double[numValues];
			for(int j = 0; j < numValues; j++)
			{
				values[j] = instances.instance(i).getValue(intRange.getUsedIndexes()[j]);
			}
			ret.add(createDenseInstance(target, values));
		}
		return ret;
	}

	public static Instances getHyperparametersOfInstances(Instances instances, String experiment)
	{
		if(!(instances.instance(0) instanceof DenseInstance))
		{
			System.err.println("Only dense instances are supported!");
		}
		Instances ret = null;
		if(experiment.equals("multiboost"))
		{
			ret = new Instances(MULTIBOOST_NUM_HYPERPARAMETERS);
			for(int i = 0; i < instances.numInstances(); i++)
			{
				Instance hyperparameterInstance = InstanceUtils.createDenseInstance(0, new double[] { instances.instance(i).getValues()[0], instances.instance(i).getValues()[1] });
				ret.add(hyperparameterInstance);
			}
		}
		else if(experiment.equals("svm"))
		{
			ret = new Instances(SVM_NUM_HYPERPARAMETERS);
			for(int i = 0; i < instances.numInstances(); i++)
			{
				Instance hyperparameterInstance = InstanceUtils.createDenseInstance(0, new double[] { instances.instance(i).getValues()[0], instances.instance(i).getValues()[1],
						instances.instance(i).getValues()[2], instances.instance(i).getValues()[3], instances.instance(i).getValues()[4], instances.instance(i).getValues()[5] });
				ret.add(hyperparameterInstance);
			}
		}
		else
		{
			System.err.println("Experiment " + experiment + " not supported!");
		}
		return ret;
	}

	public static Instances[] getHyperparametersOfInstances(Instances[] instances, String experiment)
	{
		Instances[] ret = new Instances[instances.length];
		for(int i = 0; i < instances.length; i++)
		{
			ret[i] = getHyperparametersOfInstances(instances[i], experiment);
		}
		return ret;
	}

	public static void computeCrossValidationSplits(Instances[] instancesArray, int numSplits)
	{
		int numberOfDatasets = instancesArray.length;
		if(numSplits > numberOfDatasets)
		{
			System.err.println("Cannot compute more splits than number of Datasets!");
			return;
		}
		if(numSplits <= 1)
		{
			System.err.println("You have to at least compute 2 cross validation splits, will do so now...");
			computeCrossValidationSplits(instancesArray, 2);
			return;
		}
		int splitPoint = (int) Math.round((double) numberOfDatasets / numSplits);
		System.out.println(splitPoint);

		for(int split = 0; split < numSplits - 1; split++)
		{
			for(int dataCount = (split) * splitPoint; dataCount < (split + 1) * splitPoint; dataCount++)
			{
				instancesArray[dataCount].setTestSplitId(split);
				System.out.println("Data in split " + split + "  " + dataCount);
			}
		}
		int split = numSplits - 1;
		for(int dataCount = (split) * splitPoint; dataCount < numberOfDatasets; dataCount++)
		{
			instancesArray[dataCount].setTestSplitId(split);
			System.out.println("Data in split " + split + "  " + dataCount);
		}
	}

	public static Instances[] getTestSplit(Instances[] instancesArray, int splitId)
	{
		int size = 0;
		for(int i = 0; i < instancesArray.length; i++)
		{
			if(instancesArray[i].getTestSplitId() == splitId)
				size++;
		}
		Instances[] ret = new Instances[size];
		int index = 0;
		for(int i = 0; i < instancesArray.length; i++)
		{
			if(instancesArray[i].getTestSplitId() == splitId)
			{
				ret[index] = instancesArray[i];
				index++;
			}
		}
		return ret;
	}

	public static Instances[] getTrainSplit(Instances[] instancesArray, int splitId)
	{
		int size = 0;
		for(int i = 0; i < instancesArray.length; i++)
		{
			if(instancesArray[i].getTestSplitId() != splitId)
				size++;
		}
		Instances[] ret = new Instances[size];
		int index = 0;
		for(int i = 0; i < instancesArray.length; i++)
		{
			if(instancesArray[i].getTestSplitId() != splitId)
			{
				ret[index] = instancesArray[i];
				index++;
			}
		}
		return ret;
	}

	public static int getRank(Instances instances, Instance instance)
	{
		double target = instance.target();
		int rank = 1;
		for(int j = 0; j < instances.numInstances(); j++)
		{
			if(instances.instance(j).target() > target)
				rank++;
		}
		return rank;
	}
}
