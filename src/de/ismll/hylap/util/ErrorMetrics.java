package de.ismll.hylap.util;

import de.ismll.core.Instances;

public class ErrorMetrics
{
	
	
	
	/**
	 * Computes the average rank over all datasets given the currently best found hyper-parameter combination result.
	 * 
	 * @param instances
	 *            The datasets
	 * @param bestAcc
	 *            The best accuracy for dataset i is given in bestAcc[i]
	 * @return The average rank.
	 */
	public static double averageRank(Instances[] instances, double[] bestAcc, double[] weights)
	{
		double averageRank = 0;
		for(int i = 0; i < instances.length; i++)
		{
			int rank = 1;
			int equallyRanked = -1;
			for(int j = 0; j < instances[i].numInstances(); j++)
			{
				if(instances[i].instance(j).target() > bestAcc[i])
					rank++;
				else if(instances[i].instance(j).target() == bestAcc[i])
					equallyRanked++;
			}
			double finalRank = rank;
			for(int k = 0; k < equallyRanked; k++)
				finalRank += rank + k + 1;
			if(equallyRanked < 0)
				equallyRanked = 0;
			// if(equallyRanked < 0)
			// {
			// throw new IllegalArgumentException(bestAcc[i] + " is not a value in dataset " + i + ".");
			// }
			averageRank += (weights == null ? 1 : weights[i]) * finalRank / (equallyRanked + 1);
		}
		return averageRank / instances.length;
	}

	public static double averageRank(Instances[] instances, double[] bestAcc)
	{
		double averageRank = 0;
		for(int i = 0; i < instances.length; i++)
		{
			int rank = 1;
			int equallyRanked = -1;
			for(int j = 0; j < instances[i].numInstances(); j++)
			{
				if(instances[i].instance(j).target() > bestAcc[i])
					rank++;
				else if(instances[i].instance(j).target() == bestAcc[i])
					equallyRanked++;
			}
			double finalRank = rank;
			for(int k = 0; k < equallyRanked; k++)
				finalRank += rank + k + 1;
			if(equallyRanked < 0)
				equallyRanked = 0;
			// if(equallyRanked < 0)
			// {
			// throw new IllegalArgumentException(bestAcc[i] + " is not a value in dataset " + i + ".");
			// }
			averageRank += finalRank / (equallyRanked + 1);
		}
		return averageRank / instances.length;
	}

	public static double computePrecision(int atWhat, int[] predictedRanked, int[] trueRanked)
	{
		double precision = 0;
		for(int i = 0; i < atWhat; i++)
		{
			int currentInstance = trueRanked[i];
			for(int j = 0; j < atWhat; j++)
			{
				int currentPredictedInstance = predictedRanked[j];
				if(currentInstance == currentPredictedInstance)
				{
					precision++;
					break;
				}
			}
		}
		return precision / atWhat;
	}

	public static double computeRecall(int atWhat, int[] predictedRanked, int[] trueRanked, int relevant)
	{
		double recall = 0;
		for(int i = 0; i < atWhat; i++)
		{
			int currentInstance = trueRanked[i];
			for(int j = 0; j < atWhat; j++)
			{
				int currentPredictedInstance = predictedRanked[j];
				if(currentInstance == currentPredictedInstance)
				{
					recall++;
					break;
				}
			}
		}
		return recall / relevant;
	}

	public double computeRecall(int atWhat, int[] predictedRanked, int[] trueRanked)
	{
		double recall = 0;
		for(int i = 0; i < atWhat; i++)
		{
			int currentInstance = trueRanked[i];
			for(int j = 0; j < atWhat; j++)
			{
				int currentPredictedInstance = predictedRanked[j];
				if(currentInstance == currentPredictedInstance)
				{
					recall++;
					break;
				}
			}
		}
		return recall / predictedRanked.length;
	}

	public static double computeAccuracy(double[] predictedLabels, double[] trueLabels)
	{
		if (predictedLabels.length != trueLabels.length) {
			System.err.println("Accuracy cannot be computed as predicted and true Labels do not have the same length! Returning 0 Accuracy...");
			return 0;
		}
		
		double ret = 0;
		for(int i = 0; i < predictedLabels.length; i++)
		{
			if(predictedLabels[i] == trueLabels[i])
			{
				ret++;
			}
		}
		ret = ret / predictedLabels.length;
		return ret;
	}

	public static double computeRMSE(double[] predictedLabels, double[] trueLabels)
	{
		double ret = computeRSS(predictedLabels, trueLabels);
		ret = ret / predictedLabels.length;
		return Math.sqrt(ret);
	}
	
	public static double computeRSS(double[] predictedLabels, double[] trueLabels) {
		double ret = 0;
		for(int i = 0; i < predictedLabels.length; i++)
		{
			double diff = predictedLabels[i] - trueLabels[i];
			ret += diff * diff;
		}
		return ret;
	}
}
