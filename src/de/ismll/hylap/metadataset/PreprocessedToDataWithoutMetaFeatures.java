package de.ismll.hylap.metadataset;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import de.ismll.core.InstanceUtils;
import de.ismll.core.Instances;

/**
 * Uses some preprocessed data to convert it into the final libsvm format without meta-features.
 */
public class PreprocessedToDataWithoutMetaFeatures
{
	public static void main(String[] args)
	{
		/**
		 * The hyperparameter indices in the final training set. Hyperparameter indices ranges between 0 and 99.
		 */
		int[] indicesOfHyperparameters = { 0, 1, 2, 3, 4, 5 };
		/**
		 * This is the position where the respective hyperparameter can be found in the folder <i>dataFolder</i>
		 */
		int[] indicesInSource = { 1, 2, 3, 4, 5, 6 };

		List<Integer> indicesToNormalizedInSource = Arrays.asList(new Integer[] { 4, 5, 6 });
		/**
		 * Location of the unprocessed files.
		 */
		File sourceFolder = new File("data/svm/all");

		File destFolder = new File("data/svm/converted");
		destFolder.mkdirs();

		if(indicesInSource.length != indicesOfHyperparameters.length)
		{
			System.err.println("Number of source and destination indices must be the same.");
			System.exit(1);
		}

		File[] files = sourceFolder.listFiles();
		Instances[] rawInstances = new Instances[files.length];
		Instances[] finalInstances = new Instances[files.length];
		for(int i = 0; i < files.length; i++)
			try
			{
				System.out.println("Converting dataset: " + i + " " + files[i].getName());
				rawInstances[i] = new Instances(files[i]);
				finalInstances[i] = new Instances(max(indicesOfHyperparameters));
			}
			catch(IOException e)
			{
				e.printStackTrace();
			}

		double[] maxValues = new double[indicesInSource.length];
		double[] minValues = new double[indicesInSource.length];

		for(int i = 0; i < indicesInSource.length; i++)
		{
			for(int j = 0; j < rawInstances.length; j++)
			{
				for(int k = 0; k < rawInstances[j].numInstances(); k++)
				{
					maxValues[i] = Math.max(maxValues[i], Math.abs(log2(rawInstances[j].instance(k).getValue(indicesInSource[i]))));
					minValues[i] = Math.min(minValues[i], Math.abs(log2(rawInstances[j].instance(k).getValue(indicesInSource[i]))));
				}
			}
		}

		for(int j = 0; j < rawInstances.length; j++)
		{
			for(int k = 0; k < rawInstances[j].numInstances(); k++)
			{
				int[] keys = Arrays.copyOf(indicesOfHyperparameters, indicesOfHyperparameters.length);
				double[] values = new double[indicesOfHyperparameters.length];
				for(int i = 0; i < indicesInSource.length; i++)
				{
					if(indicesToNormalizedInSource.contains(indicesInSource[i]))
					{
						values[i] = (log2(rawInstances[j].instance(k).getValue(indicesInSource[i])) - minValues[i]) / (maxValues[i] - minValues[i]);
					}
					else
						values[i] = rawInstances[j].instance(k).getValue(indicesInSource[i]);
				}
				finalInstances[j].add(InstanceUtils.createSparseInstance(rawInstances[j].instance(k).target(), values, keys));
			}
		}

		for(int i = 0; i < files.length; i++)
		{
			try
			{
				finalInstances[i].saveToDense(new File(destFolder.getPath() + "/" + files[i].getName()));
			}
			catch(IOException e)
			{
				e.printStackTrace();
			}
		}
	}

	private static double log2(double v)
	{
		if(v == 0)
			return 0;
		else
			return Math.log(v) / Math.log(2);
	}

	private static int max(int[] array)
	{
		int maxValue = array[0];
		for(int i = 1; i < array.length; i++)
			if(array[i] > maxValue)
				maxValue = array[i];
		return maxValue;
	}
}
