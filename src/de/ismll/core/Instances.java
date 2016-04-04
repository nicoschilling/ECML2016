package de.ismll.core;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

public class Instances implements Iterable<Instance>
{

	private SplitMap[] splitMap;

	protected ArrayList<Instance> instances = new ArrayList<Instance>();

	private Instances[] splits;

	private int numValues;

	private int testSplitId;

	private boolean kFoldSplitted = false;
	private boolean percentageFoldSplitted = false;

	public Instances(int numValues)
	{
		this.numValues = numValues;
	}
	
	

	public Instances(String filename) throws IOException
	{
		this(new File(filename), " ");
	}
	
	public Instances(String filename, String delimiter) throws IOException
	{
		this(new File(filename), delimiter);
	}
	
	public Instances(File file) throws IOException {
		this(file," ");
	}

	public Instances(File file, String delimiter) throws IOException
	{
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		boolean firstLine = true, sparse = false;
		int maxKey = 0;
		while((line = br.readLine()) != null)
		{
			if(firstLine)
			{
				firstLine = false;
				sparse = line.contains(":");
			}

			String[] split = line.split(delimiter);
			if(sparse)
			{
				int[] keys = new int[split.length - 1];
				double[] values = new double[split.length - 1];

				for(int i = 1; i < split.length; i++)
				{
					String[] split2 = split[i].split(":");
					keys[i - 1] = Integer.parseInt(split2[0]);
					maxKey = Math.max(maxKey, keys[i - 1] + 1);
					values[i - 1] = Double.parseDouble(split2[1]);
				}
				this.instances.add(new SparseInstance(Double.parseDouble(split[0]), keys, values));
			}
			else
			{
				double[] values = new double[split.length - 1];
				for(int i = 1; i < split.length; i++)
					values[i - 1] = Double.parseDouble(split[i]);
				maxKey = Math.max(maxKey, split.length - 1);
				this.instances.add(new DenseInstance(Double.parseDouble(split[0]), values));
			}
			this.numValues = maxKey;
		}
		br.close();
	}

	protected int numOfLines(File file) throws IOException
	{
		BufferedReader reader = new BufferedReader(new FileReader(file));
		int lines = 0;
		while(reader.readLine() != null)
			lines++;
		reader.close();
		return lines;
	}

	/**
	 * Shuffles the internal Array List of instances according to a fixed random seed.
	 */
	public void shuffle(int seed)
	{
		java.util.Random random = new java.util.Random(seed);
		Collections.shuffle(instances, random);
	}

	public double getMaxTarget()
	{
		double[] targets = this.getTargets();
		double max = -Double.MAX_VALUE;
		for(int i = 0; i < targets.length; i++)
		{
			if(targets[i] > max)
			{
				max = targets[i];
			}
		}
		return max;
	}

	public double getMinTarget()
	{
		double[] targets = this.getTargets();
		double min = Double.MAX_VALUE;
		for(int i = 0; i < targets.length; i++)
		{
			if(targets[i] < min)
			{
				min = targets[i];
			}
		}
		return min;
	}

	/**
	 * Gets the targets of all instances.
	 * 
	 * @return double array of target values.
	 */
	public double[] getTargets()
	{
		double[] ret = new double[this.numInstances()];
		for(int i = 0; i < this.numInstances(); i++)
		{
			ret[i] = this.instance(i).target();
		}
		return ret;
	}

	/**
	 * Overwrites target values with new target values, i.e. predicted values
	 * 
	 * @param newTargets
	 */
	public void setTargets(double[] newTargets)
	{
		if(newTargets.length != this.numInstances())
		{
			System.err.println("Targets cannot be overwritten due to different length of newTargets and number of Instances!");
			return;
		}
		for(int i = 0; i < newTargets.length; i++)
		{
			this.instance(i).setTarget(newTargets[i]);
		}
	}

	/**
	 * Returns the test split for a given splitID.
	 * 
	 * @param splitId
	 * @return
	 */
	public Instances getTestSplit(int splitId)
	{
		if(iskFoldSplitted())
		{
			return splits[splitId];
		}
		else if(isPercentageFoldSplitted())
		{
			Instances testInstances = new Instances(this.numValues);
			for(int i = 0; i < this.splitMap[splitId].testSamples.size(); i++)
			{
				int index = this.splitMap[splitId].testSamples.get(i);
				testInstances.add(this.instance(index));
			}
			return testInstances;
		}
		else
		{
			System.err.println("Splits have not been computed and are null!");
			return null;
		}

	}

	/**
	 * Returns a validation split for a given splitId, where the validation split has id (splitId+1) or 0 if splitId is last fold
	 * 
	 * @param splitId
	 * @return
	 */
	public Instances getValidationSplit(int splitId)
	{
		if(iskFoldSplitted())
		{
			int validationId = splitId + 1;
			if(splitId == this.splits.length - 1)
			{
				validationId = 0;
			}
			return this.splits[validationId];
		}
		else if(isPercentageFoldSplitted())
		{
			Instances validationInstances = new Instances(this.numValues);
			for(int i = 0; i < this.splitMap[splitId].validationSamples.size(); i++)
			{
				int index = this.splitMap[splitId].validationSamples.get(i);
				validationInstances.add(this.instance(index));
			}
			return validationInstances;
		}
		else
		{
			System.err.println("Splits have not been computed and are null!");
			return null;
		}

	}

	/**
	 * Returns trainSplit for a given splitId, if useValidation is true, one fold (validation fold) is left out.
	 * 
	 * @param splitId
	 * @param useValidation
	 * @return
	 */
	public Instances getTrainSplit(int splitId, boolean useValidation)
	{
		if(!iskFoldSplitted() && !isPercentageFoldSplitted())
		{
			System.err.println("Splits have not been computed and are null, please either compute k-Fold CV Splits or k Percentage Splits!");
			return null;
		}
		Instances trainInstances = new Instances(this.numValues);
		if(iskFoldSplitted())
		{
			if(useValidation)
			{
				int validationId = splitId + 1;
				if(splitId == this.splits.length - 1)
				{
					validationId = 0;
				}
				// Return the k-Fold CV train Split
				for(int i = 0; i < this.splits.length; i++)
				{
					if(i != splitId && i != validationId)
					{
						for(int instance = 0; instance < this.splits[i].instances.size(); instance++)
						{
							trainInstances.add(this.splits[i].instances.get(instance));
						}
					}
				}
			}
			else
			{
				{
					// Return the k-Fold CV train Split
					for(int i = 0; i < this.splits.length; i++)
					{
						if(i != splitId)
						{
							for(int instance = 0; instance < this.splits[i].instances.size(); instance++)
							{
								trainInstances.add(this.splits[i].instances.get(instance));
							}
						}
					}
				}
			}
		}
		else if(isPercentageFoldSplitted())
		{
			if(useValidation)
			{
				for(int i = 0; i < this.splitMap[splitId].trainSamples.size(); i++)
				{
					int index = this.splitMap[splitId].trainSamples.get(i);
					trainInstances.add(this.instance(index));
				}
			}
			else
			{
				for(int i = 0; i < this.splitMap[splitId].trainSamples.size(); i++)
				{
					int index = this.splitMap[splitId].trainSamples.get(i);
					trainInstances.add(this.instance(index));
				}
				for(int i = 0; i < this.splitMap[splitId].validationSamples.size(); i++)
				{
					int index = this.splitMap[splitId].validationSamples.get(i);
					trainInstances.add(this.instance(index));
				}
			}
		}
		return trainInstances;
	}

	/**
	 * Returns the train split for a given splitID.
	 * 
	 * @param splitId
	 * @return
	 */
	public Instances getTrainSplit(int splitId)
	{
		return getTrainSplit(splitId, false);
	}

	/**
	 * Computes numSplits many Splits of the given Instances Object and uses percentageOfTrain many instances for training, percentageOfValidation many for validation and the rest for train splits.
	 * 
	 * @param numSplits
	 * @param percentageOfTrain
	 * @param percentageOfValidation
	 */
	public void computeSplits(int numSplits, double percentageOfTrain, double percentageOfValidation, int seed)
	{
		if(percentageOfTrain + percentageOfValidation >= 1)
		{
			System.err.println("Sum of percentages may not exceed 1!");
			return;
		}
		if(percentageOfTrain <= 0 || percentageOfValidation <= 0)
		{
			System.err.println("Percentages must be strictly positive!");
		}

		this.splitMap = new SplitMap[numSplits];
		for(int i = 0; i < numSplits; i++)
		{
			splitMap[i] = new SplitMap();
		}

		int numberOfTrainExamples = (int) Math.round(this.instances.size() * percentageOfTrain);
		int numberOfValidationExamples = (int) Math.round(this.instances.size() * percentageOfValidation);
		
		java.util.Random rand = new java.util.Random(seed);

		Collections.shuffle(instances, rand);

		for(int split = 0; split < numSplits; split++)
		{
			ArrayList<Integer> notYetAssigned = new ArrayList<>();
			for(int i = 0; i < this.instances.size(); i++)
			{
				notYetAssigned.add(i);
			}
			for(int j = 0; j < numberOfTrainExamples; j++)
			{
				int randomIndex = Random.nextInt(notYetAssigned.size());
				int randomInstance = notYetAssigned.get(randomIndex);
				splitMap[split].trainSamples.add(randomInstance);
				notYetAssigned.remove(randomIndex);
			}
			for(int j = 0; j < numberOfValidationExamples; j++)
			{
				int randomIndex = Random.nextInt(notYetAssigned.size());
				int randomInstance = notYetAssigned.get(randomIndex);
				splitMap[split].validationSamples.add(randomInstance);
				notYetAssigned.remove(randomIndex);
			}
			int numberOfInstancesRemaining = notYetAssigned.size();
			for(int j = 0; j < numberOfInstancesRemaining; j++)
			{
				splitMap[split].testSamples.add(notYetAssigned.get(j));
			}
		}
		this.setkFoldSplitted(false);
		this.setPercentageFoldSplitted(true);
	}

	/**
	 * Computes k-fold cross validation splits where numSplits = k
	 * 
	 * @param numSplits
	 */
	public void computeSplits(int numSplits)
	{
		int numInstances = this.instances.size();
		this.splits = new Instances[numSplits];

		for(int i = 0; i < splits.length; i++)
		{
			this.splits[i] = new Instances(this.numValues);
		}

		int splitPoint = (int) Math.round((double) numInstances / numSplits);

		Collections.shuffle(instances, Random.getInstance());

		for(int split = 0; split < numSplits - 1; split++)
		{
			for(int instanceCount = (split) * splitPoint; instanceCount < (split + 1) * splitPoint; instanceCount++)
			{
				this.splits[split].add(instances.get(instanceCount));
			}
		}
		int split = numSplits - 1;
		for(int instanceCount = (split) * splitPoint; instanceCount < numInstances; instanceCount++)
		{
			this.splits[split].add(instances.get(instanceCount));
		}

		this.setkFoldSplitted(true);
		this.setPercentageFoldSplitted(false);
	}

	/**
	 * Adds a given instance to the Instances object.
	 * 
	 * @param instance
	 * @return true if instance was successfully added.
	 */
	public boolean add(Instance instance)
	{
		int maxKey = instance.getKeys()[instance.getKeys().length - 1];
		if(maxKey > this.numValues)
			throw new IllegalArgumentException("The instance has " + maxKey + " attributes but only " + this.numValues + " are allowed.");
		return this.instances.add(instance);
	}

	/**
	 * Adds all of the given Instances to the Instances object.
	 * 
	 * @param instances
	 * @return
	 */
	public boolean addAll(Instances instances)
	{
		int maxKey = 0;
		for(int i = 0; i < instances.numInstances(); i++)
			maxKey = Math.max(maxKey, instances.instance(i).getKeys()[instances.instance(i).getKeys().length - 1]);
		if(maxKey > this.numValues)
			throw new IllegalArgumentException("The instance has " + maxKey + " attributes but only " + this.numValues + " are allowed.");
		return this.instances.addAll(instances.instances);
	}

	public boolean remove(Instance instance)
	{
		return this.instances.remove(instance);
	}

	public Instance instance(int i)
	{
		return this.instances.get(i);
	}

	public int numInstances()
	{
		return this.instances.size();
	}

	public int numValues()
	{
		return this.numValues;
	}

	public void saveToLibsvm(File file) throws IOException
	{
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "utf-8"));
		for(Instance instance : this.instances)
		{
			bw.write("" + instance.target());
			int[] keys = instance.getKeys();
			double[] values = instance.getValues();
			for(int i = 0; i < keys.length; i++)
			{
				if(values[i] != 0)
				bw.write(" " + keys[i] + ":" + values[i]);
			}
			bw.newLine();
		}
		bw.close();
	}

	public void saveToSVMLight(File file) throws IOException
	{
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "utf-8"));
		for(Instance instance : this.instances)
		{
			bw.write("" + instance.target());
			int[] keys = instance.getKeys();
			double[] values = instance.getValues();
			for(int i = 0; i < keys.length; i++)
			{
				if(values[i] != 0)
				bw.write(" " + (keys[i] + 1) + ":" + values[i]);
			}
			bw.newLine();
		}
		bw.close();
	}

	public void saveToDense(String filename) throws IOException
	{
		saveToDense(new File(filename));
	}
	
	public void saveToDense(String filename, String delimiter) throws IOException
	{
		saveToDense(new File(filename), delimiter);
	}

	public void saveToDense(File file) throws IOException
	{
		saveToDense(file, " ");
	}

	public void saveToDense(File file, String delimiter) throws IOException
	{
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "utf-8"));
		for(Instance instance : this.instances)
		{
			bw.write("" + instance.target());
			int[] keys = instance.getKeys();
			double[] values = instance.getValues();
			for(int i = 0; i < keys.length; i++)
			{
				bw.write(delimiter + values[i]);
			}
			bw.newLine();
		}
		bw.close();
	}

	public boolean iskFoldSplitted()
	{
		return this.kFoldSplitted;
	}

	public void setkFoldSplitted(boolean kFoldSplitted)
	{
		this.kFoldSplitted = kFoldSplitted;
	}

	public boolean isPercentageFoldSplitted()
	{
		return this.percentageFoldSplitted;
	}

	public void setPercentageFoldSplitted(boolean percentageFoldSplitted)
	{
		this.percentageFoldSplitted = percentageFoldSplitted;
	}

	public int getTestSplitId()
	{
		return this.testSplitId;
	}

	public void setTestSplitId(int testSplitId)
	{
		this.testSplitId = testSplitId;
	}



	@Override
	public Iterator<Instance> iterator()
	{
		return this.instances.iterator();
	}

}
