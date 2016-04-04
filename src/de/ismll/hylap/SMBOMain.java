package de.ismll.hylap;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;

import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

import de.ismll.core.Instances;
import de.ismll.core.Logger;
import de.ismll.core.Random;
import de.ismll.hylap.acquisitionFunction.AcquisitionFunction;
import de.ismll.hylap.acquisitionFunction.ExpectedImprovement;
import de.ismll.hylap.surrogateModel.ProductOfGPExperts;
import de.ismll.hylap.surrogateModel.SurrogateModel;

public class SMBOMain
{

	// WEKA: hpRange = 102 hpIndicator = 64
	// SVM: hpRange = 6 hpIndicator = 3
	// Multiboost: hpRange = 2 hpIndicator = 0

	public static void help()
	{
		System.out.println("-f\tPath to the folder where your datasets are stored.\n" + "-dataset\tName of the dataset to evaluate.\n"
				+ "-tries\tNumber of steps for the SMBO algorithm.\n" + "-iter\tNumber of iterations, results are averaged.\n"
				+ "-output\tThe location where the results shall be saved.\n" + "-seed\tRandom seed (Default: 0, Random: r)\n"
				+ "-s\tThe surrogate model, \"pogpe\" or \"sgpe\". \n");
		System.exit(0);
	}

	public static void main(String[] args) throws IOException
	{
		HashMap<String, String> argsMap = new HashMap<String, String>();
		for(int i = 0; i < args.length; i++)
			argsMap.put(args[i], args[++i]);

		// Mandatory parameters
		if(!argsMap.containsKey("-dataset") || !argsMap.containsKey("-f") || !argsMap.containsKey("-tries") || !argsMap.containsKey("-iter")
				)
			help();
		String datasetName = argsMap.get("-dataset");
		String dataFolder = argsMap.get("-f");
		File[] files = new File(dataFolder).listFiles();
		int maxTries = Integer.parseInt(argsMap.get("-tries"));
		int numIters = Integer.parseInt(argsMap.get("-iter"));

		if(!new File(dataFolder + "/" + datasetName).exists())
		{
			Logger.severe("Data set " + datasetName + " does not exist in folder " + new File(dataFolder).getAbsolutePath() + ".");
			System.exit(1);
		}
		// Set seed
		if(argsMap.containsKey("-seed"))
		{
			if(argsMap.get("-seed").equals("r"))
				Random.setSeed(System.currentTimeMillis());
			else
				Random.setSeed(Long.parseLong(argsMap.get("-seed")));
		}
		else
			Random.setSeed(0);

		if(argsMap.containsKey("-hpRange"))
			HyperparameterCombination.HYPERPARAMETER_INDEX_RANGE_MAX = Integer.parseInt(argsMap.get("-hpRange"));
		if(argsMap.containsKey("-hpIndicatorRange"))
			HyperparameterCombination.HYPERPARAMETER_INDICATOR_RANGE_MAX = Integer.parseInt(argsMap.get("-hpIndicatorRange"));

		File outputFile = argsMap.get("-output") == null ? null : new File(argsMap.get("-output"));
		
		int algorithmOffset = 0;

		Logger.info("Loading data sets from " + new File(dataFolder).getAbsolutePath() + ".");
		Instances[] train = new Instances[files.length - 1];
		int testId = -1;

		for(int i = 0; i < files.length; i++)
		{
			if(files[i].getName().equals(datasetName))
				testId = i;
		}

		int l = 0;
		for(int j = 0; j < files.length; j++)
		{
			if(j != testId)
			{
				train[l++] = new Instances(files[j]);
			}
		}

		Instances testData = new Instances(files[testId]);

		Logger.info("Starting the SMBO framework.");
		double[][] acc = new double[maxTries][numIters];
		double[][] rank = new double[maxTries][numIters];
		double[] time = new double[maxTries];
		int[] count = new int[maxTries];
		int[][] algorithmSelection = new int[numIters][];
		for(int iter = 0; iter < numIters; iter++)
		{
			Logger.info("Starting iteration " + (iter + 1) + ".");
			AcquisitionFunction a = new ExpectedImprovement();

			SurrogateModel s = null;
			if(!argsMap.containsKey("-s"))
				s = null;
			else if(argsMap.get("-s").equals("pogpe"))
			{
				s = new ProductOfGPExperts(train, ProductOfGPExperts.ALL_EXPERTS, true, false, false);
			}
			else if(argsMap.get("-s").equals("sgpe"))
			{
				s = new ProductOfGPExperts(train, ProductOfGPExperts.SINGLE_EXPERT, true, false, false);
			}
			else
			{
				Logger.severe("Unknown surrogate function \"" + argsMap.get("-s") + "\"");
				System.exit(1);
			}

			long start = System.nanoTime();
			SMBO smbo = new SMBO(testData, a, s, algorithmOffset);
			for(int j = 0; j < maxTries; j++)
			{
				if(j > 0 && rank[j - 1][iter] == 1)
				{
					acc[j][iter] = acc[j - 1][iter];
					rank[j][iter] = rank[j - 1][iter];
				}
				else
				{
					smbo.iterate();
					acc[j][iter] = smbo.getBestAccuracy();
					rank[j][iter] = smbo.getBestRank();
					time[j] += (double) (System.nanoTime() - start) / 1000000;
					count[j]++;
					// }
				}
				algorithmSelection[iter] = smbo.getAlgorithmSelection();
			}

			StandardDeviation sd = new StandardDeviation();
			Mean mean = new Mean();

			if(outputFile == null)
			{
				Logger.info("Printing results to console.");
				System.out.println("Accuracy(mean),Accuracy(sd),Rank(mean),Rank(sd),Time in ms");
				for(int j = 0; j < maxTries; j++)
				{
					System.out.println(mean.evaluate(acc[j]) + "," + sd.evaluate(acc[j]) + "," + mean.evaluate(rank[j]) + "," + sd.evaluate(rank[j])
							+ "," + (time[j] / count[j]));
				}
			}
			else
			{
				Logger.info("Writing results to " + outputFile.getAbsolutePath() + ".");
				BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "utf-8"));
				bw.write("Accuracy(mean),Accuracy(sd),Rank(mean),Rank(sd),Time in ms");
				bw.newLine();
				for(int j = 0; j < maxTries; j++)
				{
					bw.write(mean.evaluate(acc[j]) + "," + sd.evaluate(acc[j]) + "," + mean.evaluate(rank[j]) + "," + sd.evaluate(rank[j]) + ","
							+ (time[j] / count[j]));
					bw.newLine();
				}
				bw.close();
			}
		}
	}
}
