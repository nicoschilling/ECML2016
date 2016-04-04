package de.ismll.hylap.metadataset;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import de.ismll.core.Instances;

/**
 * Computes meta features for all data sets. Output is a list of comma-separated files, first entry is the dataset name, second the list of features in libsvm format.
 */
public class ComputeMetafeatures
{
	private MetaFeature[] metaFeatures = { new ClassCrossEntropy(), new NumClasses(), new ClassProbabilityMax(), new ClassProbabilityMin(), new ClassProbabilityMean(),
			new ClassProbabilityStandardDeviation(), new NumInstances(), new LogNumInstances(), new NumAttributes(), new LogNumAttributes(), new InstancesDividedByAttributes(),
			new LogInstancesDividedByAttributes(), new AttributesDividedByInstances(), new LogAttributesDividedByInstances(), new SkewnessMax(), new SkewnessMin(), new SkewnessMean(),
			new SkewnessStandardDeviation(), new KurtosisMax(), new KurtosisMin(), new KurotisMean(), new KurtosisStandardDeviation() };

	public ComputeMetafeatures(String folder, String[] filesOfInterest) throws IOException
	{
		File[] filesAll = new File(folder).listFiles();
		File[] files = new File[filesOfInterest.length];
		int index = 0;
		for(File f : filesAll)
		{
			for(String s : filesOfInterest)
			{
				if(f.getName().equals(s))
				{
					files[index++] = f;
					break;
				}
			}
		}

		double[] metaFeatureMax = new double[this.metaFeatures.length];
		double[] metaFeatureMin = new double[this.metaFeatures.length];
		for(int i = 0; i < metaFeatureMax.length; i++)
		{
			metaFeatureMax[i] = Double.NEGATIVE_INFINITY;
			metaFeatureMin[i] = Double.POSITIVE_INFINITY;
		}
		double[][] metaFeatureValues = new double[files.length][this.metaFeatures.length];
		for(int f = 0; f < files.length; f++)
		{
			System.out.println(files[f].getName() + "(" + (f + 1) + "/" + files.length + ")");
			Instances data = new Instances(new File(files[f].getPath() + "/Splits/Split-1/train"));
			for(int i = 0; i < this.metaFeatures.length; i++)
			{
				metaFeatureValues[f][i] = this.metaFeatures[i].compute(data);
				metaFeatureMax[i] = Math.max(metaFeatureMax[i], metaFeatureValues[f][i]);
				metaFeatureMin[i] = Math.min(metaFeatureMin[i], metaFeatureValues[f][i]);
			}
		}

		BufferedWriter writer = new BufferedWriter(new FileWriter("data/metafeatures2.txt"));
		for(int f = 0; f < files.length; f++)
		{
			writer.write(files[f].getName());
			boolean first = true;
			for(int i = 0; i < this.metaFeatures.length; i++)
			{
				String sep = " ";
				if(first)
				{
					first = false;
					sep = ",";
				}
				writer.write(sep + (metaFeatureValues[f][i] - metaFeatureMin[i]) / (metaFeatureMax[i] - metaFeatureMin[i]));
			}
			writer.newLine();
		}
		writer.close();
	}

	private interface MetaFeature
	{
		public double compute(Instances data);
	}

	private class ClassCrossEntropy implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			HashMap<Double, Double> classFrequency = new HashMap<Double, Double>();
			for(int i = 0; i < data.numInstances(); i++)
			{
				double target = data.instance(i).target();
				if(classFrequency.containsKey(target))
					classFrequency.put(target, classFrequency.get(target) + 1);
				else
					classFrequency.put(target, 1.0);
			}
			double entropy = 0;
			for(double freq : classFrequency.values())
				entropy -= freq / data.numInstances() * Math.log(freq / data.numInstances());
			return entropy;
		}
	}

	private class NumClasses implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			HashSet<Double> classes = new HashSet<Double>();
			for(int i = 0; i < data.numInstances(); i++)
				classes.add(data.instance(i).target());
			return classes.size();
		}
	}

	private class ClassProbabilityMax implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			HashMap<Double, Double> classFrequency = new HashMap<Double, Double>();
			for(int i = 0; i < data.numInstances(); i++)
			{
				double target = data.instance(i).target();
				if(classFrequency.containsKey(target))
					classFrequency.put(target, classFrequency.get(target) + 1);
				else
					classFrequency.put(target, 1.0);
			}
			double max = 0;
			for(double freq : classFrequency.values())
				max = Math.max(max, freq);
			return max / data.numInstances();
		}
	}

	private class ClassProbabilityMin implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			HashMap<Double, Double> classFrequency = new HashMap<Double, Double>();
			for(int i = 0; i < data.numInstances(); i++)
			{
				double target = data.instance(i).target();
				if(classFrequency.containsKey(target))
					classFrequency.put(target, classFrequency.get(target) + 1);
				else
					classFrequency.put(target, 1.0);
			}
			double min = Double.POSITIVE_INFINITY;
			for(double freq : classFrequency.values())
				min = Math.min(min, freq);
			return min / data.numInstances();
		}
	}

	private class ClassProbabilityMean implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			HashMap<Double, Double> classFrequency = new HashMap<Double, Double>();
			for(int i = 0; i < data.numInstances(); i++)
			{
				double target = data.instance(i).target();
				if(classFrequency.containsKey(target))
					classFrequency.put(target, classFrequency.get(target) + 1);
				else
					classFrequency.put(target, 1.0);
			}
			double sum = 0;
			int count = 0;
			for(double freq : classFrequency.values())
			{
				sum += freq / data.numInstances();
				count++;
			}
			return sum / count;
		}
	}

	private class ClassProbabilityStandardDeviation implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			HashMap<Double, Double> classFrequency = new HashMap<Double, Double>();
			for(int i = 0; i < data.numInstances(); i++)
			{
				double target = data.instance(i).target();
				if(classFrequency.containsKey(target))
					classFrequency.put(target, classFrequency.get(target) + 1);
				else
					classFrequency.put(target, 1.0);
			}
			double sum = 0;
			int count = 0;
			for(double freq : classFrequency.values())
			{
				sum += freq / data.numInstances();
				count++;
			}
			double mean = sum / count;
			sum = 0;
			for(double freq : classFrequency.values())
			{
				sum += Math.pow(freq / data.numInstances() - mean, 2);
			}
			return Math.sqrt(sum / count);
		}
	}

	private class NumInstances implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			return data.numInstances();
		}
	}

	private class LogNumInstances extends NumInstances
	{
		@Override
		public double compute(Instances data)
		{
			return Math.log(super.compute(data));
		}
	}

	private class NumAttributes implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			int numAttributes = 0;
			for(int j = 0; j < data.numValues(); j++)
			{
				for(int i = 0; i < data.numInstances(); i++)
				{
					if(data.instance(i).getValue(j) != 0)
					{
						numAttributes++;
						break;
					}
				}
			}
			return numAttributes;
		}
	}

	private class LogNumAttributes extends NumAttributes
	{
		@Override
		public double compute(Instances data)
		{
			return Math.log(super.compute(data));
		}
	}

	private class LogAttributesDividedByInstances extends AttributesDividedByInstances
	{
		@Override
		public double compute(Instances data)
		{
			return Math.log(super.compute(data));
		}
	}

	private class LogInstancesDividedByAttributes extends InstancesDividedByAttributes
	{
		@Override
		public double compute(Instances data)
		{
			return Math.log(super.compute(data));
		}
	}

	private class AttributesDividedByInstances implements MetaFeature
	{
		private NumAttributes a = new NumAttributes();
		private NumInstances i = new NumInstances();

		@Override
		public double compute(Instances data)
		{
			return this.a.compute(data) / this.i.compute(data);
		}
	}

	private class InstancesDividedByAttributes implements MetaFeature
	{
		private NumAttributes a = new NumAttributes();
		private NumInstances i = new NumInstances();

		@Override
		public double compute(Instances data)
		{
			return this.i.compute(data) / this.a.compute(data);
		}
	}

	private class SkewnessMax implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			double maxSkewness = Double.NEGATIVE_INFINITY;
			for(int i = 0; i < data.numValues(); i++)
			{
				boolean activeFeature = false;
				double[] values = new double[data.numInstances()];
				double mean = 0;
				for(int j = 0; j < data.numInstances(); j++)
				{
					values[j] = data.instance(j).getValue(i);
					mean += values[j];
					if(!activeFeature && values[j] != 0)
						activeFeature = true;
				}
				if(!activeFeature)
					continue;
				mean /= data.numInstances();
				double m2 = 0;
				double m3 = 0;
				for(double v : values)
				{
					m2 += Math.pow(v - mean, 2);
					m3 += Math.pow(v - mean, 3);
				}
				m2 /= data.numInstances();
				if(m2 == 0)
					continue;
				m3 /= data.numInstances();
				maxSkewness = Math.max(maxSkewness, m3 / Math.pow(m2, 1.5));
			}
			return maxSkewness * Math.sqrt((long) data.numInstances() * (data.numInstances() - 1)) / (data.numInstances() - 2);
		}
	}

	private class SkewnessMin implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			double minSkewness = Double.POSITIVE_INFINITY;
			for(int i = 0; i < data.numValues(); i++)
			{
				boolean activeFeature = false;
				double[] values = new double[data.numInstances()];
				double mean = 0;
				for(int j = 0; j < data.numInstances(); j++)
				{
					values[j] = data.instance(j).getValue(i);
					mean += values[j];
					if(!activeFeature && values[j] != 0)
						activeFeature = true;
				}
				if(!activeFeature)
					continue;
				mean /= data.numInstances();
				double m2 = 0;
				double m3 = 0;
				for(double v : values)
				{
					m2 += Math.pow(v - mean, 2);
					m3 += Math.pow(v - mean, 3);
				}
				m2 /= data.numInstances();
				if(m2 == 0)
					continue;
				m3 /= data.numInstances();
				minSkewness = Math.min(minSkewness, m3 / Math.pow(m2, 1.5));
			}

			return minSkewness * Math.sqrt((long) data.numInstances() * (data.numInstances() - 1)) / (data.numInstances() - 2);
		}
	}

	private class SkewnessMean implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			double meanSkewness = 0;
			int count = 0;
			for(int i = 0; i < data.numValues(); i++)
			{
				boolean activeFeature = false;
				double[] values = new double[data.numInstances()];
				double mean = 0;
				for(int j = 0; j < data.numInstances(); j++)
				{
					values[j] = data.instance(j).getValue(i);
					mean += values[j];
					if(!activeFeature && values[j] != 0)
						activeFeature = true;
				}
				if(!activeFeature)
					continue;
				mean /= data.numInstances();
				double m2 = 0;
				double m3 = 0;
				for(double v : values)
				{
					m2 += Math.pow(v - mean, 2);
					m3 += Math.pow(v - mean, 3);
				}
				m2 /= data.numInstances();
				if(m2 == 0)
					continue;
				m3 /= data.numInstances();
				meanSkewness += m3 / Math.pow(m2, 1.5);
				count++;
			}

			return meanSkewness / count * Math.sqrt((long) data.numInstances() * (data.numInstances() - 1)) / (data.numInstances() - 2);
		}
	}

	private class SkewnessStandardDeviation implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			double[] skewness = new double[data.numValues()];
			ArrayList<Integer> activeIndices = new ArrayList<Integer>();
			for(int i = 0; i < data.numValues(); i++)
			{
				boolean activeFeature = false;
				double[] values = new double[data.numInstances()];
				double mean = 0;
				for(int j = 0; j < data.numInstances(); j++)
				{
					values[j] = data.instance(j).getValue(i);
					mean += values[j];
					if(!activeFeature && values[j] != 0)
						activeFeature = true;
				}
				if(!activeFeature)
					continue;
				mean /= data.numInstances();
				double m2 = 0;
				double m3 = 0;
				for(double v : values)
				{
					m2 += Math.pow(v - mean, 2);
					m3 += Math.pow(v - mean, 3);
				}
				m2 /= data.numInstances();
				if(m2 == 0)
					continue;
				m3 /= data.numInstances();
				skewness[i] = m3 / Math.pow(m2, 1.5) * Math.sqrt((long) data.numInstances() * (data.numInstances() - 1)) / (data.numInstances() - 2);
				activeIndices.add(i);
			}

			double meanSkewness = 0;
			for(double s : skewness)
				meanSkewness += s;
			meanSkewness /= activeIndices.size();
			double sdSkewness = 0;
			for(int i : activeIndices)
				sdSkewness += Math.pow(skewness[i] - meanSkewness, 2);

			return Math.sqrt(sdSkewness / activeIndices.size());
		}
	}

	private class KurtosisMax implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			double n = data.numInstances();
			double maxKurtosis = Double.NEGATIVE_INFINITY;
			for(int i = 0; i < data.numValues(); i++)
			{
				boolean activeFeature = false;
				double[] values = new double[data.numInstances()];
				double mean = 0;
				for(int j = 0; j < n; j++)
				{
					values[j] = data.instance(j).getValue(i);
					mean += values[j];
					if(!activeFeature && values[j] != 0)
						activeFeature = true;
				}
				if(!activeFeature)
					continue;
				mean /= data.numInstances();
				double m2 = 0;
				double m4 = 0;
				for(double v : values)
				{
					m2 += Math.pow(v - mean, 2);
					m4 += Math.pow(v - mean, 4);
				}
				m2 /= n;
				if(m2 == 0)
					continue;
				m4 /= n;
				maxKurtosis = Math.max(maxKurtosis, m4 / Math.pow(m2, 2) - 3);
			}

			return (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * maxKurtosis + 6);
		}
	}

	private class KurtosisMin implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			double n = data.numInstances();
			double minKurtosis = Double.POSITIVE_INFINITY;
			for(int i = 0; i < data.numValues(); i++)
			{
				boolean activeFeature = false;
				double[] values = new double[data.numInstances()];
				double mean = 0;
				for(int j = 0; j < n; j++)
				{
					values[j] = data.instance(j).getValue(i);
					mean += values[j];
					if(!activeFeature && values[j] != 0)
						activeFeature = true;
				}
				if(!activeFeature)
					continue;
				mean /= data.numInstances();
				double m2 = 0;
				double m4 = 0;
				for(double v : values)
				{
					m2 += Math.pow(v - mean, 2);
					m4 += Math.pow(v - mean, 4);
				}
				m2 /= n;
				if(m2 == 0)
					continue;
				m4 /= n;
				minKurtosis = Math.min(minKurtosis, m4 / Math.pow(m2, 2) - 3);
			}

			return (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * minKurtosis + 6);
		}
	}

	private class KurotisMean implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			double n = data.numInstances();
			int count = 0;
			double meanKurtosis = 0;
			for(int i = 0; i < data.numValues(); i++)
			{
				boolean activeFeature = false;
				double[] values = new double[data.numInstances()];
				double mean = 0;
				for(int j = 0; j < n; j++)
				{
					values[j] = data.instance(j).getValue(i);
					mean += values[j];
					if(!activeFeature && values[j] != 0)
						activeFeature = true;
				}
				if(!activeFeature)
					continue;
				mean /= n;
				double m2 = 0;
				double m4 = 0;
				for(double v : values)
				{
					m2 += Math.pow(v - mean, 2);
					m4 += Math.pow(v - mean, 4);
				}
				m2 /= n;
				if(m2 == 0)
					continue;
				m4 /= n;
				meanKurtosis += (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * (m4 / Math.pow(m2, 2) - 3) + 6);
				count++;
			}

			return meanKurtosis / count;
		}
	}

	private class KurtosisStandardDeviation implements MetaFeature
	{
		@Override
		public double compute(Instances data)
		{
			double n = data.numInstances();
			double[] kurtosis = new double[data.numValues()];
			double meanKurtosis = 0;
			ArrayList<Integer> activeIndices = new ArrayList<Integer>();
			for(int i = 0; i < data.numValues(); i++)
			{
				boolean activeFeature = false;
				double[] values = new double[data.numInstances()];
				double mean = 0;
				for(int j = 0; j < n; j++)
				{
					values[j] = data.instance(j).getValue(i);
					mean += values[j];
					if(!activeFeature && values[j] != 0)
						activeFeature = true;
				}
				if(!activeFeature)
					continue;
				mean /= n;
				double m2 = 0;
				double m4 = 0;
				for(double v : values)
				{
					m2 += Math.pow(v - mean, 2);
					m4 += Math.pow(v - mean, 4);
				}
				m2 /= n;
				if(m2 == 0)
					continue;
				m4 /= n;
				kurtosis[i] = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * (m4 / Math.pow(m2, 2) - 3) + 6);
				meanKurtosis += kurtosis[i];
				activeIndices.add(i);
			}
			meanKurtosis /= activeIndices.size();
			double sdKurtosis = 0;
			for(int i : activeIndices)
				sdKurtosis += Math.pow(kurtosis[i] - meanKurtosis, 2);
			return Math.sqrt(sdKurtosis / activeIndices.size());
		}
	}

	public static void main(String[] args)
	{
		try
		{
			File[] files = new File("data/svm/final_med").listFiles();
			String[] filesOfInterest = new String[files.length];
			for(int i = 0; i < files.length; i++)
			{
				filesOfInterest[i] = files[i].getName();
			}
			new ComputeMetafeatures("Z:/data/HyperparameterLearning/raw/Classification", filesOfInterest);// /home/nico/Documents/Datasets/Classification/
		}
		catch(IOException e)
		{
			e.printStackTrace();
		}
	}
}
