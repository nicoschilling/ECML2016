package de.ismll.hylap;

import java.util.ArrayList;
import java.util.logging.Level;

import de.ismll.core.Instance;
import de.ismll.core.InstanceUtils;
import de.ismll.core.Instances;
import de.ismll.core.Logger;
import de.ismll.hylap.acquisitionFunction.AcquisitionFunction;
import de.ismll.hylap.surrogateModel.GaussianProcess;
import de.ismll.hylap.surrogateModel.OnlineLearnable;
import de.ismll.hylap.surrogateModel.SurrogateModel;

public class SMBO
{
	private Instances instances;

	private Instances h;

	private AcquisitionFunction acquisitionFunction;

	private SurrogateModel surrogateModel;

	private ArrayList<Instance> candidates;

	private Instance bestInstance;

	private double time;


	/**
	 * The index at which the algorithm indices start.
	 */
	private int algorithmOffset;

	/**
	 * Stores information about the number each algorithm was chosen.
	 */
	private int[] algorithmSelection;

	public SMBO(Instances instances, AcquisitionFunction acquisitionFunction, SurrogateModel surrogateModel, int algorithmOffset)
	{
		this.instances = instances;
		this.acquisitionFunction = acquisitionFunction;
		this.surrogateModel = surrogateModel;
		this.h = new Instances(instances.numValues());
		this.candidates = new ArrayList<Instance>();
		if(HyperparameterCombination.HYPERPARAMETER_ALGORITHM_RANGE > 1)
			this.algorithmSelection = new int[HyperparameterCombination.HYPERPARAMETER_ALGORITHM_RANGE];
		for(int i = 0; i < this.instances.numInstances(); i++)
			this.candidates.add(this.instances.instance(i));
		this.algorithmOffset = algorithmOffset;
	}

	public void iterate()
	{
		Instance x;
			x = this.acquisitionFunction.getNext(this.h, this.surrogateModel, this.candidates);
		for(int i = this.algorithmOffset; i < HyperparameterCombination.HYPERPARAMETER_ALGORITHM_RANGE + this.algorithmOffset; i++)
			if(x.getValue(i) == 1)
				this.algorithmSelection[i - this.algorithmOffset]++;
		if(Logger.LEVEL == Level.FINE || Logger.LEVEL == Level.FINER || Logger.LEVEL == Level.FINEST)
		{
			double bestAcc = 0;
			for(int i = 0; i < this.h.numInstances(); i++)
				bestAcc = Math.max(bestAcc, this.h.instance(i).target());
			for(int i = 0; i < this.candidates.size(); i++)
				bestAcc = Math.max(bestAcc, this.candidates.get(i).target());
//			int[] keys = x.getKeys();
//			double[] values = x.getValues();
			String hpString = "";
//			for(int i = 0; i < keys.length; i++) {
//				hpString += keys[i] + ":" + values[i] + " ";
//			}
			for(int i = 0; i < HyperparameterCombination.HYPERPARAMETER_INDEX_RANGE_MAX ; i++) {
				hpString += i + ":" + x.getValue(i) + " ";
			}
				
			Logger.fine("Choosing hyper-parameter " + hpString + "\nThis accuracy: " + x.target() + " (best=" + (this.bestInstance == null ? "?" : this.bestInstance.target()) + ", max=" + bestAcc
					+ ").");

			// Multiboost
			// int iterations = (int) Math.round(Math.pow(2, x.getValue(0) * 13.2877124));
			// int products = (int) Math.round(Math.pow(2, x.getValue(1) * 4.9068906));
			// System.out.println((108 - this.candidates.size() + 1) + "," + iterations + "," + products);

			// System.out.println("\t\t\t\t\t\t<tr>");
			// System.out.println("\t\t\t\t\t\t\t<th scope=\"row\">" + (108 - this.candidates.size() + 1) + "</th>");
			// System.out.println("\t\t\t\t\t\t\t<td>" + iterations + "</td>");
			// System.out.println("\t\t\t\t\t\t\t<td>" + products + "</td>");
			// System.out.println("\t\t\t\t\t\t</tr>");

			// SVM
			// String kernel = "Linear";
			// if(x.getValue(0) == 1)
			// kernel = "RBF";
			// else if(x.getValue(1) == 1)
			// kernel = "Pol";
			// double C = Math.pow(2, Math.round(x.getValue(3) * 6));
			// double gamma = (double) Math.round(Math.pow(2, x.getValue(4) * 13.2877124) * 10000) / 10000;
			// double degree = (double) Math.round(Math.pow(2, x.getValue(5) * 3.32192809) * 1000000) / 1000000;
			// System.out.println(kernel + " " + (C >= 1 ? ((int) C + "") : C) + " " + (kernel.equals("RBF") ? (gamma >= 1 ? ((int) gamma + "") : gamma) : "") + " " + (kernel.equals("Pol") ? (int) Math.round(degree) : ""));

			// DecimalFormat formatter = new DecimalFormat();
			// formatter.setMaximumFractionDigits(10);
			// System.out.println("\t\t\t\t<tr>");
			// System.out.println("\t\t\t\t\t<th scope=\"row\">" + (288 - this.candidates.size() + 1) + "</th>");
			// System.out.println("\t\t\t\t\t<td>" + (kernel.equals("Pol") ? "Polynomial" : kernel) + "</td>");
			// System.out.println("\t\t\t\t\t<td>" + (C >= 1 ? ((int) C + "") : formatter.format(C).replace(",",".")) + "</td>");
			// System.out.println("\t\t\t\t\t<td>" + (kernel.equals("Pol") ? (int) Math.round(degree) : "-") + "</td>");
			// System.out.println("\t\t\t\t\t<td>" + (kernel.equals("RBF") ? (gamma >= 1 ? ((int) gamma + "") : formatter.format(gamma).replace(",",".")) : "-") + "</td>");
			// System.out.println("\t\t\t\t</tr>");
			// System.out.println((288 - this.candidates.size() + 1) + "," + (kernel.equals("Pol") ? "Polynomial" : kernel) + "," + (C >= 1 ? ((int) C + "") : formatter.format(C).replace(",",".")) + "," + (kernel.equals("Pol") ? (int) Math.round(degree) : "-") + "," + (kernel.equals("RBF") ? (gamma
			// >= 1 ? ((int) gamma + "") : formatter.format(gamma).replace(",",".")) : "-"));
			// DecimalFormat formatter = new DecimalFormat();
			// formatter.setMaximumFractionDigits(10);
			// try
			// {
			// BufferedReader br = new BufferedReader(new FileReader("data/svm/logs/covtypem-SVM" + kernel + "-C-" + (C >= 1 ? ((int) C) + "" : formatter.format(C).replace(",",".")) + (kernel.equals("RBF") ? "-G-" + (gamma >= 1 ? ((int) gamma + "") : formatter.format(gamma).replace(",",".")) :
			// (kernel.equals("Pol") ? "-D-" + (int) degree : "")) + ".txt"));
			// this.time = 0;
			// String line = "";
			// while((line = br.readLine()) != null)
			// {
			// if(line.contains("TrainTime"))
			// {
			// String[] split = line.split("=");
			// time = Double.parseDouble(split[1]);
			// }
			// }
			// br.close();
			// Logger.fine("Time needed for evaluating f: " + time);
			// }
			// catch(IOException e)
			// {
			// e.printStackTrace();
			// }
		}
		this.candidates.remove(x);
		if(this.bestInstance == null || this.bestInstance.target() < x.target())
			this.bestInstance = x;
		this.h.add(x);
		if(this.surrogateModel != null)
		{
			if(this.surrogateModel instanceof OnlineLearnable)
			{
				if(this.surrogateModel instanceof GaussianProcess && this.h.numInstances() == 1)
					this.surrogateModel.train(this.h);
				else
					((OnlineLearnable) this.surrogateModel).onlineUpdate(x);
			}
			else
				this.surrogateModel.train(this.h);
		}
	}

	public double getBestAccuracy()
	{
		return this.bestInstance.target();
	}

	public int getBestRank()
	{
		return InstanceUtils.getRank(this.instances, this.bestInstance);
	}

	public double getTime()
	{
		return this.time;
	}

	public int[] getAlgorithmSelection()
	{
		return this.algorithmSelection;
	}
}
