package de.ismll.hylap.surrogateModel;

import de.ismll.core.Instance;
import de.ismll.core.Instances;
import de.ismll.core.regression.GaussianProcessRegression;
import de.ismll.kernel.Kernel;

public class GaussianProcess implements SurrogateModel, OnlineLearnable
{
	private GaussianProcessRegression model = new GaussianProcessRegression();

	private Instances data;

	@Override
	public void train(Instances instances)
	{
		this.data = new Instances(instances.numValues());
		this.data.addAll(instances);
		this.model.train(this.data);
	}

	@Override
	public double[] predict(Instance instance)
	{
		return this.model.predictWithUncertainty(instance);
	}

	@Override
	public void onlineUpdate(Instance instance)
	{
		this.model.onlineUpdate(instance);
	}

	public Instances getInstances()
	{
		return this.data;
	}

	public Kernel getKernel()
	{
		return this.model.getKernel();
	}

	public void setKernel(Kernel kernel)
	{
		this.model.setKernel(kernel);
	}

	public boolean isLearnKernelParameters()
	{
		return this.model.isLearnKernelParameters();
	}

	public void setLearnKernelParameters(boolean learnKernelParameters)
	{
		this.model.setLearnKernelParameters(learnKernelParameters);
	}
	
	public void setEpochs(int epochs)
	{
		this.model.setEpochs(epochs);
	}
}
