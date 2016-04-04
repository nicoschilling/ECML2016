package de.ismll.hylap.surrogateModel;

import de.ismll.core.Instance;

public interface OnlineLearnable
{
	/**
	 * Update the model considering one additional instancce.
	 */
	public void onlineUpdate(Instance instance);
}
