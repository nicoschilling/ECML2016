package de.ismll.core;

import java.util.ArrayList;

public class SplitMap
{
	public ArrayList<Integer> validationSamples;
	public ArrayList<Integer> trainSamples;
	public ArrayList<Integer> testSamples;
	
	public SplitMap() {
		validationSamples = new ArrayList<Integer>();
		trainSamples = new ArrayList<Integer>();
		testSamples = new ArrayList<Integer>();
	}
}
