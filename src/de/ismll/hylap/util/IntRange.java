package de.ismll.hylap.util;

import java.util.StringTokenizer;

/**
 * 
 * syntax is <number>DELIMITER_INTERVAL<number>[DELIMITER_INDIVIDUAL_INTERVAL<number>DELIMITER_INTERVAL<number>]*
 * 
 * @author Nicolas Schilling
 * 
 */
public class IntRange
{

	private static final String DELIMITER_INTERVAL = ",";

	private static final String DELIMITER_INDIVIDUAL_INTERVAL = ";";

	private int[] usedIndexes;

	/**
	 * Integer, specifying the size of the intVector v.
	 */
	int size;

	public IntRange()
	{

	}

	public IntRange(int[] convert)
	{
		usedIndexes = new int[convert.length];
		for(int i = 0; i < convert.length; i++)
		{
			usedIndexes[i] = convert[i];
		}
	}

	public IntRange(String in)
	{
		StringTokenizer intervals = new StringTokenizer(in, DELIMITER_INDIVIDUAL_INTERVAL);
		// String[] split = in.split(DELIMITER_INDIVIDUAL_INTERVAL);

		int tokenCount = intervals.countTokens();

		int token = 0;

		int[][] startAndEnd = new int[tokenCount][2];

		int absoluteLength = 0;
		while(intervals.hasMoreElements())
		{
			String interval = (String) intervals.nextElement();

			StringTokenizer numbers = new StringTokenizer(interval, DELIMITER_INTERVAL);

			int start = Integer.parseInt((String) numbers.nextElement());
			int end = Integer.parseInt((String) numbers.nextElement());

			startAndEnd[token][0] = start;
			startAndEnd[token][1] = end;

			int difference = end - start + 1;

			absoluteLength += difference;

			token++;

		}

		int[] usedIdx = new int[absoluteLength];
		int count = 0;
		for(int i = 0; i < tokenCount; i++)
		{
			int start = startAndEnd[i][0];
			int end = startAndEnd[i][1];
			for(int k = start; k <= end; k++)
			{
				usedIdx[count] = k;
				count++;
			}
		}
		this.usedIndexes = new int[usedIdx.length];

		for(int i = 0; i < usedIdx.length; i++)
		{
			usedIndexes[i] = usedIdx[i];
		}
	}

	/**
	 * Converts a String to an IntRange Object. Intervals on used indexes are seperated by a comma, you can use a semicolon to use more than one interval. Example: 1,5;15,22 will use the indexes 1 to
	 * 5 and 15 to 22
	 * 
	 * @param in
	 * @return
	 */
	public static IntRange convert(Object in)
	{
		String use;
		if(in instanceof String)
		{
			use = (String) in;
		}
		else
		{
			use = in.toString();
		}

		StringTokenizer intervals = new StringTokenizer(use, DELIMITER_INDIVIDUAL_INTERVAL);

		int tokenCount = intervals.countTokens();

		int token = 0;

		int[][] startAndEnd = new int[tokenCount][2];

		int absoluteLength = 0;
		while(intervals.hasMoreElements())
		{
			String interval = (String) intervals.nextElement();

			StringTokenizer numbers = new StringTokenizer(interval, DELIMITER_INTERVAL);

			int start = Integer.parseInt((String) numbers.nextElement());
			int end = Integer.parseInt((String) numbers.nextElement());

			startAndEnd[token][0] = start;
			startAndEnd[token][1] = end;

			int difference = end - start + 1;

			absoluteLength += difference;

			token++;

		}

		int[] usedIdx = new int[absoluteLength];
		int count = 0;
		for(int i = 0; i < tokenCount; i++)
		{
			int start = startAndEnd[i][0];
			int end = startAndEnd[i][1];
			for(int k = start; k <= end; k++)
			{
				usedIdx[count] = k;
				count++;
			}
		}

		IntRange ret = new IntRange(usedIdx);

		return ret;
	}

	public int[] getUsedIndexes()
	{
		return usedIndexes;
	}

	public void setUsedIndexes(int[] usedIndexes)
	{
		this.usedIndexes = usedIndexes;
	}

}
