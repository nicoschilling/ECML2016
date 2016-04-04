package de.ismll.hylap.metadataset;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;

/**
 * Combines a data set in libsvm format containing only the hyper-parameters with the meta-features.
 */
public class HyperparameterMetafeatureCombiner
{
	public static void main(String[] args)
	{
		String inputFolder = "data/svm/converted";
		String outputFolder = "data/svm/final";

		try
		{
			HashMap<String, String> metafeatures = new HashMap<String, String>();
			BufferedReader br = new BufferedReader(new FileReader(new File("data/metafeatures.txt")));
			String line;
			while((line = br.readLine()) != null)
			{
				String[] split = line.split(",");
				metafeatures.put(split[0], split[1]);
			}
			br.close();

			for(File file : new File(inputFolder).listFiles())
			{
				PrintWriter writer = new PrintWriter(outputFolder + "/" + file.getName(), "UTF-8");
				br = new BufferedReader(new FileReader(file));
				while((line = br.readLine()) != null)
				{
					writer.write(line + " " + metafeatures.get(file.getName()) + System.getProperty("line.separator"));
				}
				br.close();
				writer.close();
			}
		}
		catch(IOException e)
		{
			e.printStackTrace();
		}
	}
}
