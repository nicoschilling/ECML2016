package de.ismll.hylap.metadataset;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

/**
 * Parses the cluster log files of the multiboost classifier, extracts needed information and save it to dense format.
 */
public class ParseMultiboostLogs
{
	public static void main(String[] args) throws IOException
	{
		HashMap<String, Integer> numOfInstances = new HashMap<String, Integer>();
		HashSet<String> strangeDatasets = new HashSet<String>();
		for(File file : new File("data/multiboost/logs").listFiles())
		{
			String[] split = file.getName().split("_");
			String dataset = split[0];

			if(file.getName().contains(".e") && !strangeDatasets.contains(dataset))
			{
				BufferedReader br = new BufferedReader(new FileReader(file));
				String line;
				while((line = br.readLine()) != null)
				{
					if(line.contains("Backtrace"))
					{
						strangeDatasets.add(dataset);
						break;
					}
				}
				br.close();
			}
		}

		for(String d : strangeDatasets)
			System.out.println(d);

		HashSet<String> runningExperiments = new HashSet<String>();
		HashMap<String, BufferedWriter> writers = new HashMap<String, BufferedWriter>();
		int running = 0;
		for(File file : new File("data/multiboost/logs").listFiles())
		{
			System.out.println(file.getName());
			String[] fileSplit = file.getName().split("_");
			String dataset = fileSplit[0];
			String iterations = fileSplit[1];
			fileSplit = fileSplit[2].split("\\.");
			String numProductTerms = fileSplit[0];
			if(strangeDatasets.contains(dataset))
				continue;

			if(file.getName().contains(".o"))
			{
				BufferedReader br = new BufferedReader(new FileReader(file));
				String line;
				double trainErr = -1, testErr = -1;
				while((line = br.readLine()) != null)
				{
					if(line.contains("Error Rate"))
					{
						String[] split = line.split(" ");
						String value = split[split.length - 1];

						double v = 0;
						if(value.startsWith("100"))
							v = 1;
						else if(Double.parseDouble(value.substring(0, split[split.length - 1].length() - 1)) >= 10)
							v = Double.parseDouble("0." + split[split.length - 1].substring(0, split[split.length - 1].length() - 1).replace(".", ""));
						else
							v = Double.parseDouble("0.0" + split[split.length - 1].substring(0, split[split.length - 1].length() - 1).replace(".", ""));
						if(trainErr == -1)
							trainErr = v;
						else
							testErr = v;
					}
				}
				br.close();
				if(testErr != -1)
				{
					if(numOfInstances.containsKey(dataset))
						numOfInstances.put(dataset, numOfInstances.get(dataset) + 1);
					else
						numOfInstances.put(dataset, 1);
					BufferedWriter bw = null;
					if(writers.containsKey(dataset))
						bw = writers.get(dataset);
					else
					{
						bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("data/multiboost/extracted/" + dataset), "utf-8"));
						writers.put(dataset, bw);
					}
					bw.write((1 - testErr) + " " + (1 - trainErr) + " " + iterations + " " + numProductTerms);
					bw.newLine();
				}
				else
				{
					System.out.println(file.getName());
					runningExperiments.add(dataset);
					running++;
				}
			}
		}

		for(BufferedWriter bw : writers.values())
			bw.close();

		System.out.println("Non complete datasets:");
		for(Entry<String, Integer> e : numOfInstances.entrySet())
		{
			if(e.getValue() != 108)
			{
				System.out.println(e.getKey());
				new File("data/multiboost/extracted/" + e.getKey()).delete();
			}
		}
		System.out.println("Running jobs: " + running);
	}
}
