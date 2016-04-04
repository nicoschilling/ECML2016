package de.ismll.hylap.results;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import de.ismll.core.Instances;

public class SMBOResults
{
	public static void main(String[] args) throws IOException
	{
		// File[] files = new File("data/svm/final").listFiles();
		// Random.setSeed(System.nanoTime());
		// for(int i = 0; i < 25; i++)
		// {
		// File randomFile = files[Random.nextInt(files.length)];
		// File targetFile = new File("data/svm/final_small/"+randomFile.getName());
		// if(targetFile.exists())
		// i--;
		// else
		// Files.copy(randomFile.toPath(), targetFile.toPath());
		// }
		// System.exit(0);
//		int tries = 288;
//		 for(int i = 2; i < 14432; i += tries + 1)
//		 {
//		 System.out.print("A" + i + ";");
//		 }
//		 System.exit(0);
		
//		for(int i = 0; i < 110; i ++)
//		 {
//			for(int j = 0; j < 288; j++)
//				System.out.println("=A" + ((i+1) * 289));
//			System.out.println();
//		 }
//		 System.exit(0);
		
		int tries = 288;
		String classifier = "svm";
		boolean printAccuracy = false;
		String[] methods = { "random", "gp", "rf", "scot_0.001", "mklgp_2", "gp_nnfei" };
		
		
		String datasetFolder = classifier + "/final_med";
		String resultFolder = "results/" + classifier + "_med/";
		File[] files = new File("data/" + datasetFolder).listFiles();

		for(String m : methods)
		{
			System.out.print(m + "\t");
		}
		System.out.println("best");

		for(File f : files)
		{
			Instances instances = new Instances(f);
			double bestResult = -1;
			for(int i = 0; i < instances.numInstances(); i++)
				bestResult = Math.max(bestResult, instances.instance(i).target());
			String[][] result = new String[tries][methods.length];
			for(int i = 0; i < methods.length; i++)
			{
				BufferedReader br = null;
				for(File f2 : new File(resultFolder).listFiles())
				{
					if(f2.getName().equals(f.getName() + "_" + methods[i]))
					{
						br = new BufferedReader(new FileReader(f2));
						break;
					}
				}

				if(br == null)
					continue;

				String line = br.readLine();
				int j = 0;
				while((line = br.readLine()) != null)
				{
					String[] split = line.split(",");
					result[j++][i] = split[(printAccuracy ? 0 : 2)];
				}
				br.close();
			}

			for(int i = 0; i < result.length; i++)
			{
				for(int j = 0; j < result[i].length; j++)
				{
					System.out.print((result[i][j] == null ? "" : result[i][j].replace(".", ",")) + "\t");
				}
				//Print best possible
				System.out.print((printAccuracy ? new String(""+bestResult).replace(".", ",") : "1"));
				System.out.println();
			}
			System.out.println();
		}
	}
}
