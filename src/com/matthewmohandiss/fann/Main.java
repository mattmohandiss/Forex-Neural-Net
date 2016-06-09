package com.matthewmohandiss.fann;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Created by Matthew on 3/25/16.
 */

public class Main {
	public static Network forexBot;
	public static Double goal;
	public static double min = 75.72;
	public static double max = 134.78;

	public static void main(String[] args) {
		forexBot = new Network(4 + 1, 5 + 1, 1);    // +1 adds bias to input and hidden layers
		long startTime = System.currentTimeMillis();
		int totalDataCount = 4507;                  //number of daily price entries in data file
		int trainingDataCount = 4000;               //number of entries that should be used for training

		System.out.println("Finding ideal weights using " + trainingDataCount + " test cases");

		for (int i = 1; i <= trainingDataCount; i++) { //iterate through training data assigning values to inputs

			try {
				fillInputs(i);
				ArrayList<Double> futureInputs = input(i + 1);
				goal = normalize(futureInputs.get(3));
			} catch (IOException exception) {
				exception.printStackTrace();
			}

			Double result = forexBot.propagate();
			forexBot.backPropagate(result, goal);
			forexBot.clear();
		}
		System.out.println("Finished in " + (System.currentTimeMillis() - startTime) + " milliseconds");

		//manualValidate();
		autoValidate(trainingDataCount + 1, totalDataCount);
	}

	private static void autoValidate(int startLine, int endLine) {
		System.out.println();
		System.out.println("predictions:");
		for (int i = startLine; i <= endLine; i++) {
			try {
				fillInputs(i);
			} catch (IOException e) {
				e.printStackTrace();
			}
			System.out.println(unNormalize(forexBot.propagate()));
			forexBot.clear();
		}
	}

	private static void manualValidate() {
		Scanner reader = new Scanner(System.in);
		System.out.println();
		System.out.println("Enter data in the form 'open,low,high,close'");

		String str = reader.next();

		int firstComma = str.indexOf(",");
		int secondComma = str.indexOf(",", firstComma + 1);
		int thirdComma = str.indexOf(",", secondComma + 1);
		double open = Double.parseDouble(str.substring(0, firstComma));
		double low = Double.parseDouble(str.substring(firstComma + 1, secondComma));
		double high = Double.parseDouble(str.substring(secondComma + 1, thirdComma));
		double close = Double.parseDouble(str.substring(thirdComma + 1));
		forexBot.inputLayer.set(0, normalize(open));
		forexBot.inputLayer.set(1, normalize(low));
		forexBot.inputLayer.set(2, normalize(high));
		forexBot.inputLayer.set(3, normalize(close));

		System.out.println("Tomorrow's predicted close: " + unNormalize(forexBot.propagate()));
		System.out.println("Raw: " + forexBot.propagate());
		forexBot.print();
		System.out.println(forexBot.averageWeights());
		System.out.println();
		System.out.println("Enter more data in the form 'open,low,high,close'");
	}

	private static void fillInputs(int line) throws IOException {
		ArrayList<Double> inputs = input(line);
		forexBot.inputLayer.set(0, normalize(inputs.get(0))); //open
		forexBot.inputLayer.set(1, normalize(inputs.get(1))); //low
		forexBot.inputLayer.set(2, normalize(inputs.get(2))); //high
		forexBot.inputLayer.set(3, normalize(inputs.get(3))); //close
	}

	private static ArrayList input(int line) throws IOException {
		FileInputStream fs = new FileInputStream(System.getProperty("user.dir") + "/src/com/matthewmohandiss/fann/USDJPY_daily.csv");
		BufferedReader br = new BufferedReader(new InputStreamReader(fs));

		for (int i = 0; i < line; ++i)
			br.readLine();
		String str = br.readLine();
		ArrayList<Integer> commaList = new ArrayList<>(4);
		commaList.add(0);

		for (int i = 0; i < 4; i++) {
			commaList.add(str.indexOf(",", commaList.get(i) + 1));
		}
		commaList.remove(0);

		double open = Double.parseDouble(str.substring(commaList.get(0) + 1, commaList.get(1)));
		double low = Double.parseDouble(str.substring(commaList.get(1) + 1, commaList.get(2)));
		double high = Double.parseDouble(str.substring(commaList.get(2) + 1, commaList.get(3)));
		double close = Double.parseDouble(str.substring(commaList.get(3) + 1));
		ArrayList<Double> list = new ArrayList(4);
		list.add(open);
		list.add(low);
		list.add(high);
		list.add(close);

		//System.out.println("line: "+line+" open: "+open+" low: "+low+" high: "+high+" close: "+close);
		return list;
	}

	static double normalize(double value) {
		return (value - min) / (max - min);
	}

	static double unNormalize(double normalizedValue) {
		return min + normalizedValue * max - normalizedValue * min;
	}
}
