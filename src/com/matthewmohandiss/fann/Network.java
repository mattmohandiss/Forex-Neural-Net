package com.matthewmohandiss.fann;

import java.util.ArrayList;

/**
 * Created by Matthew on 3/25/16.
 */

public class Network {
	public ArrayList<Double> inputLayer;
	public ArrayList<Node> outputLayer;
	private ArrayList<Node> hiddenLayer;

	public Network(int inputLayerSize, int hiddenLayerSize, int outputLayerSize) {
		inputLayer = new ArrayList<>(inputLayerSize);
		hiddenLayer = new ArrayList<>(hiddenLayerSize); // Note that the weights between layers are stored in the 'downstream' layer
		outputLayer = new ArrayList<>(outputLayerSize);

		for (int i = 0; i < inputLayerSize; i++) {
			inputLayer.add(1.0);
		}

		for (int i = 0; i < hiddenLayerSize; i++) {
			hiddenLayer.add(new Node(inputLayerSize)); // create hidden layer
		}

		for (int i = 0; i < outputLayerSize; i++) {
			outputLayer.add(new Node(hiddenLayerSize)); // create output layer
		}

		hiddenLayer.get(hiddenLayer.size() - 1).sum = 1.0; // set bias
	}

	public Double propagate() {
		for (int i = 0; i < hiddenLayer.size() - 1; i++) { // minus 1 to preserve bias value
			for (int j = 0; j < inputLayer.size(); j++) {
				hiddenLayer.get(i).sum += inputLayer.get(j) * hiddenLayer.get(i).weights.get(j);
			}
			hiddenLayer.get(i).sum = activate(hiddenLayer.get(i).sum); // apply activation function
		}

		for (int i = 0; i < outputLayer.size(); i++) {
			for (int j = 0; j < hiddenLayer.size(); j++) {
				outputLayer.get(i).sum += hiddenLayer.get(j).sum * outputLayer.get(i).weights.get(j);
			}
			outputLayer.get(i).sum = activate(outputLayer.get(i).sum); // apply activation function
		}
		return outputLayer.get(0).sum;
	}

	public void backPropagate(Double actual, Double target) { //backpropagate starting with the otuput layer working backwards
		Double outputDelta = -(target - actual) * actual * (1 - actual);

		for (int i = 0; i < hiddenLayer.size(); i++) {
			for (int j = 0; j < inputLayer.size(); j++) {
				Double dErrordOutput = outputDelta * outputLayer.get(0).weights.get(i);
				Double dOutputdHiddenNet = hiddenLayer.get(i).sum * (1 - hiddenLayer.get(i).sum);
				Double dInputdWeight = hiddenLayer.get(i).weights.get(j);
				hiddenLayer.get(i).weights.set(j, hiddenLayer.get(i).weights.get(j) - (dErrordOutput * dOutputdHiddenNet * dInputdWeight));
			}
		}

		for (int i = 0; i < hiddenLayer.size(); i++) {
			Double weightGradient = outputDelta * hiddenLayer.get(i).sum; // optionally multiply hiddenLayer sum by learning rate
			outputLayer.get(0).weights.set(i, outputLayer.get(0).weights.get(i) - weightGradient); // adjust weights between hidden and output layers
		}
	}

	private double activate(double x) {
		return 1 / (1 + Math.pow(Math.E, -x));      // standard sigmoid
		//return (2/(1+Math.pow(Math.E,-x)))-1;     // sigmoid stretched from -1 to 1
		//return Math.tan(x);                       //tangent function
	}

	public void clear() {
		for (int i = 0; i < hiddenLayer.size(); i++) {
			hiddenLayer.get(i).sum = 0; //clear hidden layer
		}
		for (int i = 0; i < outputLayer.size(); i++) {
			outputLayer.get(i).sum = 0; //clear outputs
		}
		hiddenLayer.get(hiddenLayer.size() - 1).sum = 1.0; // set bias
	}

	public void print() {
		String inputLayerPrint = "";
		String hiddenLayerPrint = "";
		String outputLayerPrint = "";

		for (Double input :
				inputLayer) {
			inputLayerPrint += input + "    ";
		}
		for (Node node :
				hiddenLayer) {
			hiddenLayerPrint += node.sum + "    ";
		}
		for (Node node :
				outputLayer) {
			outputLayerPrint += node.sum + "    ";
		}

		for (int i = 0; i < (hiddenLayerPrint.length() - inputLayerPrint.length()) / 2; i++) {
			System.out.print(" ");
		}
		System.out.println(inputLayerPrint.trim());
		System.out.println(hiddenLayerPrint.trim());
		for (int i = 0; i < (hiddenLayerPrint.length() - outputLayerPrint.length()) / 2; i++) {
			System.out.print(" ");
		}
		System.out.println(outputLayerPrint.trim());
	}

	public double averageWeights() { //for debugging purposes
		Double sum = 0.0;
		int count = 0;
		for (int i = 0; i < hiddenLayer.size(); i++) {
			for (int j = 0; j < inputLayer.size(); j++) {
				sum += hiddenLayer.get(i).weights.get(j);
				count++;
			}
		}

		for (int i = 0; i < outputLayer.size(); i++) {
			for (int j = 0; j < hiddenLayer.size(); j++) {
				sum += outputLayer.get(i).weights.get(j);
				count++;
			}
		}
		return sum / count;
	}
}
