package com.matthewmohandiss.fann;

import java.util.ArrayList;

/**
 * Created by Matthew on 3/25/16.
 */
public class Node {
	public double sum;
	public ArrayList<Double> weights;

	public Node(int NodesInPreviousLayer) {
		weights = new ArrayList<>(NodesInPreviousLayer);

		for (int i = 0; i < NodesInPreviousLayer; i++) {
			weights.add(Math.random()); //randomize weights
		}
	}
}
