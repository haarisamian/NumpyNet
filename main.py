"""
main.py

Example script for training and evaluating a neural network binary classifier using the NeuralNetwork class.
Loads data from ASCII files, trains the model, and evaluates accuracy.
"""

from neural_network import NeuralNetwork
import numpy as np
from typing import List

def load_ascii_data(filename: str) -> (List[np.ndarray], List[int]):
    """
    Load input and label data from an ASCII file.

    Args:
        filename (str): Path to the ASCII data file.

    Returns:
        Tuple[List[np.ndarray], List[int]]: Tuple of input arrays and integer labels.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    inputs = []
    labels = []

    # The file is split in half: first half is inputs, second half is labels
    num_lines = len(lines)
    num_inputs = num_lines // 2

    # Parse input feature vectors
    for i in range(num_inputs):
        # Each input line: space-separated floats
        inputs.append(np.array([float(x) for x in lines[i].strip().split()]))

    # Parse labels
    for i in range(num_inputs, num_lines):
        # Each label line: single float, cast to int
        labels.append(int(float(lines[i].strip().split()[0])))

    return inputs, labels

def main():
    # Load training data
    train_inputs, train_labels = load_ascii_data('anntrainingdataascii')

    # Instantiate a neural network: input dim 4, output dim 1, hidden width 8, 6 layers
    classifier = NeuralNetwork(4, 1, 8, 6)

    # Evaluate initial accuracy
    initial_accuracy = classifier.accuracy(train_inputs, train_labels)
    print(f"Initial training accuracy: {initial_accuracy:.4f}")

    # Train the classifier
    classifier.train(train_inputs, train_labels, iterr=1000, learning_rate=1e-2, batch_size=20)

    # Evaluate post-training accuracy
    post_training_accuracy = classifier.accuracy(train_inputs, train_labels)
    print(f"Post-training accuracy: {post_training_accuracy:.4f}")

    # Load test data and evaluate
    test_inputs, test_labels = load_ascii_data('anntestingdataascii')
    test_accuracy = classifier.accuracy(test_inputs, test_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Example: try different architectures
    classifier2 = NeuralNetwork(4, 1, 16, 6)  # wider
    classifier3 = NeuralNetwork(4, 1, 8, 10)  # deeper

    classifier2.train(train_inputs, train_labels, iterr=1000, learning_rate=1e-2, batch_size=20)
    classifier3.train(train_inputs, train_labels, iterr=1000, learning_rate=1e-2, batch_size=20)

    train_acc2 = classifier2.accuracy(train_inputs, train_labels)
    train_acc3 = classifier3.accuracy(train_inputs, train_labels)
    test_acc2 = classifier2.accuracy(test_inputs, test_labels)
    test_acc3 = classifier3.accuracy(test_inputs, test_labels)

    print(f"Train Accuracies: {post_training_accuracy:.4f} {train_acc2:.4f} {train_acc3:.4f}")
    print(f"Test Accuracies: {test_accuracy:.4f} {test_acc2:.4f} {test_acc3:.4f}")

if __name__ == "__main__":
    main()