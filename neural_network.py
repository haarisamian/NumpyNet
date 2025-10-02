"""
neural_network.py

A simple, fully-connected neural network implemented from scratch using NumPy.
"""

import numpy as np
import random
from typing import List, Tuple, Any

def sig(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1/(1+np.exp(-x))

def dsig(x: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid function."""
    s = sig(x)
    return s * (1 - s)

class NeuralNetwork:
    """
    A simple fully-connected neural network for binary classification.
    """

    def __init__(self, dim_in: int, dim_out: int, dim_hidden: int, layers: int):
        """
        Initialize the neural network.

        Args:
            dim_in (int): Number of input features.
            dim_out (int): Number of output units.
            dim_hidden (int): Number of hidden units per hidden layer.
            layers (int): Total number of layers (input + hidden + output).
        """
        self.layers = layers
        self.hidden_layers = layers - 2
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden

        # Xavier/Glorot initialization for sigmoid/tanh
        # Each weight matrix is initialized with stddev = sqrt(2 / (fan_in + fan_out))
        self.WI = np.random.randn(dim_hidden, dim_in) * np.sqrt(2. / (dim_in + dim_hidden))
        self.WH = np.random.randn(dim_hidden, dim_hidden, self.hidden_layers) * np.sqrt(2. / (2 * dim_hidden))
        self.WO = np.random.randn(dim_out, dim_hidden) * np.sqrt(2. / (dim_hidden + dim_out))

        # Biases initialized to 0
        self.BI = np.zeros((dim_hidden, 1))
        self.BH = np.zeros((dim_hidden, self.hidden_layers))
        self.BO = np.zeros((dim_out, 1))

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Perform a forward pass through the network.

        Args:
            x (np.ndarray): Input array.

        Returns:
            Tuple containing the output, list of z values, and list of activations.
        """
        input = x.reshape(-1, 1)
        z_layer: List[np.ndarray] = []
        a_layer: List[np.ndarray] = []

        a_layer.append(input)

        # First layer
        z = self.WI @ input + self.BI
        z_layer.append(z)
        a = sig(z)
        a_layer.append(a)

        # Hidden layers
        for i in range(self.hidden_layers):
            z = self.WH[:,:,i] @ a + self.BH[:,i].reshape(-1,1)
            z_layer.append(z)
            a = sig(z)
            a_layer.append(a)

        # Output layer
        z = self.WO @ a + self.BO
        a = sig(z)

        y_hat = a

        return y_hat, z_layer, a_layer

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output for a given input.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Predicted output.
        """
        y = self.forward(x)
        return y[0]

    def classify(self, x: np.ndarray) -> int:
        """
        Classify the input as 0 or 1.

        Args:
            x (np.ndarray): Input array.

        Returns:
            int: Predicted class (0 or 1).
        """
        y_hat = self.predict(x)
        return int(y_hat.item() >= 0.5)

    def loss(self, x: np.ndarray, label: float) -> float:
        """
        Compute the binary cross-entropy loss.

        Args:
            x (np.ndarray): Input array.
            label (float): True label.

        Returns:
            float: Loss value.
        """
        y_hat = self.predict(x)
        y = label
        loss = -y*np.log(y_hat) - (1-y)*np.log(1-y_hat)
        return float(loss)

    def accuracy(self, inputs: List[np.ndarray], labels: List[float]) -> float:
        """
        Compute the accuracy of the model.

        Args:
            inputs (List[np.ndarray]): List of input arrays.
            labels (List[float]): List of true labels.

        Returns:
            float: Accuracy value.
        """
        correct = 0
        total = len(inputs)
        for i in range(len(inputs)):
            y = self.classify(inputs[i])
            if y == labels[i]:
                correct += 1
        return correct / total

    def backprop(self, x: np.ndarray, label: float) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Perform backpropagation and compute gradients.

        Args:
            x (np.ndarray): Input array.
            label (float): True label.

        Returns:
            Tuple of weight gradients and bias gradients.
        """
        y_hat, zL, aL = self.forward(x)

        dLdz = y_hat - label
        dLdz = dLdz.reshape(-1,1)

        # Initialize weight gradient matrices for I, H, O layers
        WI_grad = np.zeros((self.dim_hidden, self.dim_in))
        WH_grad = np.zeros((self.dim_hidden, self.dim_hidden, self.hidden_layers))
        WO_grad = np.zeros((self.dim_out, self.dim_hidden))

        # Bias gradients initialized to 0
        BI_grad = np.zeros((self.dim_hidden, 1))
        BH_grad = np.zeros((self.dim_hidden, self.hidden_layers))
        BO_grad = np.zeros((self.dim_out, 1))

        # Calculate output gradients
        WO_grad = dLdz @ aL[-1].T
        BO_grad = dLdz
        dLdz = dsig(zL[-1]) * (self.WO.T @ dLdz)

        # Propagate through hidden layers
        for i in range(self.hidden_layers - 1, -1, -1):
            WH_grad[:,:,i] = dLdz @ aL[i+1].T
            BH_grad[:,i] = dLdz.flatten()
            dLdz = dsig(zL[i]) * (self.WH[:,:,i].T @ dLdz)

        # Calculate input layer gradients
        WI_grad = dLdz @ aL[0].T
        BI_grad = dLdz

        W_grads = (WI_grad, WH_grad, WO_grad)
        B_grads = (BI_grad, BH_grad, BO_grad)

        return W_grads, B_grads

    def shuffle(self, inputs: List[np.ndarray], outputs: List[Any]) -> Tuple[List[np.ndarray], List[Any]]:
        """
        Shuffle the inputs and outputs in unison.

        Args:
            inputs (List[np.ndarray]): List of input arrays.
            outputs (List[Any]): List of outputs.

        Returns:
            Tuple of shuffled inputs and outputs.
        """
        combined = list(zip(inputs, outputs))
        random.shuffle(combined)
        i, o = zip(*combined)
        i, o = list(i), list(o)
        return i, o

    def train(
        self,
        inputs: List[np.ndarray],
        labels: List[float],
        iterr: int,
        learning_rate: float,
        batch_size: int = 1
    ) -> None:
        """
        Train the neural network.

        Args:
            inputs (List[np.ndarray]): List of input arrays.
            labels (List[float]): List of true labels.
            iterr (int): Number of training iterations (epochs).
            learning_rate (float): Learning rate.
            batch_size (int, optional): Batch size. Defaults to 1.

        Raises:
            ValueError: If batch_size is greater than the number of input samples.
        """
        if batch_size > len(inputs):
            raise ValueError("batch_size cannot be greater than the number of input samples")

        for _ in range(iterr):
            inputs, labels = self.shuffle(inputs, labels)
            batches = len(inputs) // batch_size
            n = len(inputs)
            loss = 0
            for i in range(batches):
                WI_g = np.zeros((self.dim_hidden, self.dim_in))
                WH_g = np.zeros((self.dim_hidden, self.dim_hidden, self.hidden_layers))
                WO_g = np.zeros((self.dim_out, self.dim_hidden))

                BI_g = np.zeros((self.dim_hidden, 1))
                BH_g = np.zeros((self.dim_hidden, self.hidden_layers))
                BO_g = np.zeros((self.dim_out, 1))

                for j in range(batch_size):
                    index = i*batch_size + j
                    W_curr, B_curr = self.backprop(inputs[index], labels[index])
                    loss += self.loss(inputs[index], labels[index])
                    WI_g += (W_curr[0] / batch_size)
                    WH_g += (W_curr[1] / batch_size)
                    WO_g += (W_curr[2] / batch_size)

                    BI_g += (B_curr[0] / batch_size)
                    BH_g += (B_curr[1] / batch_size)
                    BO_g += (B_curr[2] / batch_size)

                self.WI -= learning_rate * WI_g
                self.WH -= learning_rate * WH_g
                self.WO -= learning_rate * WO_g

                self.BI -= learning_rate * BI_g
                self.BH -= learning_rate * BH_g
                self.BO -= learning_rate * BO_g

            WI_g = np.zeros((self.dim_hidden, self.dim_in))
            WH_g = np.zeros((self.dim_hidden, self.dim_hidden, self.hidden_layers))
            WO_g = np.zeros((self.dim_out, self.dim_hidden))

            BI_g = np.zeros((self.dim_hidden, 1))
            BH_g = np.zeros((self.dim_hidden, self.hidden_layers))
            BO_g = np.zeros((self.dim_out, 1))

            remainder = len(inputs) % batch_size
            if remainder != 0:
                for k in range(len(inputs)-remainder, len(inputs)):
                    W_curr, B_curr = self.backprop(inputs[k], labels[k])
                    loss += self.loss(inputs[k], labels[k])
                    WI_g += (W_curr[0] / remainder)
                    WH_g += (W_curr[1] / remainder)
                    WO_g += (W_curr[2] / remainder)

                    BI_g += (B_curr[0] / remainder)
                    BH_g += (B_curr[1] / remainder)
                    BO_g += (B_curr[2] / remainder)

            self.WI -= learning_rate * WI_g
            self.WH -= learning_rate * WH_g
            self.WO -= learning_rate * WO_g

            self.BI -= learning_rate * BI_g
            self.BH -= learning_rate * BH_g
            self.BO -= learning_rate * BO_g

            avg_loss = loss/n
            if _ % 100 == 0:
                print("AVERAGE EPOCH LOSS: " + str(avg_loss))
        return