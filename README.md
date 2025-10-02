# NumpyNet
A simple, fully-connected neural network implemented from scratch in NumPy.

NumpyNet is a simple, readable implementation of a feedforward neural network for binary classification. It's designed for learning, experimentation, or just an introduction to how neural networks work under the hood without any additional deep learning frameworks. 

All of the code related to training, intializing, evaluating the neural network is located in the neural_network.py file.

# Features
 - From-scratch implementation: No PyTorch, no TensorFlow—just NumPy arrays.
 - Customizable architecture: Choose your number of layers and hidden units.
 - Batch training: Mini-batch gradient descent.
 - Binary classification: Sigmoid activations and cross-entropy loss.

# Quickstart
Install requirements

```bash
pip install -r requirements.txt
```

Train and evaluate the network
```bash
python main.py
```

By default, this will:

Load training and test data from anntrainingdataascii and anntestingdataascii
Train a neural network on the data
Print initial and final accuracy, and compare different architectures

# Example Usage
```python
from neural_network import NeuralNetwork
import numpy as np

# 4 input features, 1 output, 8 hidden units, 6 layers
nn = NeuralNetwork(dim_in=4, dim_out=1, dim_hidden=8, layers=6)

x = np.random.randn(4)
y_pred = nn.predict(x)
print("Prediction:", y_pred)
```

# File Structure
```
.
├── neural_network.py   # Core neural network implementation
├── main.py             # Example: training and evaluation script   
├── anntrainingdataascii
├── anntestingdataascii
├── requirements.txt 
├── .gitignore
├── LICENSE
└── README.md
```

# How it Works
NumpyNet stacks fully-connected layers with sigmoid activations. Training uses mini-batch gradient descent (can be SGD with batch_size = 1) and backpropagation, all written from scratch, and the parameters are initialized using Xavier initialization for weights. You can easily change the number of layers, hidden units, or batch size.

# License
MIT License © 2025 Haaris Mian

