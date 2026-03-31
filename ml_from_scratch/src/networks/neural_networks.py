"""
neural_networks.py
~~~~~~~~~~~~~~~~~~
"""
import os
import sys
from pathlib import Path
import abc
import numpy as np
import pandas as pd
from typing import Any
from collections import OrderedDict
from pandas.core.series import Series
from scipy import optimize

SRC_PATH = Path(__file__).resolve().parent.parent
ACTIVATION_PATH = os.path.join(SRC_PATH, 'activations')
sys.path.append(ACTIVATION_PATH)
from src.activations.activation_functions import ActivationFunctions
print(f"{sys.version=}\n{sys.version_info=}")



class NeuralNetwork:
    """Neural network class"""

    def __init__(self, layer_sizes: list[int], X: np.array, y: np.array, learning_rate=0.01):
        self.X: np.array = X
        self.y: np.array = y
        self.y_hat: np.array = None

        self.layer_sizes: list[int] = layer_sizes
        self.learning_rate: float = learning_rate

        self.weights: list[Any] = []
        self.bias: list[Any] = []
        self.a_values: list[Any] = []
        self.z_values: list[Any] = []
        self.activation_functions: ActivationFunctions = ActivationFunctions()
        self.partial_derivatives: OrderedDict[str, Any] = OrderedDict()

    def initialize_weights(self):
        """Initialize the weight matrices"""
        iterations = len(self.layer_sizes) - 1
        for i in range(iterations):
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]))
            # print(f"W[{i}].shape := {self.weights[i].shape}")
            self.bias.append(np.zeros((self.layer_sizes[i+1], 1), dtype=float))
            # print(f"Bias[{i}].shape := {self.bias[i].shape}")

    def mean_squared_error(self) -> np.float64:
        """Mean Squared Error"""
        n = self.y.shape[0]
        J = (1/n) * np.sum((self.y_hat - self.y)**2)
        # print(f"{J=}")
        return J

    def mse_prime(self):
        """The derivative of the Mean Squared Error"""
        n = self.y.shape[0]
        J_prime = -(2/n) * (self.y - self.y_hat)
        print(f"{J_prime.shape=}")
        return J_prime

    def back_propagation(self, layer: int, current_partial=None):
        """Back Propagation algorithm"""
        next_a = self.a_values[(layer - 2)]
        current_z = self.z_values[(layer - 2)]
        current_w = self.weights[(layer - 2)]
        current_z_prime = self.activation_functions.sigmoid_prime(input_data=current_z)

        if layer <= 2:
            partial_1 = current_partial * current_z_prime
            partial_1 = partial_1 @ current_w.T
            partial_2 = next_a.T @ partial_1
            self.partial_derivatives['∂J/∂W' + str(layer - 1)] = partial_2
            return None

        if current_partial is None:
            j_prime = self.mse_prime()
            partial_1 = j_prime * current_z_prime
            partial_2 = next_a.T @ partial_1
            self.partial_derivatives['∂J/∂W' + str(layer - 1)] = partial_2
            return self.back_propagation(layer=layer-1, current_partial=partial_1)

        elif layer > 2 and current_partial is not None:
            partial_1 = current_partial * current_z_prime
            partial_1 = partial_1 @ current_w.T
            partial_2 = next_a.T @ partial_1
            self.partial_derivatives['∂J/∂W' + str(layer - 1)] = partial_2
            return self.back_propagation(layer=layer-1, current_partial=partial_1)

    def forward_propagation(self, current=None, layer=1):
        """Forward propagation algorithm"""
        global current_w
        global prev_a

        if layer >= len(self.layer_sizes) - 1:
            current_w = self.weights[-1]
        elif layer < len(self.layer_sizes) - 1:
            current_w = self.weights[layer - 1]

        if layer >= len(self.layer_sizes)-1:
            prev_a = current
            current_z = prev_a @ current_w #+ self.bias[layer - 1]
            current_a = self.activation_functions.sigmoid_function(input_data=current_z)
            self.z_values.append(current_z)
            self.a_values.append(current_a)
            self.y_hat = current_a
            return self.y_hat

        if current is None:
            prev_a = self.X
            self.a_values.append(prev_a)
        else:
            prev_a = current

        current_z = prev_a @ current_w + self.bias[layer-1].T
        current_a = self.activation_functions.sigmoid_function(input_data=current_z)
        self.z_values.append(current_z)
        self.a_values.append(current_a)
        return self.forward_propagation(current=current_a, layer=layer+1)






def main():
    X = np.array([
        [1, 2, 3],
        [0, 1, 0],
        [1, 0, 1]
    ], dtype=float)
    y = np.array(([5], [2], [7]), dtype=float)
    print(f"{y.shape=}")
    nn1 = NeuralNetwork(X=X, y=y, layer_sizes=[3, 4, 3, 2, 1])
    nn1.initialize_weights()
    y_hat = nn1.forward_propagation()
    print(f"{y_hat.shape=}")
    nn1.back_propagation(layer=5)
    print(f"{nn1.partial_derivatives=}")



if __name__ == '__main__':
    main()
