# Multilayer Perceptron (MLP) Example

This directory contains a from-scratch implementation of a 2-layer Neural Network (one hidden layer) using the Sigmoid activation function and Backpropagation.

## Contents
- [mlp_analysis.ipynb](./mlp_analysis.ipynb): Demonstrates the network solving the **XOR problem**, a classic non-linearly separable challenge.

## Implementation Details
- **Architecture**: Input Layer $\rightarrow$ Hidden Layer $\rightarrow$ Output Layer.
- **Optimization**: Gradient Descent with Backpropagation.
- **Activation**: Sigmoid function used for both the hidden and output layers.
- **Weight Initialization**: Uniform random initialization to break symmetry.

## Performance
The model is capable of capturing non-linear decision boundaries, allowing it to correctly classify the XOR truth table with 100% accuracy after convergence.