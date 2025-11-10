# my-micrograd-project
A simple autograd engine (Micrograd) and neural network library built from scratch in Python.
# My Micrograd Project

This is a from-scratch implementation of a scalar-valued automatic differentiation (autograd) engine and a Multi-Layer Perceptron (MLP) library in Python.

This project is based on Andrej Karpathy's "Neural Networks from Scratch" series. The goal was to build an engine that understands backpropagation to gain a deep, foundational knowledge of how neural networks work.

## Core Components

* **/micrograd/engine.py**: Contains the `Value` class, which is the heart of the autograd engine. It tracks operations to build a computational graph and can compute gradients via backpropagation.
* **/micrograd/nn.py**: A simple neural network library (`Neuron`, `Layer`, `MLP`) built on top of the `Value` object.
* **[NAME-OF-YOUR-NOTEBOOK].ipynb**: A demo notebook that shows how to use the library to train an MLP on a small dataset.

## How to Use

The demo notebook `[NAME-OF-YOUR-NOTEBOOK].ipynb` shows a complete example of:
1.  Initializing an MLP.
2.  Performing a forward pass.
3.  Calculating the loss.
4.  Running the backward pass (`loss.backward()`).
5.  Updating the model's parameters.
