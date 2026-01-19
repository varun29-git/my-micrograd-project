# my-micrograd-project

A simple autograd engine and neural network library built from scratch in Python.

## What this is

This is my implementation of a scalar-valued autograd engine and a Multi-Layer Perceptron (MLP).

The goal was to apply **first-principles thinking** to understand how neural networks really work. By building the core components from scratch, this project breaks down backpropagation and computational graphs into simple, understandable pieces.

## What's inside

* `/micrograd`: This is the library itself.
    * `engine.py`: Has the `Value` class, the core of the autograd.
    * `nn.py`: Has the `Neuron`, `Layer`, and `MLP` classes.
* `demo_use.ipynb`: A simple demo notebook that imports and uses the `micrograd` library to train a small neural network.


