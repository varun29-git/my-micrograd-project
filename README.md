# my-micrograd-project

A simple autograd engine (Micrograd) and neural network library built from scratch in Python.

## What this is

This is my implementation of a scalar-valued autograd engine and a Multi-Layer Perceptron (MLP) library.

I built this by following Andrej Karpathy's "Neural Networks from Scratch" series. The goal was to stop *using* libraries like PyTorch as a black box and actually understand how backpropagation works under the hood.

## What's inside

* `/micrograd`: This is the library itself.
    * `engine.py`: Has the `Value` class, which is the core of the autograd engine.
    * `nn.py`: Has the `Neuron`, `Layer`, and `MLP` classes.
* `demo_use.ipynb`: A simple demo notebook that imports the `micrograd` library and trains a small MLP on a toy dataset.

## 🚀 How to Run

Click the link below to run the project in Google Colab.

**[Run the `demo_use.ipynb` notebook in Colab](https://colab.research.google.com/github/varun29-git/my-micrograd-project/blob/main/demo_use.ipynb)**
