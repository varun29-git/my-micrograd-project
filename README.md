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

## 🚀 How to Run

The `demo_use.ipynb` notebook in this repo relies on the `micrograd` library folder. The most reliable way to run this is to clone the entire project.

### Method 1: Run Locally 

1.  Clone this repository to your computer using the "Code" button above.
2.  Make sure you have Jupyter Notebook or Jupyter Lab installed (`pip install jupyterlab`).
3.  In your terminal, navigate into the project's folder and run `jupyter lab`.
4.  Open `demo_use.ipynb`. All cells will run as-is.

### Method 2: Run in Google Colab

1.  Open a new, blank Google Colab notebook.
2.  In the first code cell, clone this repository. (You can get the URL from the "Code" button on this page).

    ```python
    !git clone [https://github.com/varun29-git/my-micrograd-project.git](https://github.com/varun29-git/my-micrograd-project.git)
    ```

3.  In a second code cell, add the project folder to Python's path so it can find the library.

    ```python
    import sys
    sys.path.append('/content/my-micrograd-project')
    
    print(" Library path added. You can now run the demo.")
    ```

4.  Upload the `demo_use.ipynb` file (which you've downloaded from this repo) to your Colab environment. Once uploaded, you can open it and run all the cells.
