import random
from .engine import Value

class Neuron:
  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self, x):
    # w * x + b
    act = sum(wi*xi for wi, xi in zip(self.w,x)) + self.b
    out = act.tanh()
    return out
  
  def parameters(self):
    return self.w + [self.b]

class Layer:
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)] # Each neuron takes nin inputs
    '''
    nin = number of inputs to each neuron
    nout = number of neurons in the layer
    '''
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    '''
    Loop through each neuron n in the layer
    Call that neuron with input x -> n(x)
    Collect all outputs in a list outs
    '''
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    params = []
    for neurons in self.neurons:
      ps = neurons.parameters()
      params.extend(ps)
    return params

class MLP:
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x) # output of one layer becomes input to next
    return x
  
  def parameters(self):
    params = []
    for layer in self.layers:
      ps = layer.parameters()
      params.extend(ps)
    return params
