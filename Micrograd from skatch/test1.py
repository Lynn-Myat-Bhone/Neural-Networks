import numpy as np 
import matplotlib.pyplot as  plt
from Engine import Value 
import  random


class Neuron:
    def __init__(self,n_in):
        self.weight = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.bias = Value(random.uniform(-1,1))
        
    def __call__(self,x):
        # ( W * n_in ) + b
        act = sum((w * xi for w, xi in zip(self.weight, x)), self.bias)
        return act.tanh()

class Layer:
    def __init__(self,n_in,n_nn):
        self.neurons = [Neuron(n_in) for _ in range(n_nn)]
    
    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    

class MLP():

    def __init__(self, n_in, li_nn):
        sz = [n_in] + li_nn
        self.layers = [Layer(sz[i], sz[i+1])for i in range(len(li_nn))]

    def __call__(self, x):
        for layer in self.layers:
            final = layer(x)
        return final




xs=[
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0]    
]
ygt = [1.0,-1.0,-1.0,1.0] # desire target
ygt = [Value(target) for target in [1.0, -1.0, -1.0, 1.0]]#converto to Value 


n = MLP(3,[4,4,1])
y_pred = [n(x) for x in xs]
print(f'forwad : {y_pred}')


loss = Value(sum((correct - error)**2 for error,correct in zip(y_pred,ygt)))
print(f'loss :{loss}')

loss.backward()
print(n.layers[-1].neurons[-1].weight[-1].grad)