import numpy as np 
import matplotlib.pyplot as  plt
from Engine import Value 
import  random


class Neuron:
    def __init__(self,n_in):
        self.weight = [Value(random.uniform(-0.1, 0.1)) for _ in range(n_in)]
        self.bias = Value(random.uniform(-0.1,0.1))
        
    def __call__(self,x):
        # ( W * n_in ) + b
        act = sum((w * xi for w, xi in zip(self.weight, x)), self.bias)
        return act.tanh()

    def parameters(self):
        return self.weight +[self.bias]
    
class Layer:
    def __init__(self,n_in,n_nn):
        self.neurons = [Neuron(n_in) for _ in range(n_nn)]
    
    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        params = []
        for i in self.neurons:
            n_p = i.parameters()
            params.extend(n_p)
        return params
    

class MLP():

    def __init__(self, n_in, li_nn):
        sz = [n_in] + li_nn
        self.layers = [Layer(sz[i], sz[i+1])for i in range(len(li_nn))]

    def __call__(self, x):
        for layer in self.layers:
            final = layer(x)
        return final
    
    def parameters(self):
        parms =[]
        for i in self.layers:
            ps = i.parameters()
            parms.extend(ps)
        return parms




xs=[
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0]    
]
n = MLP(3,[4,4,1])
ygt = [1.0,-1.0,-1.0,1.0] # desire target
# ygt = [Value(target) for target in [1.0, -1.0, -1.0, 1.0]]#converto to Value 



y_pred = [n(x) for x in xs]
# print(f'forwad : {y_pred}')


loss = sum((correct - error)**2 for error,correct in zip(y_pred,ygt))
loss.backward()






# Inspect gradients for all parameters
for i, param in enumerate(n.parameters()):
    print(f"Parameter {i}: {param}, Gradient: {param.grad}")




print(f'loss :{loss}')
# print(len(n.parameters()))
# print(n.layers[-1].neurons[-1].weight[-1].grad)
# print(n.layers[0].neurons[0].weight[0].grad)
# print(n.layers[0].neurons[0].bias)

