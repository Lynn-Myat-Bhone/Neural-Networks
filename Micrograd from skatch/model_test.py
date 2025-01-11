import numpy as np 
import matplotlib.pyplot as  plt
from Engine import Value,MLP,Neuron,Layer 
import  random

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

n = MLP(3,[2,3,1])

ygt = [1.0,-1.0,-1.0,1.0]  # Targets



y_pred = [n(x) for x in xs]
learning_rate = 0.01
epochs = 100
for epoch in range(epochs):
    # Forward pass
    y_pred = [n(x) for x in xs]
    
    # Compute loss
    loss = sum((correct - error)**2 for error, correct in zip(y_pred, ygt))
    print(f"Epoch {epoch}, Loss: {loss.data}")
    
    for p in n.parameters():
        p.grad = 0.0
        
    for i, (x, pred) in enumerate(zip(xs, y_pred)):
        print(f"Input {i}: {x}, Prediction: {pred.data}")
    # Backward pass
    loss.backward()
    
    # Update parameters
    for p in n.parameters():
        p.data += -learning_rate * p.grad




    
