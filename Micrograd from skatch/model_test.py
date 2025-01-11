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
learning_rate=.001
epochs = 70
for epoch in range(epochs):
    # Forward pass
    y_pred = [n(x) for x in xs]
    
    # Compute loss
    loss = sum((correct - error)**2 for error, correct in zip(y_pred, ygt))
    print(f"Epoch {epoch}, Loss: {loss.data}")
    # print(ygt)
    for i, (x, pred) in enumerate(zip(xs, y_pred)):
        print(f"Input {i}: {x}, Prediction: {pred.data}")
    # Backward pass
    loss.backward()
    
    # Update parameters
    for p in n.parameters():
        p.data += -learning_rate * p.grad


# loss = sum((correct - error)**2 for error,correct in zip(y_pred,ygt))
# print(f'loss:{loss}')
# loss.backward()
     
# for p in n.parameters():
#     p.data += 0.01 * p.grad
    
# # Print forward pass predictions
# print(ygt)
# for i, (x, pred) in enumerate(zip(xs, y_pred)):
#     print(f"Input {i}: {x}, Prediction: {pred.data}")
    

# # Inspect gradients for all parameters
# for layer in n.layers:
#     for neuron in layer.neurons:
#         for weight in neuron.weight:
#             print(f"Weight: {weight.data}, Grad: {weight.grad}")
#         print(f"Bias: {neuron.bias.data}, Grad: {neuron.bias.grad}")


    
