import math
import random
class Value:
    def __init__(self,data,_prev=(),op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_prev)  # Track parent nodes
        self.op = op            # Operation is create to record the operation in backpropagation 
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data,(self,other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad  
        out._backward = _backward          
               
        return out
    
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data,(self,other),'*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __sub__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data - other.data,(self,other),'-')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad 
        out._backward = _backward
            
        return out
    
    def __pow__(self,other):
        out = Value(self.data**other,(self,),f'**{other}')
        
        def _backward():
            assert isinstance(other, (int, float)), "only supporting int/float powers for now"
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out
    
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)  
        out = Value(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1- t**2) * out.grad    
        out._backward = _backward
        return out
    
    #for backpropagation
    def backward(self):
        topo = []
        visited = set()
        stack =[self]
        
        while stack:
            node = stack[-1]
            if node  in visited:
                #if node is visited, pop the node and put in topo
                stack.pop()
                topo.append(node)
                
                
            else :
                #Mark the node at visited , push all the parents to stack
                visited.add(node)
                for i in node._prev:
                    if i not in visited:
                        stack.append(i)
                        
        
        self.grad = 1.00
        for i in reversed(topo):
            i._backward()
    
    
    
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
    
    def __repr__(self):
        return f"Value(data={self.data},grad ={self.grad})"





# a = Value(2)
# b = Value(3)
# c = a+ b
# d = c**2
# # e=d.tanh()
# print(f'forwad : {d}')
# print(f'backward : {d.backward()}')
# print(f'd.grad: {d.grad}')
# print(f'a.grad: {a.grad}')
# print(f'b.grad: {b.grad}')
# print(f'c.grad: {c.grad}')
