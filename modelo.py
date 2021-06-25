import torch
import torch.nn as nn

class Red(nn.Module):
    def __init__(self, input_size, hidden_size,num_classes):
        super(Red, self).__init__()
        self.l1= nn.Linear(input_size,hidden_size)
        self.l2= nn.Linear(hidden_size,hidden_size)
        self.l3= nn.Linear(hidden_size,num_classes)
        self.activacion = nn.ReLU()
    def forward(self, x):

        out= self.l1(x)
        out= self.activacion(out)
        out= self.l2(x)
        out= self.activacion(out)
        out= self.l3(x)
        
        return out