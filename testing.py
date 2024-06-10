import torch.nn as nn
import torch

m = nn.MaxPool1d(3, stride=2)
input = torch.randn(20, 16, 50)
output = m(input)
print(output[0],"\n\n",len(output))