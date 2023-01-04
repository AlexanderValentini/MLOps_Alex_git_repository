import torch
from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.drop_p = drop_p

        self.fc1 = nn.Linear(input_size, hidden_layers[0])
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.fc4 = nn.Linear(hidden_layers[2], hidden_layers[3])
        self.fc5 = nn.Linear(hidden_layers[3], hidden_layers[4])        
        self.fc6 = nn.Linear(hidden_layers[4], hidden_layers[5])
        
        self.output_layer = nn.Linear(hidden_layers[5],output_size)

        self.dropout = nn.Dropout(drop_p)

    def forward(self, x): 

        x = x.view(x.shape[0], -1)
        x = x.to(torch.float32)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        x = self.dropout(F.relu(self.fc6(x)))                                
        x = self.output_layer(x)
        
        return x