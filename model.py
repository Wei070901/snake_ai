import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size, num_actions):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output = nn.Linear(256, num_actions)
        
        # 初始化權重
        nn.init.xavier_uniform_(self.fc1[0].weight)
        nn.init.xavier_uniform_(self.fc2[0].weight)
        nn.init.xavier_uniform_(self.fc3[0].weight)
        nn.init.xavier_uniform_(self.output.weight)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.output(x)
