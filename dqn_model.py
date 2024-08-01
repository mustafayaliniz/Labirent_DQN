import torch
import torch.nn as nn
import torch.optim as optim

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size, device):
        super(DQNModel, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(state_size, 24).to(self.device)
        self.fc2 = nn.Linear(24, 24).to(self.device)
        self.fc3 = nn.Linear(24, action_size).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.device = device
        self.model = DQNModel(state_size, action_size, self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss().to(self.device)

    def predict(self, state):
        self.model.eval()
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.model(state)
        return q_values.cpu().numpy()

    def fit(self, state, target):
        self.model.train()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.optimizer.zero_grad()
        output = self.model(state)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()