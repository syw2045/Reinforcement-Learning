import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3_adv = nn.Linear(n_mid, n_out) # Advantage function
        self.fc3_v = nn.Linear(n_mid, 1) # Value function

    def forward(self,x ):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        adv = self.fc3_adv(h2)
        val = self.fc3_v(h2).expand(-1,adv.size(1))

        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return output
