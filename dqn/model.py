import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict

# Q-Network
class Model(nn.Module):
    def __init__(self, nb_actions):
        super(Model, self).__init__()

        self.nb_actions = nb_actions

        self.conv_layers = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)),
          ('relu2', nn.ReLU()),
          ('conv3', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)),
          ('relu3', nn.ReLU()),
        ]))

        self.fc_layers = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2304, 256)),
            ('last_relu', nn.ReLU()),
            ('fc2', nn.Linear(256, nb_actions))
        ]))

    def forward(self, x):
        batch_size = x.size()[0]

        x = self.conv_layers(x)

        x = x.view(batch_size, -1)
        outs = self.fc_layers(x)

        return outs

    def pick_action(self, obs):
        outs = self.forward(obs).squeeze().cpu().data.numpy()
        action = int(np.argmax(outs))
        return action
