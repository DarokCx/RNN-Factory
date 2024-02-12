# create statefull torch.nn module
import torch
import torch.nn as nn
class StateModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.initState(*args, **kwargs)
    def initState(self, *args, **kwargs):
        self.state = torch.zeros(*args, **kwargs)

    def setState(self, state):
        if not self.training:
            self.state = state.clone().detach()

    def getState(self):
        return self.state.clone()
    
    def resetState(self):
        self.initState(*self.state.shape)