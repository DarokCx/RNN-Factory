from .RNN import Model

from torch.utils.cpp_extension import load
import torch
import os
currentdir = os.path.dirname(os.path.realpath(__file__))

class v5cpp(Model):
    def __init__(self, args):
        self.model_name = 'v5-cpu'
        # load torch.cpp with march=native, O3, and -fopenmp
        openmp = "-fopenmp"
        # if on mac
        if os.uname().sysname == "Darwin":
            openmp = "-openmp"
             
        self.torch_cpp = load(name="torch_cpp", sources=[currentdir + "/rwkv.cpp", currentdir+"/rwkv.cuh/src/cpuops.cpp"], extra_cflags=["-O3", openmp, "-march=native", "-I"+currentdir+"/rwkv.cuh/include"], verbose=True)
        self.torch_cpp.init(args.load_model)
        modelinfo = self.torch_cpp.getModelInfo()
        print(modelinfo)
        #parse json
        import json
        modelinfo = json.loads(modelinfo)
        print(modelinfo)
        
        self.layers = modelinfo['layers']
        self.hidden = modelinfo['timeshift_state_size']
        self.vocab_size = pow(2, 16)
        self.wkv_state_size = modelinfo['wkv_state_size']
        self.batches = 1
        
        
        self.state = {
            
        }
        
        for i in range(self.layers):
            self.state[f'blocks.{i}.att.timeshift'] = torch.zeros(self.batches, self.hidden, dtype=torch.float32)
            self.state[f'blocks.{i}.ffn.timeshift'] = torch.zeros(self.batches, self.hidden, dtype=torch.float32)
            self.state[f'blocks.{i}.att'] = torch.zeros(self.batches, self.wkv_state_size, dtype=torch.float32)
        

        super(v5cpp, self).__init__()
        self.eval()

    def forward(self, idx, state=-1, **kwargs):

        if isinstance(idx, list):
            idx = torch.tensor(idx)
        # if idx is int, make tensor
        if isinstance(idx, int):
            idx = torch.tensor([idx])
        
        # if idx is not 3 dim tensor, make it 3 dim
        if len(idx.shape) == 1:
            idx = idx.unsqueeze(0)
            idx = idx.repeat(self.batches, 1)
            
        if state != -1 or state != None or state != {}:
            self.setState(state)

        output = self.torch_cpp.forward_cpu(idx)
        return output, self.getState()
    
    def resetState(self):
        self.torch_cpp.resetState()

    def getState(self):
        for i in self.state.keys():
           self.state[i] = self.torch_cpp.getStateKey(i)
                
        return self.state
        
    def setState(self, state):
        for i in state.keys():
            for (k) in range(state[i].shape[0]):
                self.torch_cpp.setStateKey(i,state[i][k],k)
    
        