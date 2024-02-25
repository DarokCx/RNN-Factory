from .RNN import Model

from torch.utils.cpp_extension import load
import torch
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
from .simple.model import RWKV

from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch_neuronx
# import torch.ao.quantization.quantize_fx as quantize_fx
# import torch_tensorrt

class v5simple( Model):
    def __init__(self, args):
        self.model_name = 'v5-simple'
        
        super(v5simple, self).__init__()
        self.model = RWKV(load_model=args.load_model)
        self.model = self.model.eval()
        self.dtype = torch.bfloat16
        self.model = self.model.to(self.dtype)
        self.layers = self.model.n_layer
        self.hidden = self.model.n_embd
        self.head_size = self.model.head_size
        self.heads = self.model.n_head
        
        self.eval()
        self.requires_grad_(False)
        self.device = torch.device("cpu")
        # self.model = torch.jit.script(self.model)
        
        # from torch.quantization import quantize_dynamic
        # self.model = quantize_dynamic(
        #     model=self.model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8, inplace=False
        # )
        
        # self.model = torch_tensorrt.compile(self.model, "default", (
        #     torch.tensor([[1]]).cuda(),
        #     *self.new_state(1)
        # ))            
        self.model = self.model.to(self.device)
        # self.model = torch.jit.script(self.model)
        # 
        # self.cpum = RWKV(load_model=args.load_model).cpu().bfloat16().eval()
        
        partitioner_config = torch_neuronx.PartitionerConfig()
        self.model = torch_neuronx.trace(self.model, (torch.tensor([[1]]),*self.new_state(1)), partitioner_config=partitioner_config)
        self.model = torch_neuronx.dynamic_batch(self.model)
        
        
        
        
        self.setState(self.new_state(1))
        
        

    def forward(self, idx, state=-1, **kwargs):

        if isinstance(idx, list):
            idx = torch.tensor(idx, device=self.device)
        # if idx is int, make tensor
        if isinstance(idx, int):
            idx = torch.tensor([idx], device=self.device)
        
        # if idx is not 3 dim tensor, make it 3 dim
        if len(idx.shape) == 1:
            idx = idx.unsqueeze(0)
            idx = idx.repeat(1, 1)
            
        if state != -1 and state != None and state != {}:
            self.setState(state)
        else:
            state = self.new_state(idx.shape[0])
            
        b,t = idx.shape
        
        # if(b == 1 and t > 1):
        #     output, outsstates, outwkvstates = self.cpum.forward(idx.cpu(), self.state[0].cpu(), self.state[1].cpu())
        
        # else:
            
        # idx = idx.to(self.device)
            
        # print(idx.device)
        # print(self.state[0].device)
        # print(self.state[1].device)
        # print(self.model)
        # print(self.state[1].shape)
        # print(self.state.__len__())
        output, outsstates, outwkvstates = self.model.forward(idx, self.state[0].to(self.device, self.dtype), self.state[1].to(self.device, self.dtype))
    
        self.setState((outsstates, outwkvstates))
        
        return output, self.getState()
    
    def resetState(self, B = -1):
        # self.state = self.new_state(self.state.shift_states.shape[2])
        if B == -1:
            B = self.state[0].shape[2]
        self.state = self.new_state(B)
        
        
    def new_state(self, B):
        return (
            torch.zeros(self.layers, 2, B, self.hidden, dtype=self.dtype, device=self.device),
            torch.zeros(self.layers, B,self.heads, self.head_size, self.head_size, dtype=self.dtype, device=self.device)
        )
    
    def newState(self, B):
        return self.new_state(B)
    
    def cuda(self):
        # super(v5simple, self).cuda()
        self.device = torch.device("cuda")
        self.state = (self.state[0].cuda(), self.state[1].cuda())
        self.model = self.model.cuda()
        return self
    
    def cpu(self):
        super(v5simple, self).cpu()
        self.state = (self.state[0].cpu(), self.state[1].cpu())
        self.device = torch.device("cpu")
        self.model = self.model.cpu()
        return self

    def getState(self):
        return self.state
        
    def setState(self, state):
        self.state = state
        
    def half(self):
        self.dtype = torch.float16
        self.model = self.model.half()
        self.state = (self.state[0].half(), self.state[1].half())
        return self
    
        