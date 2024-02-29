### ---
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
### ---

global RWKV_JIT_ON, RWKV_TORCH_COMPILE, RWKV_NO_CUDA

from .module.CoreDependencies import *
from .module.ChannelMix import RWKV_ChannelMix
from .module.TimeMix import RWKV_TimeMix

# ---
# Isolating out known operations that **does not work** with torch.compile
# and wrapping them within a torch._dynamo.disable, this is required to get
# the baseline torc.compile to work
# ---

# In the latest version of deepspeed + torch compile,
# deepspeed.checkpointing now works ? - this is inconsistent, so i am disabling for now


### ---
# RWKV: State Blocks
### ---


### ---
# The RWKV Model blocks
### ---

class Block(nn.Module):

    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att, dim_ffn):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = RWKV_TimeMix(layer_id, n_layer, n_embd, n_head, head_size, dim_att)
        self.ffn = RWKV_ChannelMix(layer_id, n_layer, n_embd, dim_ffn)

        # Setup droupout at block level

    @TCompileBaseline
    def forward(self, x, time_mix_shift, channel_mix_state, time_mix_state):

        att_out, att_shift, att_state = self.att(
            self.ln1(x),
            time_mix_shift,
            time_mix_state
        )

       
            # Handle without dropout
        x = x + att_out
        ffn_out, ffn_state = self.ffn(
            self.ln2(x),
            channel_mix_state,
        )
        x = x + ffn_out
        
        return x, att_shift, ffn_state, att_state

class FirstBlock(nn.Module):
        def __init__(self, n_layer, n_embd, n_head, head_size, dim_att, dim_ffn):
            super().__init__(  )
    
            self.ln0 = nn.LayerNorm(n_embd)
            
            self.layer_id = 0

            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

            self.att = RWKV_TimeMix(0, n_layer, n_embd, n_head, head_size, dim_att)
            self.ffn = RWKV_ChannelMix(0, n_layer, n_embd, dim_ffn)

    
        @TCompileBaseline
        def forward(self, x,  time_mix_shift , channel_mix_state, time_mix_state):
            
            x = self.ln0(x)

            att_out, att_shift, att_state = self.att(
                self.ln1(x),
                time_mix_shift,
                time_mix_state
            )

        
                # Handle without dropout
            x = x + att_out
            ffn_out, ffn_state = self.ffn(
                self.ln2(x),
                channel_mix_state,
            )
            x = x + ffn_out
            
            return x, att_shift, ffn_state, att_state
            
### ---
# Core RWKV module
### ---
class RWKV(nn.Module):

    def __init__(self,
                 # Model file path to load from
                 load_model: str,
                 # Model size settings, which we either
             
                 ):

 
        # Setup the parent class
        super().__init__()
           
        try:
            self.batches = micro_bsz
        except:
            self.batches = 1
            micro_bsz = 1

        try:
            grad_cp
        except:
            grad_cp = 0

        try:
            ctx_len
        except:
            ctx_len = 1024

        try:
            modelpath = load_model

        except:
            modelpath = None
        
        if modelpath:
            file = torch.load(modelpath, map_location="cpu")
            keys = list(file.keys())
            print("keys", keys)
            # remove _orig_mod from keys for compatibility with torch.compile
            newObj = {}
            for key in keys:
                if "_orig_mod." in key:
                    newKey = key.replace("_orig_mod.", "")
                    newObj[newKey] = file[key]
                else:
                    newObj[key] = file[key]
            file = newObj
            keys = list(file.keys())

            # detect model details
            vocab_size, n_embd = file["emb.weight"].shape
            n_embd = n_embd
            vocab_size = vocab_size
            n_layer = 0
            for key in keys:
                if key.startswith("blocks."):
                    layer = int(key.split(".")[1])
                    if layer > n_layer:
                        n_layer = layer
            n_layer = n_layer + 1
            print("n_layer", n_layer)
            # try:
            dim_ffn = file[f"blocks.0.ffn.value.weight"].shape[1]
            # except:
            #     dim_ffn = 2 * n_embd
            # model layers are model.2.x.yyy: find highest x
            
            try:
                n_head = file[f"blocks.0.att.time_decay"].shape[0]
                print("n_head", n_head)
            except:
                n_head = 64
           
        else:
            file = None

        try:
            dim_ffn = dim_ffn
        except:
            dim_ffn = int(3.5 * n_embd)
            
        self.emb = nn.Embedding(vocab_size, n_embd)
        
        self.n_embd = n_embd
        
        self.n_layer = n_layer
        
        self.n_head = n_head
        
        self.head_size = n_embd // n_head
        
        self.dim_ffn = dim_ffn
        
        
        self.blocks = nn.ModuleList([
            FirstBlock(n_layer, n_embd, n_head, self.head_size, n_embd, dim_ffn)
            ,*[
            Block(i, n_layer, n_embd, n_head, self.head_size, n_embd, dim_ffn) for i in range(1,n_layer)
        ]])
        
        self.ln_out = nn.LayerNorm(n_embd)
        
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.load_state_dict(file)
        
        
            

    # def new_state(self, B):
    #     return BlockStateList.create(
    #             self.n_layer, B, self.n_embd, 
    #             self.n_head, self.head_size,
    #             self.emb.weight.device, self.emb.weight.dtype
    #         )
        
    # @TCompileBaseline
    def forward(self, idx: torch.Tensor, last_shift_states: torch.Tensor = None,
                last_wkv_states: torch.Tensor = None):
        x = self.emb(idx)



        new_shift_states = torch.zeros_like(last_shift_states)
        new_wkv_states = torch.zeros_like(last_wkv_states)
        
        # last_shift_states can be None, when we are performing direct inference
     

        ## The output X token
        

        for i,b in enumerate(self.blocks):
            # print("last_state", cur_bs_list.shift_states.)
            x, new_wkv_states[:,i,0],new_shift_states[:,i,1], new_shift_states[:,i]  = b(x, last_shift_states[:,i,0],last_shift_states[:,i,1], last_wkv_states[:,i])
           


        x = self.ln_out(x)
        x = self.head(x)

        return x, last_shift_states, last_wkv_states

