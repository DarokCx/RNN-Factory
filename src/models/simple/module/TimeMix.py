# Dependencies
from .CoreDependencies import *
from .OptimizedOps import modified_lerp
from .rwkv_inner import rwkv_inner
import os
# try:
import torch_neuronx
try:
    custom_op.load_library('libwkv5.so')
except:
    from torch_neuronx.xla_impl import custom_op


    custom_op.load(
        name="wkv5",
        compute_srcs=['./src/models/simple/module/justaws.cpp'],
        shape_srcs=['./src/models/simple/module/justawsshape.cpp'],
        multicore=False,
        verbose=True,
        build_directory=os.getcwd(),
    )
    
    custom_op.load_library('libwkv5.so')
# except:
#     from torch.utils.cpp_extension import load
#     wkv5_cuda = load(name="wkv5", sources=["./src/models/simple/module/customawsoperator.cpp"],
                                # verbose=True, extra_cflags=["-O3", "-march=native", "-fPIC"])


# RWKV TimeMix module
class RWKV_TimeMix(torch.nn.Module):
    #chunk_len:int = 128, precision:int = 64
    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att, chunk_len:int = 1, precision:int = 64):
        super().__init__()
        
        self.dim_att = dim_att
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.layer_id = layer_id

        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = 8

        # V5-R4 changes
        # https://github.com/BlinkDL/RWKV-LM/commit/5aab658f945ba80745d36c2ab411fb43df3a74f9    
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(dim_att)
            for n in range(dim_att):
                decay_speed[n] = -6 + 5 * (n / (dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(dim_att)
            for n in range(dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(n_embd, dim_att, bias=False)
        self.key = nn.Linear(n_embd, dim_att, bias=False)

        self.value = nn.Linear(n_embd, dim_att, bias=False)
        self.output = nn.Linear(dim_att, n_embd, bias=False)
        self.gate = nn.Linear(n_embd, dim_att, bias=False)
        self.ln_x = nn.GroupNorm(n_head, dim_att, eps=64e-5)

        self.chunk_len = chunk_len
        self.precision = precision
        
        self.silu = nn.SiLU()
        
        

    def forward(self, x, last_state_shift, last_state_wkv):
        shift_state_out = x[:,-1]

        # assert x.size(-2) % self.chunk_len == 0, "fast non-cuda rwkv5.2+ requires ctxlen to be an exact multiple of chunk_len"

        # Get the x sizing
        B, T, C = x.shape
        H = last_state_wkv.shape[-3]
        K = last_state_wkv.shape[-2]
        V = K

        # Perform the tokenshift, and get the respective state
        xx = torch.concat((last_state_shift.unsqueeze(1), x[:, :-1]), dim=1)

        # Get the xk, xv, xr, xg, and rkvg
        xk = modified_lerp(x, self.time_mix_k, xx)
        xv = modified_lerp(x, self.time_mix_v, xx)
        xr = modified_lerp(x, self.time_mix_r, xx)
        xg = modified_lerp(x, self.time_mix_g, xx)

        r = self.receptance(xr).reshape(B,T,H,-1) # BHTK
        k = self.key(xk) .reshape(B,T,H,-1)     # BHTK
        v = self.value(xv) .reshape(B,T,H,-1)   # BHTV
        g = self.silu(self.gate(xg))

        w = torch.exp(-torch.exp(self.time_decay.float())).view(H,-1)

        u = self.time_faaaa.float().view(H,-1)

        # Logits and state
        wkv_state = last_state_wkv.float()

        
        rm = r.contiguous()
        km = k.contiguous()
        vm = v.contiguous()
        
        out = torch.ops.TimeMix.forward_cpu(wkv_state, rm.float(), km.float(), vm.float(), w, u)
                    
        x_logits =  out[:,:,:T].contiguous().transpose(1,2).reshape(B, T, C).bfloat16()

        # Reshape and normalize the logits
        x_logits = x_logits.view(-1, C)
        x_logits = self.ln_x(x_logits).view(B, T, C)
        x_logits = self.output(x_logits * g)

        # Return the logits and the state
        return (x_logits, shift_state_out, out[:,:,T:].contiguous())
    

def compute_wkv_state(
        k, v, r,
        time_faaaa: torch.nn.Parameter,
        time_decay: torch.nn.Parameter,
        wkv_state, 
        n_head:int, head_size:int,
        B:int, TT:int
    ):
    # Compute attent and the initial output tensor
    at = k @ v
    u = time_faaaa.view(1,1,n_head, 1, -1)

    # Slightly inefficent, but it works, lets compute all the tokens
    w = time_decay.exp().neg().exp().reshape(1, n_head,-1,1)

    out = (u * r) @ at
    for t in range(TT):
        out[:,t] += r[:,t] @ wkv_state
        
        # We make a clone copy, so the previous object backprop state is tracked seperately
        wkv_state = wkv_state.clone()
        wkv_state *= w
        wkv_state += at[:,t]

    return wkv_state, out


# @TCompileMax
# @JITFunction
# def x_logits_output_parsing(out_emb, head_size_divisor, B, TT, C, self_ln_x, self_output, g):
#     x_logits = out_emb.view(-1, C)
#     x_logits = self_ln_x(x_logits / head_size_divisor).view(B, TT, C)
#     return self_output(x_logits * g)
