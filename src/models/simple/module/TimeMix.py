# Dependencies
from .CoreDependencies import *
from .OptimizedOps import modified_lerp

class fillGroupNorm(nn.Module):
    def __init__(self, n_head, dim_att, eps=1e-5):
        super().__init__()
        self.n_head = n_head
        self.dim_att = dim_att
        self.dims = dim_att // n_head
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim_att))
        self.bias = nn.Parameter(torch.zeros(dim_att))

    def forward(self, x):
        x = x.reshape(-1, self.n_head, self.dims)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / (std + self.eps)
        x = x.reshape(-1, self.dim_att)
        x = x * self.weight + self.bias
        
        return x

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
        self.ln_x = fillGroupNorm(n_head, dim_att, eps=64e-5)

        self.chunk_len = chunk_len
        self.precision = precision
        
        self.silu = nn.SiLU()
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        a = super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        
        self.time_decay = nn.Parameter(torch.exp(-torch.exp(self.time_decay.float())).view(1,self.n_head,1,-1))
        self.time_faaaa = nn.Parameter(self.time_faaaa.view(1,self.n_head,1,-1))
        return a
        
        

    def forward(self, x, last_state_shift, last_state_wkv):
        shift_state_out = x[:,-1]

        # assert x.size(-2) % self.chunk_len == 0, "fast non-cuda rwkv5.2+ requires ctxlen to be an exact multiple of chunk_len"

        # Get the x sizing
        B, T, C = x.shape
        H = last_state_wkv.shape[-3]
        K = last_state_wkv.shape[-2]
        V = K

        # Perform the tokenshift, and get the respective state
        output = torch.concat((last_state_shift.unsqueeze(1), x[:, :-1]), dim=1)

        # Get the xk, xv, xr, xg, and rkvg
        xk = modified_lerp(x, self.time_mix_k, output)
        xv = modified_lerp(x, self.time_mix_v, output)
        xr = modified_lerp(x, self.time_mix_r, output)
        xg = modified_lerp(x, self.time_mix_g, output)

        
        g = self.silu(self.gate(xg))
        
        
        #torch.zeros(B, T, H, V, dtype=torch.bfloat16, device=x.device)
        
        r = self.receptance(xr).view(B,T,H,-1,1) # BTH1Z
        k = self.key(xk) .view(B,T,H,1,-1)       # BTH1Z
        v = self.value(xv) .view(B,T,H,-1,1)     # BTHZ1
        
        kv = v @ k # BTHZ1 @ BTH1Z = BTHZZ
         
        for t in range(T):
                
                # reuse output as a buffer
                output[:,t] =  ((last_state_wkv + self.time_faaaa * kv[:,t]) @ r[:,t]).view(B,C)

                last_state_wkv = last_state_wkv * self.time_decay + kv[:,t]
                        
       

        # Reshape and normalize the logits
        x_logits = self.ln_x(output).view(B, T, C)
        x_logits = self.output(x_logits * g)

        # Return the logits and the state
        return (x_logits, shift_state_out, last_state_wkv)
    


# @TCompileMax
# @JITFunction
# def x_logits_output_parsing(out_emb, head_size_divisor, B, TT, C, self_ln_x, self_output, g):
#     x_logits = out_emb.view(-1, C)
#     x_logits = self_ln_x(x_logits / head_size_divisor).view(B, TT, C)
#     return self_output(x_logits * g)
