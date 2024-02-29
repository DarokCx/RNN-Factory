########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
from src.tokenizer import neox, world
tokenizer = world
import numpy as np
import types, time, gc
import torch
from src.samplers import sample_logits
args = types.SimpleNamespace()


# MODEL_NAME = "3B.pth"
# # MODEL_NAME = "/home/harrison/Documents/RNN-Factory/src/rwkv-raccoon-1b5.pth"
# args.load_model = MODEL_NAME
args.micro_bsz = 5
import inquirer
questions = [
            # choose between bfloat16/float32
            inquirer.List('size',
                message="How big do you want the model to be?",
                choices=['7B', '3B', '1.5B'],
            ),
            inquirer.List('type',
                message="Which do you want to benchmark?",
                choices=['Multiprocessing', 'State generation'],
            ),
            inquirer.List('stream increment',
                message="How many streams do you want to increment by?",
                choices=['1', '2', '4' ,'8', '16', '32', '64', '128', '256'],
            ),
]

answers = inquirer.prompt(questions)


# from src.models.modules.Linear import InferenceLinear, Quantized, Linear


from src.models import RWKV_v4, RWKV_v5, Experimental, v5cpp, v5simple
args.linear = torch.nn.Linear


args.vocab_size = 2**16

znargs = types.SimpleNamespace()
if answers['size'] == '3B':
    znargs.load_model = f"./3b.pth"

if answers['size'] == '7B':
    znargs.load_model = f"./7b.pth"

if answers['size'] == '1.5B':
   znargs.load_model = f"./1B5.pth"
    
# if answers['precision'] == 'uint8':
#     args.load_model="/home/harrison/CUDAMAX/Eagle7B_Q8.safetensors"
#     model = v5cpp(args)
# else:   

#     model = RWKV_v5(args)

model = v5simple(znargs)
# choose between cpu/gpu


TEMPERATURE = 0.9
top_p = 0.9

model = model.eval()
model = model.requires_grad_(False)
model = model.bfloat16()

print ("Memory use:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")

def init():

    
    return model

tokeqs = [1]


import gradio as gr


def runmodel(tokens, streams):
    

    toks = [tokeqs]*streams if answers['type'] == 'Multiprocessing' else [tokeqs * streams]
    st = model.new_state(streams)
    
    logits, state = model.forward(toks, state=st)
    
    newtokens = [[]]*streams if answers['type'] == 'Multiprocessing' else [[]]
    #warmup
    tokso = [torch.argmax(logits[j],dim=-1).item() for j in range(newtokens.__len__())]
    newtokens = [newtokens[j] + [tokso[j]] for j in range(newtokens.__len__())]
    logits, _ = model.forward(toks, state=state)
    
    timee = time.clock_gettime(0)
    import tqdm
    
    for i in tqdm.tqdm(range(tokens)):
        tokso = [torch.argmax(logits[j],dim=-1).item() for j in range(newtokens.__len__())]
        newtokens = [newtokens[j] + [tokso[j]] for j in range(newtokens.__len__())]
        logits, _ = model.forward(toks, state=state)
        
    otime = time.clock_gettime(0)-timee
    otime2 = (tokens*streams)/otime
    tps = tokens/otime
    # gc
    torch.cuda.empty_cache()
    del logits
    del toks
    del newtokens
    from gc import collect
    collect()
    return otime2, tps

samples = 11
increase = int(answers['stream increment'])
granularity = 20
  
from tqdm import tqdm
stats = [
    runmodel(granularity,int(1 if i == 0 else i*increase)) for i in tqdm(range(0,samples))
]

# fullTokens = sum([i[0] for i in stats])
# perStreamTokens = fullTokens/len(stats)

# display graph
import cpuinfo
import matplotlib.pyplot as plt
plt.plot([i[0] for i in stats])
plt.ylabel('Absolute tokens per second')
plt.xlabel("Concurrent streams/Simultaneous requests" if answers['type'] == 'Multiprocessing' else "Token Processing")
plt.title(f'''RWKV V5 {answers['type']} Benchmark\nDetails:{answers["size"]} ({answers["precision"]}) {device}\n Device: {
    torch.cuda.get_device_name(0) if device == "cuda" else cpuinfo.get_cpu_info()["brand_raw"]
}''')
plt.xticks(range(0,samples),[str(int(1 if i == 0 else i*increase)) for i in range(0,samples)])
plt.ylim(bottom=0)
# add subplot showing relative tokens per second
if answers['type'] == 'Multiprocessing':
    plt.twinx()
    plt.plot([i[1] for i in stats], color='red')
    plt.ylabel('Client facing tokens per second')
    plt.ylim(bottom=0)
    plt.tight_layout()

plt.savefig('benchmark.png')

# display table
import pandas as pd
df = pd.DataFrame(stats,
                  columns=['Absolute tokens per second', 'Client facing tokens per second'] if answers['type'] == 'Multiprocessing' else ['Total Tokens Procesed per second', "States of this size processable per second"],
                  index=[int(1 if i == 0 else i*increase) for i in range(0,samples)]
)
print(df)
df.to_csv('benchmark.csv')

# upload to wandb


