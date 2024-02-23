########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import types, time, gc
import torch
from extract import extract_column
import pandas as pd


########################################################################################################
# Step 1: set model & config (use v4 to run your trained-from-scratch models. v4 and v4neo are compatible)
########################################################################################################


dataset_name = 'MBZUAI/LaMini-instruction'
split = 'train'  # Choose the dataset split, e.g., 'train', 'test', etc.
column_name = 'response'  # Specify the column you want to extract


# MODEL_NAME = '/home/harrison/Documents/RNN-Factory/src/training/pipeline/models/5.pth'

from src.samplers import sample_logits
from src.models.modules.Linear import InferenceLinear

from src.models import RWKV_v4, RWKV_v5, Experimental
args = types.SimpleNamespace()
args.linear = InferenceLinear
args.load_model = '7B.pth'

model = RWKV_v5(args).cuda()


from src.tokenizer import world #neox, world, racoon
tokenizer = world

context = '''
### Instruction:
Please translate the next sentence into French.
### Sentence:

'''

doGreedy = True


NUM_TRIALS = 1
LENGTH_PER_TRIAL = 1000

TEMPERATURE = 0.9
top_p = 0.9
top_p_newline = 0.9  # only used in TOKEN_MODE = char

DEBUG_DEBUG = False  # True False --> show softmax output

########################################################################################################




# get model memory use
print("Memory use:", torch.cuda.memory_allocated() / 1024 ** 3, "GB")





testdata = torch.randint(0, 100, (1,5))
model.resetState()
atonce = model.forward(testdata)
print(f'At once:', atonce.shape)

model.resetState()
# model = model.cuda()
for i in range(len(testdata[0])):
    atatime = model.forward(testdata[:,i:i+1])
    error = torch.max(torch.abs(atonce[0,i] - atatime)).item()
    # 3 decimal places
    error = int(error * 1000)
    print(f'[{i}]', error / 1000, 'max error')

########################################################################################################

# if tokenizer.charMode:
#     context = tokenizer.refine_context(context)
#     ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context]
# else:
data = extract_column(dataset_name, split, column_name)

batchsize = 125
batches = 1000
instructions = []
ctext = []
translation = []

dict = {'Instruction': instructions, 'Input': ctext, 'Response': translation}

with open("output.txt", "w") as f:
    for step in range(0, batches*batchsize, batchsize):
        
        context_len_origin = [tokenizer.encode(context+data[i]+"\n### Response:\n").__len__() for i in range(step, step+batchsize)]
        ctext = context
        # Example: Print the first 10 entries
        ctx = [tokenizer.encode(context+data[i]+"\n### Response:\n") for i in range(step, step+batchsize)]
        fctx = [[]]*batchsize
        ##### extract data/

        src_len = len(ctx)
        src_ctx = ctx.copy()

        ## add context to data
        instructions = [context]*src_len
        dict['Instruction'].extend(instructions)
        dict['Input'].extend(data[step:step+batchsize])

        print("\nYour prompt has " + str(src_len) + " tokens.")
        print(
            "Note: currently the first run takes a while if your prompt is long, as we are using RNN to preprocess the prompt. Use GPT to build the hidden state for better speed.\n"
        )

        time_slot = {}
        time_ref = time.time_ns()

        def record_time(name):
            if name not in time_slot:
                time_slot[name] = 1e20
            tt = (time.time_ns() - time_ref) / 1e9
            if tt < time_slot[name]:
                time_slot[name] = tt

        init_state = None
        init_out = None
        state = None
        out = None

        for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
            
            # print(("-" * 50) + '\n' + context, end="")

            time_ref = time.time_ns()
            # ctx = data[i] #src_ctx.copy()

            if TRIAL == 0:
                
                gc.collect()
                torch.cuda.empty_cache()

            record_time('preprocess')
            out_last = src_len
            for i in range(src_len, src_len + (1 if DEBUG_DEBUG else LENGTH_PER_TRIAL)):
                x = [ctx[o][-1:] for o in range(len(ctx))]
                

                if i == src_len:
                    model.resetState()

                    states = [model.forward([ctx[o]], None) for o in range(len(ctx))]
                    # print(states.__len__())
                    keys = states[0][1].keys()
                    for key in keys:
                        states[0][1][key] = torch.cat([states[o][1][key] for o in range(len(ctx))], dim=0)
                    
                    model.setState(states[0][1])
                    logits = [states[o][0] for o in range(len(ctx))]
                    out = torch.stack(logits, dim=0).reshape(len(ctx), -1)
                
                
                else:
                    out = model.forward(x)
                if DEBUG_DEBUG:
                    print("model", np.array(x), "==>", np.array(out), np.max(out.cpu().numpy()), np.min(out.cpu().numpy()))
                # if TOKEN_MODE == "pile":
                # out[0] = -99  # disable <|endoftext|>
                if doGreedy:
                    toks = torch.argmax(out, dim=-1)
                    ctx = [ctx[o] + [toks[o].item()] for o in range(len(toks))]
                    # print(len(toks))
                    # print(fctx)
                    fctx = [fctx[o] + [toks[o].item()] for o in range(len(toks))]
                else:
                    ttt = sample_logits(
                        out, temperature=TEMPERATURE, top_p=top_p
                    )
                
            

                # if tokenizer.charMode:
                #     char = tokenizer.itos[ttt]
                #     print(char, end="", flush=True)
                # else:
                    

            record_time('total')
            # print(f'\n\n{time_slot}\n\n')
            print(
                f"\n\n--- preprocess {round(time_slot['preprocess'], 2)}s, generation {round(time_slot['total']-time_slot['preprocess'], 2)}s ", end = ''
            )
        for o in range(len(ctx)):
            for z in range(len(ctx[o])):
                if ctx[o][z] == 0:
                    ctx[o] = ctx[o][o:z]
                    break
        for o in range(len(fctx)):
            for z in range(len(fctx[o])):
                if fctx[o][z] == 0:
                    fctx[o] = fctx[o][:z]
                    break
        
        
        
        dict['Response'].extend([tokenizer.decode(fctx[o]) for o in range(len(fctx))])

ctx = [tokenizer.decode(ctx[o]) for o in range(len(ctx))]
# append to file
stats = [[dict['Instruction'][i], dict['Input'][i], dict['Response'][i]] for i in range(len(dict['Instruction']))]
df = pd.DataFrame(stats,
        columns=['Instruction', 'Input', 'Response']
        )
# print(df)
df.to_csv('test.csv')
        # f.write('\n'.join(ctx))
            

print(("-" * 50) + '\n')
