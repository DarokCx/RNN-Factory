import copy
import os, gc, torch
import time
import types
#from huggingface_hub import hf_hub_download
from pynvml import *
from torch.nn import functional as F
import numpy as np
import json
# nvmlInit()
# gpu_h = nvmlDeviceGetHandleByIndex(0)
ctx_limit = 8192

#os.environ["CUDA_VISIBLE_DEVICES"] = ''
os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # if '1' then use CUDA kernel for seq mode (much faster)

torch.set_num_threads(4)

from src.samplers import sample_logits

from src.models import RWKV_v4, RWKV_v5, Experimental
args = types.SimpleNamespace()
args.load_model = '7B.pth'
model = RWKV_v5(args)
args.withPipeline = True
models = [
    model,
]

pipelines = []

from rwkv.utils import PIPELINE, PIPELINE_ARGS
from src.tokenizer import neox, world, racoon

lockedModels = []

for model in models:
    pipelines.append(racoon)
    lockedModels.append(False)

# set thread count with pytorch

import asyncio

async def getModel():
    while True:
        for i in range(len(models)):
            if not lockedModels[i]:
                #lockedModels[i] = True
                return models[i], pipelines[i], i
        await asyncio.sleep(0.1)
        
def unlockModel(i):
    lockedModels[i] = False

def sample_logits_typical(logits, temperature=1.0, top_p=0.95, **kwargs):
        probs = F.softmax(logits.float(), dim=-1)
        logits = -torch.log(probs)
        ent = torch.nansum(logits * probs, dim=-1, keepdim=True)
        shifted_logits = torch.abs(logits - ent)
        sorted_ids = torch.argsort(shifted_logits)
        sorted_logits = shifted_logits[sorted_ids]
        sorted_probs = probs[sorted_ids]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = np.sum(cumulative_probs < top_p)
        probs[shifted_logits > sorted_logits[cutoff]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)

def sample_logits(logits, temperature=1.0, top_p=0.85, top_k=0):
    probs = F.softmax(logits.float(), dim=-1)
    top_k = int(top_k)
    if probs.device == torch.device('cpu'):
        probs = probs.numpy()
        sorted_ids = np.argsort(probs)
        sorted_probs = probs[sorted_ids][::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        probs = probs / np.sum(probs)
        out = np.random.choice(a=len(probs), p=probs)
        return int(out)
    else:
        sorted_ids = torch.argsort(probs)
        sorted_probs = probs[sorted_ids]
        sorted_probs = torch.flip(sorted_probs, dims=(0,))
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs >= top_p)])
        probs[probs < cutoff] = 0
        if top_k < len(probs) and top_k > 0:
            probs[sorted_ids[:-top_k]] = 0
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)
        
async def evaluate(
    prompt,
    model,
    pipeline,
    token_count=1024,
    temperature=1.4,
    top_p=0.3,
    presencePenalty = 0.5,
    countPenalty = 0.5,
    typicalSampling = True,
    state = None
):
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [], # ban the generation of some tokens
                     token_stop = [0, 65532]) # stop generation whenever you see any token here

    ctx = prompt
    
    # gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    # print(f'vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')
    
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    for i in range(int(token_count)):
        out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state)
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)
            
        if typicalSampling:
            token = sample_logits_typical(out, temperature=args.temperature, top_p=args.top_p)
        else:
            token = sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        
        if token in args.token_stop:
            tmp = pipeline.decode(all_tokens[out_last:])
            yield tmp, state
            break
        all_tokens += [token]
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        
        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            yield tmp, state
            out_last = i + 1


# time the evaluation
# starttime = time.time()

# tokens_generated = 0
# run generator and output the result
# print("## Prompt ##")
# print(prompt)

# print("## Normal Sampling ##")
# for token in evaluate(prompt, model_type = model, typicalSampling=False):
#     print(token, end='', flush=True)
#     tokens_generated += 1

# print('\n')


# print("## Typical Sampling ##")
# for token in evaluate(prompt, model_type = model, typicalSampling=True):
#     print(token, end='', flush=True)
#     tokens_generated += 1


# print('\n')

def removeTokens(text):
    return text.replace("<|im_start|>", "").replace("<|im_end|>", "")

# dictionary of tuples of key => (state, expiration)
cachedStates = {}

async def buildPrompt(conversation, model, pipeline):
    first_message = conversation[0]
    fullprompt = f"<|im_start|>{first_message['role']}\n{removeTokens(first_message['content']).strip()}<|im_end|>\n"
    # add system prompt to cache
    cacheKey = hash(fullprompt)
    if cacheKey not in cachedStates.keys():
        out, statea = model.forward(pipeline.encode(fullprompt)[-ctx_limit:], None)
        cachedStates[hash(fullprompt)] = (statea, time.time() + 30) # mod30 secs
    
    for m in conversation[1:-1]:
        if m['role'] == 'user':
            fullprompt += "<|im_start|>user\n" + removeTokens(m['content']).strip() + "<|im_end|>\n"
        elif m['role'] == 'assistant':
            fullprompt += "<|im_start|>assistant\n" + removeTokens(m['content']).strip() + "<|im_end|>\n"
        elif m['role'] == 'system':
            fullprompt += "<|im_start|>system\n" + removeTokens(m['content']).strip() + "<|im_end|>\n"
            
    # hash current prompt to check for cached state
    state = None
    cacheKey = hash(fullprompt)
    if cacheKey in cachedStates.keys():
        state, expiration = cachedStates[cacheKey]
        prompt = ""
        # reset expiration
        cachedStates[cacheKey] = (state, time.time() + 60) # 1 minute
        state = copy.deepcopy(state)
        print("## Using Cached State ##")
    else:
        prompt = fullprompt
    
    # trim message
    last_message = conversation[-1]
            
    prompt += f"<|im_start|>{last_message['role']}\n" + removeTokens(last_message['content']).strip() + "<|im_end|>\n<|im_start|>assistant\n"
    fullprompt += f"<|im_start|>{last_message['role']}\n" + removeTokens(last_message['content']).strip() + "<|im_end|>\n<|im_start|>assistant\n"
    
    return prompt, state, fullprompt
    

async def handleRWKV(conversation, model, pipeline):
    typicalSampling = True
    
    prompt, statee, fullprompt = await buildPrompt(conversation, model, pipeline)
    
    full_response = fullprompt
    response = ""
    async for token, statee in evaluate(prompt, model, pipeline, typicalSampling=typicalSampling, state=statee):
        full_response += token
        response += token
        yield token
        await asyncio.sleep(0.000001)

    
    print ("## Prompt ##")
    print (prompt)
    print ("## Response ##")
    print (response)
    
    print ("##################")
        
    cacheKey = full_response.strip() + "<|im_end|>\n"
    cachedStates[hash(cacheKey)] = (statee, time.time() + 60 * 60) # cache state for 1 hour
    gc.collect()
        

from aiohttp import web
import logging
import aiohttp

async def buildOutputChunk(token):
    object = {
        'object': 'chat.completion.chunk',
        'choices': [
            {
              "delta": {
                "content": token
              },
              "finish_reason": None,
              "index": 0
            }
        ],
    }
    return "data: " + json.dumps(object) + "\n\n"


async def handle(request):
    model, pipeline, index = await getModel()
    try:
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={'Content-Type': 'text/plain'},
        )
        await response.prepare(request)
        # get the request data (json)
        data = await request.json()    
        
        startTime = time.time()
        totalTokens = 0
        
        # run handleRwkv generator and output the result
        async for token in handleRWKV(data['messages'], model, pipeline):
            await response.write((await buildOutputChunk(token)).encode())
            await asyncio.sleep(0.000001)
            totalTokens += 1
            
        unlockModel(index)

        await response.write("data: [DONE]\n\n".encode())
            
        print(f"## Time taken: {time.time() - startTime} ##")
        print(f"## Tokens generated: {totalTokens} ##")
        print(f"## Tokens per second: {totalTokens / (time.time() - startTime)} ##")
        
        await response.write_eof()
        return response
    except OSError:
        print("## Client disconnected ##")
        unlockModel(index)

app = web.Application()
logging.basicConfig(level=logging.DEBUG)
app.add_routes([
    web.post('/v1/chat/completions', handle),
    # web.post('/v1/embeddings', proxy_embedding)
])

def cleanCachedStates():
    while True:
        time.sleep(15) # every minute
        for k in cachedStates.keys():
            if cachedStates[k][1] < time.time():
                del cachedStates[k]
                print("## Cleared A Cached State ##")
                break
            
threading.Thread(target=cleanCachedStates).start()

web.run_app(app, port=9997)