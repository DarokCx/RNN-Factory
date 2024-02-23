import torch.nn.functional as F
import numpy as np
from ..tokenizer import world
import jsonstreamer
import json

def sample_logits(out, temperature=1.0, top_p=0.8):
    probs = F.softmax(out.cpu().float(), dim=-1).numpy()
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = float(sorted_probs[np.argmax(cumulative_probs < top_p)])
    probs[probs < cutoff] = 0
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
    probs = probs / np.sum(probs)
    out = np.random.choice(a=len(probs), p=probs)
    return out, False

def sampleJson(ctx, out , temperature=1.0, top_p=0.8):
    
    correct = False
    
    if correct:
        print("Correct and ended")
        return 0
    
    while not correct:
        nexttoken = sample_logits(out, temperature=temperature, top_p=top_p)
        
        
        try:
            output = "{" + world.decode(ctx + [nexttoken])
        except:
            out[nexttoken] = -999
            continue
        
        streamer = jsonstreamer.JSONStreamer()
        
        try:
            streamer.consume(output)
            correct = True
            
        except:
            correct = False
            out[nexttoken] = -999
            
    isComplete = False
    try:
        output = json.loads(output)
        isComplete = True
    except:
        pass
    
    return nexttoken, isComplete
    