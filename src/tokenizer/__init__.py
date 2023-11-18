from .tokenizer import RWKV_TOKENIZER, neox
from .raccoon import RaccTRIE_TOKENIZER
import os


fname = "rwkv_vocab_v20230424.txt"
world = RWKV_TOKENIZER(os.path.join(os.path.dirname(__file__), fname))

neox = neox

racfname = "rwkv_vocab_v20230922_chatml.txt"
racoon = RaccTRIE_TOKENIZER(os.path.join(os.path.dirname(__file__), racfname))
