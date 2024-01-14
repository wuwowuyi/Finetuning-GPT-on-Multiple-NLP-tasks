"""
Adapted from nanoGPT's train.py
"""
import os
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
original_model = 'gpt2'
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

# default settings for sst
dataset = 'sst'
out_dir = os.path.join('out', 'out-sst') # ignored if init_from is not 'resume'
batch_size = 16
num_classes = 5  # sst has 5 classes: negative 0, somewhat negative 1, neutral 2, somewhat positive 3, positive 4
block_size = 64
ckpt_file = 'ckpt.pt'

exec(open('configurator.py').read())  # overrides from command line
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':  # init from checkpoint
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, ckpt_file)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
else:
    raise ValueError("init_from must be either resume or a gpt model, for example, gpt2")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

data_dir = os.path.join('data', dataset)
X = np.fromfile(os.path.join(data_dir, 'val_x.bin'), dtype=np.uint16).reshape(-1, block_size)
Y = np.fromfile(os.path.join(data_dir, 'val_y.bin'), dtype=np.uint16)
pos = np.fromfile(os.path.join(data_dir, 'val_pos.bin'), dtype=np.uint16)  # position of last non-pad token
total, total_batches = X.shape[0], X.shape[0] // batch_size + (1 if X.shape[0] % batch_size > 0 else 0)
def get_batch(ix: int):
    x = torch.as_tensor(X[ix * batch_size: ix * batch_size + batch_size].astype(np.int64), device=device)
    y = torch.as_tensor(Y[ix * batch_size: ix * batch_size + batch_size].astype(np.int64), device=device)
    p = torch.as_tensor(pos[ix * batch_size: ix * batch_size + batch_size].astype(np.int32), device=device)
    return x, y, p


enc = tiktoken.get_encoding(original_model)
target_tokens = torch.as_tensor([enc.encode(str(i))[0] for i in range(num_classes)]).to(device)

# run validation
wrong = 0
with torch.no_grad():
    with ctx:
        for ix in range(total_batches):
            x, y, p = get_batch(ix)
            logits, _ = model(x, last_only=False)
            b = logits.shape[0]
            logits = logits[torch.arange(b), p]  # logits for the last non-pad token
            predicts = torch.argmax(logits[:, target_tokens], dim=-1).squeeze()
            wrong += torch.count_nonzero(predicts - y)

print(f"{ 1 - wrong/total:.3f}")
