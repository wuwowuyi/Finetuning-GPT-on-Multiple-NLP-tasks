import os

import pandas as pd
import tiktoken
import numpy as np


train_file = os.path.join(os.path.dirname(__file__), 'ids-cfimdb-train.csv')
val_file = os.path.join(os.path.dirname(__file__), 'ids-cfimdb-dev.csv')

# each data point is in the format: <sentence> sentiment: <label> <pad_token>...
# For example: You 'll probably love it . sentiment: 0 <|endoftext|>...
# where <|endoftext|> is the pad token.
block_size = 512  # longest sequence length
enc = tiktoken.get_encoding("gpt2")
pad_token = enc.eot_token  # pad by eot_token '<|endoftext|>'
q = enc.encode_ordinary(' sentiment: ')

def pad(ids):
    if len(ids) < block_size:
        ids += [pad_token] * (block_size - len(ids))
    return ids[-block_size:]  # crop long sequence

# train data
for split, f in zip(("train", "val"), (train_file, val_file)):
    df = pd.read_csv(f, sep=r'\t', header=0, index_col=0)
    token_lists = enc.encode_ordinary_batch(df["sentence"])
    X = np.stack(
        [pad(tokens + q) for tokens in token_lists]
    ).astype(np.uint16)
    Y = np.asarray(df['sentiment'], dtype=np.uint16)
    pos = np.asarray([min(len(tokens) + len(q), block_size) - 1
                      for tokens in token_lists], dtype=np.uint16)
    X.tofile(os.path.join(os.path.dirname(__file__), f'{split}_x.bin'))
    Y.tofile(os.path.join(os.path.dirname(__file__), f'{split}_y.bin'))
    pos.tofile(os.path.join(os.path.dirname(__file__), f'{split}_pos.bin'))
    lengths = [len(tokens) for tokens in token_lists]
    print(f"{split} has {sum(lengths):,} tokens")

# train has 371,408 tokens, 1,707 data points
# val has 55,186 tokens, 245 data points
