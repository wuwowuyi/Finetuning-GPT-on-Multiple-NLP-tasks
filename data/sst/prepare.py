import os

import pandas as pd
import tiktoken
import numpy as np


train_file = os.path.join(os.path.dirname(__file__), 'ids-sst-train.csv')
val_file = os.path.join(os.path.dirname(__file__), 'ids-sst-dev.csv')

# each data point is in the format: <sentence> sentiment: <label> <pad_token>...
# For example: You 'll probably love it . sentiment: 4 <|endoftext|>...
# where <|endoftext|> is the pad token.
block_size = 64  # longest sequence length
enc = tiktoken.get_encoding("gpt2")
pad_token = enc.eot_token  # pad by eot_token '<|endoftext|>'
q = enc.encode_ordinary(' sentiment: ')

def pad(ids):
    if len(ids) < block_size:
        ids += [pad_token] * (block_size - len(ids))
    return ids[-block_size:]  # crop long sequence

# train data
df = pd.read_csv(train_file, sep=r'\t', header=0, index_col=0, dtype=object)
token_lists = enc.encode_ordinary_batch(df["sentence"])
labels = enc.encode_ordinary_batch(df["sentiment"])
X = np.stack(
    [pad(tokens + q) for tokens in token_lists]
).astype(np.uint16)
Y = np.stack(
    [pad(tokens[1:] + q + label) for tokens, label in zip(token_lists, labels)]
).astype(np.uint16)
X.tofile(os.path.join(os.path.dirname(__file__), f'train_x.bin'))
Y.tofile(os.path.join(os.path.dirname(__file__), f'train_y.bin'))
lengths = [len(tokens) for tokens in token_lists]
print(f"train has {sum(lengths):,} tokens")

# dev data. Use half for validation, half for test.
df = pd.read_csv(val_file, sep=r'\t', header=0, index_col=0, dtype=object)
token_lists = enc.encode_ordinary_batch(df["sentence"])
labels = enc.encode_ordinary_batch(df["sentiment"])
X = np.stack(
    [pad(tokens + q) for tokens in token_lists]
).astype(np.uint16)
half = int(len(X) * 0.5)
X_val, X_test = X[:half], X[half:]
Y_val = np.stack(
    [pad(tokens[1:] + q + label) for tokens, label in zip(token_lists[:half], labels[:half])]
).astype(np.uint16)
Y_test = np.asarray(df['sentiment'][half:], dtype=np.uint16)
# store the position of the last non-pad token. max position is block_size - 1
pos_test = np.asarray([min(len(tokens) + len(q), block_size) - 1 for tokens in token_lists[half:]], dtype=np.uint16)

X_val.tofile(os.path.join(os.path.dirname(__file__), f'val_x.bin'))
Y_val.tofile(os.path.join(os.path.dirname(__file__), f'val_y.bin'))
X_test.tofile(os.path.join(os.path.dirname(__file__), f'test_x.bin'))
Y_test.tofile(os.path.join(os.path.dirname(__file__), f'test_y.bin'))
pos_test.tofile(os.path.join(os.path.dirname(__file__), f'test_pos.bin'))

lengths = [len(tokens) for tokens in token_lists]
print(f"val and test has {sum(lengths):,} tokens")

# train has 191,641 tokens
# val and test have 24,946 tokens


