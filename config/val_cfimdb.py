import os

from data.cmifdb.prepare import target_tokens as tt

out_dir = os.path.join('out', 'out-cmifdb')

# data
dataset = 'cmifdb'
num_classes = 2  # 2 classes: negative 0, positive 1

block_size = 512
batch_size = 16

target_tokens = tt
