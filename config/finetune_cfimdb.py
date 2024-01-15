import os
import time

# use default settings
wandb_log = True
wandb_project = 'gpt2-finetune-cfimdb'
wandb_run_name = 'cfimdb' + str(time.time())

output_ckpt = "cfimdb-" + str(time.time())
out_dir = os.path.join('out', 'out-cmifdb')

# data
dataset = 'cmifdb'
num_classes = 2  # 2 classes: negative 0, positive 1

block_size = 512
batch_size = 16
gradient_accumulation_steps = 32

epochs = 20
learning_rate = 5e-5
dropout = 0.2
