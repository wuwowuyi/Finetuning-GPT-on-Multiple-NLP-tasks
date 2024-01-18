import time

from data.sst.prepare import target_tokens as tt

target_tokens = tt

# use default settings
wandb_log = True
output_ckpt = "sst-" + str(time.time())


learning_rate = 5e-5
dropout = 0.2
epochs = 20
batch_size = 64
gradient_accumulation_steps = 32

lam = 0.5
