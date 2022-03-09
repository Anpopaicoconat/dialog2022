# dialog2022
pretrained model: sberbank-ai/ruRoberta-large
optimizer adamw (eps = 1e-8)
linear_schedule (num_warmup_steps=100, lr=2e-5 , lr decay=1.5)
max_len = 256
batch_size = 1, accumulation_steps = 32
4ep for train & 1ep for tune on val dataset
