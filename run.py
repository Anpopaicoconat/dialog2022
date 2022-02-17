import os
from collections import defaultdict, Counter
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import transformers 

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

class collate_class():
    def __init__(self, padding='max_length', max_length=256, truncation=True):
        self.padding = padding
        self.max_length = max_length
        self.truncation = truncation

    def __call__(self, input_data):
        texts, labels = zip(*input_data)
        labels = torch.LongTensor(labels)
        inputs = tokenizer(texts, return_tensors='pt', padding=self.padding, max_length=self.max_length, truncation=self.truncation)
        inputs['Class'] = labels
        return inputs

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, sort=False, le=None):
        super().__init__()
        self.texts = data['Text'].values
        if 'Class' in data.columns: # если есть разметка
            assert not data['Class'].isnull().any(), "Some labels are null"
            if le is not None:
                self.labels = le.transform(data['Class'])
            else:
                self.labels = data['Class'].values
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        if hasattr(self, 'labels'):
            return self.texts[idx], self.labels[idx]
        else:
            return self.texts[idx], []

class Metric: # metric class for storing metrics (accuracy, loss)
    def __init__(self):
        self.storage = defaultdict(list)
    
    def store(self, **kwargs):
        for key in kwargs:
            self.storage[key].append(kwargs[key])
            
    def reset(self):
        self.storage.clear()
        
    def log(self):
        for key in self.storage:
            self.storage[key] = np.mean(self.storage[key])
        return self.storage.items()

epoch = 3
print_freq = 500
batch_size = 8
max_len = 256
accumulation_steps = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir=''
train = pd.read_csv(data_dir + 'train.csv')
test = pd.read_csv(data_dir + 'test.csv')
val = pd.read_csv(data_dir + 'val.csv')

le = LabelEncoder()
le.fit(train['Class'].values)
n_classes = len(le.classes_)

tokenizer = transformers.GPT2Tokenizer.from_pretrained('ru_gpt_large', local_files_only=True)#sberbank-ai/rugpt2large
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = transformers.GPT2ForSequenceClassification.from_pretrained('ru_gpt_large', num_labels=n_classes, local_files_only=True) #sberbank-ai/rugpt2large
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)

lr = 2e-5
#for param in list(model.parameters())[:-1]:
#    param.requires_grad = False

optimizer = transformers.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr = lr, # default is 5e-5, our notebook had 2e-5
                               eps = 1e-8 # default is 1e-8.
                               )

train = TextDataset(train, le=le)
val = TextDataset(val, le=le)
test = TextDataset(test, le=le)

collate_fn = collate_class(padding='max_length', max_length=max_len, truncation=True)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model.load_state_dict(torch.load('ru_gpt_bi-classifier.pt')) 
for i_epoch in range(epoch):
    i_batch = 0
    losses = 0
    accs = 0
    ns = 0
    # for batch in tqdm(train_loader):
#         i_batch+=1
#         batch = {k:batch[k].to(model.device) for k in batch}
#         labels = batch.pop('Class')
#         out = model(**batch, labels=labels)
#         logits = out.logits
#         pred = logits.argmax(axis=1)
#         accs += torch.sum((pred == labels).double())
#         ns += len(pred)

#         loss = out.loss
#         losses += loss
#         (loss / accumulation_steps).backward()
        
#         if (i_batch % accumulation_steps == 0) or (i_batch == len(train_loader)):
#             optimizer.step()
#             optimizer.zero_grad()
            
#         if i_batch % print_freq == 0:
#             print('loss:', losses/ns, 'acc:', accs/ns)

        
#     print('\n\nepoch', i_epoch, '\nloss:', losses/ns, 'acc:', accs/ns, '\n\n')
#     torch.save(model.state_dict(), "ru_gpt_bi-classifier.pt")
    torch.cuda.empty_cache()
    #val
    val_i_batch = 0
    val_losses = 0
    val_accs = 0
    val_ns = 0    
    for batch in tqdm(val_loader):
        val_i_batch+=1
        labels = batch.pop('Class')
        batch = {k:batch[k].to(model.device) for k in batch}

        out = model(**batch, labels=labels)
        logits = out.logits.to('cpu')
        pred = logits.argmax(axis=1)
        val_accs += torch.sum((pred == labels).double())
        val_ns += len(pred)

        loss = out.loss.to('cpu')
        val_losses += loss
        
        if val_i_batch % print_freq == 0:
            print('val_loss:', val_losses/val_ns, 'val_acc:', val_accs/val_ns)
    print('='*10, '\n\nepoch', i_epoch, '\nloss:', losses/ns, 'acc:', accs/ns, 'val_loss:', val_losses/val_ns, 'val_acc:', val_accs/val_ns, '\n\n', '='*10)
    
    
