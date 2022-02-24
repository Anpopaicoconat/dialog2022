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

epoch = 5
print_freq = 1
batch_size = 1
max_len = 256
accumulation_steps = 128
lr = 5e-5


data_dir='bi/' #'multi/'
models_dir = '/home/posokhov@ad.speechpro.com/projects/models/'
model_name = "ruRoberta-large"
model_path = models_dir+model_name
save_dir = 'save/'
load_name = ''
load_path = save_dir+load_name
save_name = 'multi-ruRoberta-large.pt'
save_path = save_dir+save_name

train = pd.read_csv(data_dir + 'train.csv')
test = pd.read_csv(data_dir + 'test.csv')
val = pd.read_csv(data_dir + 'val.csv')

le = LabelEncoder()
le.fit(train['Class'].values)
n_classes = len(le.classes_)

train = TextDataset(train, le=le)
val = TextDataset(val, le=le)
test = TextDataset(test, le=le)

collate_fn = collate_class(padding='max_length', max_length=max_len, truncation=True)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
print('test_loader:', len(test_loader), 'val_loader', len(val_loader), 'test_loader:', len(test_loader), le.classes_)

if model_name == 'rugpt2large':
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(model_path, local_files_only=True) #sberbank-ai/rugpt2large
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = transformers.GPT2ForSequenceClassification.from_pretrained(model_path, num_labels=n_classes, local_files_only=True) #sberbank-ai/rugpt2large
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    
    UNFREEZE_LAST_N = 0
    for param in list(model.parameters())[:-1]:
        param.requires_grad = False
    for i, m in enumerate(model.transformer.h):        
        #Only un-freeze the last n transformer blocks
        if i+1 > len(model.transformer.h) - UNFREEZE_LAST_N:
            print("un-freeze block number {} ".format(i+1))
            for parameter in m.parameters():
                parameter.requires_grad = True 
    for parameter in model.transformer.ln_f.parameters():        
        parameter.requires_grad = True
        
elif model_name == 'ruRoberta-large':
    model_path = models_dir+model_name
    tokenizer = transformers.RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
    model = transformers.RobertaForSequenceClassification.from_pretrained(model_path, num_labels=n_classes, local_files_only=True)
    
elif model_name == 'ruBert-large':
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = transformers.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr = lr, # default is 5e-5, our notebook had 2e-5
                               eps = 1e-8 # default is 1e-8.
                               )

t_total = len(train_loader) // accumulation_steps
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

state_dict = torch.load(load_path)
state_dict.pop('classifier.out_proj.weight')
state_dict.pop('classifier.out_proj.bias')
try:
    model.load_state_dict(state_dict, strict=False) 
    last_val_accs = 0.58532
    print('load:', load_path, 'last_acc:', last_val_accs)
    print('new:', save_path)
except BaseException as e:
    print(e)
    last_val_accs = 0.58532
    print('create new model.' 'last_acc:', last_val_accs)
    print('new:', save_path)

for i_epoch in range(epoch):
    model.train()
    i_batch = 0
    losses = 0
    accs = 0
    ns = 0
    loader = tqdm(train_loader)
    loader.set_description('train')
    for batch in loader:
        i_batch+=1
        batch = {k:batch[k].to(model.device) for k in batch}
        labels = batch.pop('Class')
        out = model(**batch, labels=labels)
        logits = out.logits
        pred = logits.argmax(axis=1).to('cpu').detach()
        accs += sum(pred == labels.to('cpu').detach()).double()
        ns += len(pred)

        loss = out.loss
        losses += loss.to('cpu').detach()
        (loss / accumulation_steps).backward()
        
        if (i_batch % accumulation_steps == 0) or (i_batch == len(loader)):
            optimizer.step()
            optimizer.zero_grad()
            
        if i_batch % (print_freq * accumulation_steps) == 0:
            loader.set_postfix({'loss': (losses/ns).item(), 'acc': (accs/ns).item()})
    scheduler.step()
    torch.cuda.empty_cache()
    
    #val
    model.eval()
    val_i_batch = 0
    val_losses = 0
    val_accs = 0
    val_ns = 0    
    loader = tqdm(val_loader)
    loader.set_description('val')
    for batch in loader:
        val_i_batch+=1
        batch = {k:batch[k].to(model.device) for k in batch}
        labels = batch.pop('Class')

        out = model(**batch) #, labels=labels
        logits = out.logits.to('cpu')
        pred = logits.argmax(axis=1)
        val_accs += torch.sum((pred == labels.to('cpu')).double())
        val_ns += len(pred)
        
        if val_i_batch % (print_freq * accumulation_steps) == 0:
            loader.set_postfix({'val_acc': (val_accs/val_ns).item()})
    print('epoch', i_epoch, '\nloss:', losses/ns, 'acc:', accs/ns, 'val_acc:', val_accs/val_ns, '\n') #'val_loss:', val_losses/val_ns, 
    if val_accs/val_ns > last_val_accs: 
        last_val_accs = val_accs/val_ns
        torch.save(model.state_dict(), save_path)
        print('model saved')
    
    
