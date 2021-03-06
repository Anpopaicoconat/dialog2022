import os
from collections import defaultdict, Counter
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import transformers 

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from utilities import collate_class, TextDataset, Metric
    
def predict(x_loader, df, out_name='out.csv'):
    accs = 0
    ns = 0  
    preds = []
    logits = []
    loader = tqdm(x_loader)
    loader.set_description('val')
    for batch in loader:
        batch = {k:batch[k].to(model.device) for k in batch}
        labels = batch.pop('Class')
        with torch.no_grad():
            logit = model(**batch).logits
            pred = logit.argmax(axis=1)
        if labels.size()[1] > 0:
            accs += torch.sum((pred == labels).double())
        logits.append(logit.cpu().numpy())
        preds.append(pred.cpu().numpy())
        ns += len(pred)
        loader.set_postfix({'val_acc': (accs/ns)})
    preds = np.concatenate(preds)
    logits = np.concatenate(logits)
    preds = pd.DataFrame(le.inverse_transform(preds), columns=['Class'])
    logits = pd.DataFrame(logits, columns=le.classes_)
    predicts_pd = pd.concat([df['Id'], preds], axis=1, ignore_index=True)
    predicts_pd.columns=['Id', 'Class']
    logits_pd = pd.concat([df['Id'], logits], axis=1, ignore_index=True)
    predicts_pd.to_csv(out_name, index=False)
    logits_pd.to_csv('logits_'+out_name, index=False)

print_freq = 1
batch_size = 64
max_len = 256


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir='multi/'
train = pd.read_csv(data_dir + 'train.csv')
test = pd.read_csv(data_dir + 'test.csv')
val = pd.read_csv(data_dir + 'val.csv')

le = LabelEncoder()
le.fit(train['Class'].values)
n_classes = len(le.classes_)
models_dir = '/home/posokhov@ad.speechpro.com/projects/models/'
save_dir = 'save/'

#gpt
#tokenizer = transformers.GPT2Tokenizer.from_pretrained('ru_gpt_large', local_files_only=True)#sberbank-ai/rugpt2large
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#model = transformers.GPT2ForSequenceClassification.from_pretrained('ru_gpt_large', num_labels=n_classes, local_files_only=True) #sberbank-ai/rugpt2large
#model.resize_token_embeddings(len(tokenizer))
#model.config.pad_token_id = tokenizer.pad_token_id
#robert
model_name = "ruRoberta-large"
model_path = models_dir+model_name
tokenizer = transformers.RobertaTokenizer.from_pretrained(model_path)
model = model = transformers.RobertaForSequenceClassification.from_pretrained(model_path, num_labels=n_classes, local_files_only=True)

model.to(device)
model.eval()

trainset = TextDataset(train, le=le)
valset = TextDataset(val, le=le)
testset = TextDataset(test, le=le)

collate_fn = collate_class(tokenizer=tokenizer, padding='max_length', max_length=max_len, truncation=True)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
print('test_loader:', len(test_loader), 'val_loader', len(val_loader), 'test_loader:', len(test_loader))

t_total = len(train_loader)

save_path = save_dir+'(add_val)multi-ruRoberta-large'+'.pt'

model.load_state_dict(torch.load(save_path)) 
print('load:', save_path)
last_val_accs = 0.5834

predict(test_loader, test, out_name='roberta-multi.csv')
