import os
from collections import defaultdict, Counter
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import transformers 
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from utilities import collate_class, TextDataset, EnsembleDataset, Metric

class ensemble(pl.LightningModule):
    def __init__(self, inp_size, num_labels, lr):
        super().__init__()
        self.linear1 = torch.nn.Linear(inp_size, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, num_labels)
        self.softmax = torch.nn.Softmax()

        self.metric = Metric()
        self.learning_rate = lr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        labels = batch.pop('Class')
        logits = self(batch)
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(axis=1)
        accuracy = torch.mean((predictions == labels).double())
        self.metric.store(loss=loss.item(), accuracy=accuracy.item())
        if batch_idx % 100: # every 100 batches - log metrics (mean of last 100 batches)
            for k,v in self.metric.log():
                self.log(f'train/{k}', v)
            self.metric.reset()
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop('Class')
        logits = self(batch)
        loss = F.cross_entropy(logits, labels)
        self.log('val/loss', loss)
        predictions = logits.argmax(axis=1)
        self.log('val/accuracy', torch.mean((predictions == labels).double()))
lr = 0.1
epoch = 10
train_path = ''
test_path = ''
val_path = ''
train = pd.read_csv(train_path)
test = pd.read_csv(test_path + 'test.csv')
val = pd.read_csv(val_path + 'val.csv')

le = LabelEncoder()
le.fit(train['Class'].values)
n_classes = len(le.classes_)
inp_size = len(train[0])

train = EnsembleDataset(train, le=le)
val = EnsembleDataset(val, le=le)
test = EnsembleDataset(test, le=le)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
print('test_loader:', len(test_loader), 'val_loader', len(val_loader), 'test_loader:', len(test_loader), le.classes_)

model = ensemble(inp_size=inp_size, num_labels=n_classes, lr=lr)
trainer = pl.Trainer(max_epochs=epoch)
trainer.fit(model, train_dataloaders=train_loader)
