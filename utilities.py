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
    def __init__(self, tokenizer, padding='max_length', max_length=256, truncation=True):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.truncation = truncation

    def __call__(self, input_data):
        texts, labels = zip(*input_data)
        labels = torch.LongTensor(labels)
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, max_length=self.max_length, truncation=True) #padding=self.padding, truncation=self.truncation
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
