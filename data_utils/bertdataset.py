import numpy as np
import sys
from torch.utils.data import Dataset
import torch
from transformers import  BertTokenizer, AutoTokenizer
from .utils import split_url
import os

# Get the path of the current file
file_path = os.path.abspath(__file__)

# Get the directory containing the file
directory_path = os.path.dirname(file_path)

class BERTDataset(Dataset):
    """
    A PyTorch Dataset class for URL classification tasks using transformer models.
    Optimized for malicious URL detection with DomURLs_BERT and URLBert models.

    Args:
        texts (list): List of input URLs.
        labels (list): List of target labels.
        max_length (int): Maximum length of input texts.
        pretrained_path (str): pretrained tokenizer path

    Attributes:
        texts (list): List of input URLs.
        labels (list): List of target labels.
        max_length (int): Maximum length of input texts.
        length (int): Number of samples in the dataset.
        tokenizer: Transformer tokenizer for the specified model.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Retrieves a sample from the dataset at the given index.
    """
    def __init__(self, texts, labels, max_length=128, pretrained_path='amahdaouy/DomURLs_BERT'):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.length = len(self.labels)
        self.preprocess_input = True if pretrained_path=='amahdaouy/DomURLs_BERT' else False
        self.tokenizer = None
        if 'urlbert' in pretrained_path.lower():
            self.tokenizer =  BertTokenizer(f"{directory_path}/urlbert_vocab/vocab.txt")
        else:
            self.tokenizer =  AutoTokenizer.from_pretrained(pretrained_path)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        
        if self.preprocess_input:
            text = split_url(text)
            
        encoded_input = self.tokenizer(
                text,
                max_length = self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        data = {
            "input_ids":input_ids.flatten(),
            "attention_mask": attention_mask.flatten()
        }
        
        return data, label