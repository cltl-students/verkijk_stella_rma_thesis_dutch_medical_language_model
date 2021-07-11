"""
@Author StellaVerkijk
This is a pytorch dataloader for a Roberta-based model. 
This dataloader was based on the dataloader from Pieter Delobelle for RobBERT but was adapted to be able to load bigger files. 
"""

import json
import os
import pickle
import random
import time
from typing import Dict, List, Optional
from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset

from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM, RobertaTokenizer
from tokenizers.processors import BertProcessing, RobertaProcessing
from tokenizers import ByteLevelBPETokenizer


class LineByLineTextDatasetRobbert(Dataset):
    def __init__(self, tokenizer, file_paths: list, block_size=512):

        self.block_size = block_size
        
        self.tokenizer = tokenizer
        
        self.tokenizer.post_processor = RobertaProcessing(
            ("</s>", self.tokenizer.convert_tokens_to_ids("</s>")),
            ("<s>", self.tokenizer.convert_tokens_to_ids("<s>")),
        )
        
        self.examples = []
        for file_path in file_paths:
            print("🔥", file_path)
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    if len(line) > 0 and not line.isspace():
                        self.examples.append(line.strip('\n'))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.tokenizer.encode(self.examples[i])[: self.block_size - 2], dtype=torch.long)
    
    

