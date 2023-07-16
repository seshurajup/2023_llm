import pandas as pd
from string import Template
from pathlib import Path

import os

import warnings
warnings.simplefilter("ignore")

import torch
from transformers import pipeline, AutoTokenizer

from tqdm.notebook import tqdm

data_path = Path('../input/kaggle-llm-science-exam')

from transformers import AutoModelForCausalLM, AutoTokenizer

llm_backbone = 'tiiuae/falcon-7b'

tokenizer = AutoTokenizer.from_pretrained(
    llm_backbone,
    use_fast=False,
    trust_remote_code=True,
    padding_side="left"
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    llm_backbone,
    torch_dtype=torch.float16,
    #load_in_4bit=True,
    device_map="mps",
    trust_remote_code=True,
)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    test = pd.read_csv(data_path / 'test.csv', index_col='id')
    test["answer"] = "A"
else:
    test = pd.read_csv(data_path / 'train.csv', index_col='id')
test.head()

from torch import nn
class Perplexity(nn.Module):
    def __init__(self, reduce: bool = True):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.reduce = reduce

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        perplexity = []
        for i in range(labels.shape[0]):
            perplexity.append(self.loss_fn(shift_logits[i], shift_labels[i]))
        perplexity = torch.stack(perplexity, dim=0)
        #perplexity = torch.exp(perplexity)
        if self.reduce:
            perplexity = torch.mean(perplexity)
        return perplexity 
    
perp = Perplexity()

import numpy as np
def precision_at_k(r, k):
    """Precision at k"""
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k

def MAP_at_3(predictions, true_items):
    """Score is mean average precision at 3"""
    U = len(predictions)
    map_at_3 = 0.0
    for u in range(U):
        user_preds = predictions[u]
        user_true = true_items[u]
        user_results = [1 if item == user_true else 0 for item in user_preds]
        for k in range(min(len(user_preds), 3)):
            map_at_3 += precision_at_k(user_results, k+1) * user_results[k]
    return map_at_3 / U

maps = []
preds = []
for idx, row in tqdm(test.iterrows(), total=len(test)):
        
    
    with torch.no_grad():
        cols = ["A", "B", "C", "D", "E"]
        perps = []
        samples = []
        for col in cols:
            samples.append("<|prompt|>"+row["prompt"]+"</s><|answer|>"+row[col])
        inputs = tokenizer(samples, return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).to("mps")

        output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        output = output.logits
        labels = inputs["input_ids"]
        labels.masked_fill_(~inputs["attention_mask"].bool(), -100)
        for j in range(len(cols)):
            p = perp(output[j].unsqueeze(0), labels[j].unsqueeze(0))
            perps.append(p.detach().cpu())
            
        del inputs
        del labels
        del output
        del p

    perps = np.array(perps)
        
    predictions = [np.array(cols)[np.argsort(perps)]]
    preds.append(predictions)
    tp = [row.answer]
    map = MAP_at_3(predictions, tp)
    maps.append(map)
    print(np.mean(maps))