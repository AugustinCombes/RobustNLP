import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertModel, BertTokenizer

class IMDB(Dataset):
    def __init__(self, 
                 path="dataset/original/imdb/bert/bae/bert-base-uncased-imdb_bae.csv", 
                 tokenizer_ref='prajjwal1/bert-tiny',
                 mode="normal_only"):

        # load data
        df = pd.read_csv(path)
        clean_attack = lambda s: s.replace('[', '').replace(']', '')
        df["original_text"] = df.original_text.apply(clean_attack)
        df["perturbed_text"] = df.perturbed_text.apply(clean_attack)
        df = df[['original_text', 'perturbed_text', 'original_score', 'perturbed_score', 'result_type', 'ground_truth_output']]
        
        # get tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_ref)
        
        # tokenize raw text
        assert mode in ["adv_mix", "normal_only"]

        tokenizer = BertTokenizer.from_pretrained(tokenizer_ref)

        column = df.original_text if mode == "normal_only" else pd.concat([df.original_text,df[df.result_type=="Successful"].perturbed_text])
        self.token_list = list(column.apply(lambda x: torch.tensor(tokenizer(x, truncation=True, max_length=512)['input_ids'])))
        
        # add label info, original & perturbated score ...
        # self.additional_info = list(df[["original_score", "perturbed_score", "result_type", "ground_truth_output"]].T.to_dict().values())
        if mode == "normal_only":
            self.label = torch.tensor(df["ground_truth_output"].to_numpy().reshape(-1, 1))
        else : # 0 if normal, 1 if adversarial
            self.label = torch.tensor([0 if idx < len(df.original_text) else 1 for idx in range(len(self.token_list))])

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, index):
        return {
            "tokens": self.token_list[index],
            "labels": self.label[index].float().reshape(-1, 1)
        }
    
def collator(batch):
    toks = torch.nn.utils.rnn.pad_sequence(
        [sample["tokens"] for sample in batch],
        padding_value=0,
        batch_first=True
    )
    labs = torch.nn.utils.rnn.pad_sequence(
        [sample["labels"] for sample in batch],
        padding_value=0,
        batch_first=True
    )
    return {
        "tokens":toks, 
        "labels":labs.reshape(-1)
        }

device = 'cpu'

class ClassifierWithAuxiliaryAutoencoder(torch.nn.Module):
    def __init__(self, compressize, abn_dim):
        super(ClassifierWithAuxiliaryAutoencoder, self).__init__()
        self.bert = BertModel.from_pretrained('prajjwal1/bert-tiny')
        
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)

        self.autoencoder = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, compressize, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(compressize, self.bert.config.hidden_size, bias=False),
            torch.nn.Tanh()
        )

        self.abnormality_module = torch.nn.Sequential(
            # torch.nn.Linear(1+2*self.bert.config.hidden_size, abn_dim),
            torch.nn.Linear(self.bert.config.hidden_size, abn_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(abn_dim, abn_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(abn_dim, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, input_ids):
        '''
        Given tokenized input_ids, this layer both
        - finetunes the last layers of DistillBert to the sentiment analysis task
        - trains an auxiliary autoencoder acting over the pooled cls output

        Outputs : dict with
        - 'sentiment' -> probability for the text to be positive without abnormality
        - 'pool' -> pooling embedding of the BERT model
        - 'reconstructed_pool' -> autoencoded pooling embedding of the BERT model
        - 'is_adv' -> probability that the input is adversarial
        '''
        
        pooled = self.bert(input_ids).get('pooler_output')

        #Without abnormality : 
        sentiment = torch.sigmoid(self.classifier(pooled))

        #Autoencoder part :
        reconstructed_pool = self.autoencoder(pooled)

        #Abnormality : 
        mse_pool = torch.square(pooled - reconstructed_pool)
        # pooled_and_mse = torch.concat([pooled, mse_pool], -1)
        # abnormality_score = self.abnormality_module(torch.concat([pooled_and_mse, sentiment], -1))
        
        abnormality_score = self.abnormality_module(mse_pool)

        return {
            'sentiment': sentiment,
            'pool': pooled,
            'reconstructed_pool':reconstructed_pool,
            'is_adv':abnormality_score
        }