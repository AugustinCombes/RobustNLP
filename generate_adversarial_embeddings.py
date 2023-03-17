from os.path import join as pjoin
import os
import pandas as pd
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, pipeline, RobertaTokenizer, RobertaModel

dir_template = "dataset/original/{benchmark}/{model}/{attack}"
target_template = "pca_normal_data/{benchmark}_{model}_{attack}_{type}"

benchmark = 'imdb'
models = ['bert', 'roberta']
attacks = ['bae']

name2ref = {
    'bert': 'textattack/bert-base-uncased',
    'roberta': 'textattack/roberta-base'
}

class EmbeddingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



n_steps = len(models)*len(attacks)*2
current_step = 0

if __name__ == "__main__":
    for model_name in models:
        for attack in attacks:
            path = pjoin(
                dir_template.format(**dict(
                    benchmark=benchmark,
                    model=model_name,
                    attack=attack
                )),
                name2ref[model_name].split('/')[-1] + f"_{attack}.csv"
            )

            try:
                df = pd.read_csv(path)
            except:
                print('Could not find downloaded raw database at', path)
                break
            
            clean_attack = lambda s: s.replace('[', '').replace(']', '')
            df["original_text"] = df.original_text.apply(clean_attack)
            df["perturbed_text"] = df.perturbed_text.apply(clean_attack)
            df = df[['original_text', 'perturbed_text', 'original_score', 'perturbed_score', 'result_type']]

            for type in ['source', 'adv']:
                current_step +=1
                target_database_path = target_template.format(**dict(
                    benchmark=benchmark,
                    model=model_name,
                    attack=attack,
                    type=type
                )) + ".pkl"

                if os.path.exists(target_database_path):
                    print(f'>>>\n    Step {current_step}/{n_steps} : Found existing {type} dataset for {model_name} with {attack} attack\n<<<\n')
                    continue
                
                print(f'>>>\n   Step {current_step}/{n_steps} : Creating {type} dataset for {model_name} with {attack} attack...\n<<<\n')
                
                ref = '-'.join([name2ref[model_name], benchmark])
                if model_name == 'bert':
                    tokenizer = BertTokenizer.from_pretrained(ref, do_lower_case=False, model_max_length=512)
                    model = BertModel.from_pretrained(ref)
                elif model_name == 'roberta':
                    tokenizer = RobertaTokenizer.from_pretrained(ref, do_lower_case=False, model_max_length=512)
                    model = RobertaModel.from_pretrained(ref)
                
                # Generate embeddings using last layer of the model before cls
                res_list = list()
                column = df.original_text.to_numpy().astype(str) if type =='source' else df.perturbed_text.to_numpy().astype(str)
                for text in tqdm(column):
                    with torch.no_grad():
                        text = torch.tensor(tokenizer(text)['input_ids']).reshape(1, -1)[:, :512]
                        res_list.append(
                            model(text).last_hidden_state[:, 0, :].squeeze()
                            )
                
                ds = EmbeddingDataset(torch.stack(res_list, 0))
                dataloader = DataLoader(ds, batch_size=1, shuffle=False)

                with open(target_database_path, 'wb') as f:
                    pickle.dump(ds, f)
                
                del dataloader
                del ds
                del res_list

                print("Done ✓")